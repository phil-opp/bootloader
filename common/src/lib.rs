#![cfg_attr(not(test), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]

use crate::legacy_memory_region::{LegacyFrameAllocator, LegacyMemoryRegion};
use crate::x86_bridge::{IdentityMappedAccess, X86FrameAllocator, X86PageSize, X86PageTable};
use bootloader_api::{
    BootInfo, BootloaderConfig,
    config::Mapping,
    info::{FrameBuffer, FrameBufferInfo, MemoryRegion, TlsTemplate},
};
use bootloader_boot_config::{BootConfig, LevelFilter};
use core::{alloc::Layout, arch::asm, mem::MaybeUninit, slice};
use kernel_elf_loader::{
    self as kel, AddressSpace,
    loader::{KernelPlacement, Loader, RegionPlacement},
};
use x86_64::{
    PhysAddr, VirtAddr,
    structures::paging::{
        FrameAllocator, Mapper, OffsetPageTable, Page, PageTableFlags, PageTableIndex,
        PhysFrame, page_table::PageTableLevel,
    },
};

/// Provides a function to gather entropy and build a RNG.
mod entropy;
/// Provides a type that logs output as text to pixel-based framebuffers.
pub mod framebuffer;
mod gdt;
/// Provides a frame allocator based on a BIOS or UEFI memory map.
pub mod legacy_memory_region;
/// Provides a logger that logs output as text in various formats.
pub mod logger;
/// Provides a type that logs output as text to a Serial Being port.
pub mod serial;
/// Bridge between x86_64 types and kernel-elf-loader traits.
pub mod x86_bridge;

const PAGE_SIZE: u64 = 4096;

/// Initialize a text-based logger using the given pixel-based framebuffer as output.
pub fn init_logger(
    framebuffer: &'static mut [u8],
    info: FrameBufferInfo,
    log_level: LevelFilter,
    frame_buffer_logger_status: bool,
    serial_logger_status: bool,
) {
    let logger = logger::LOGGER.get_or_init(move || {
        logger::LockedLogger::new(
            framebuffer,
            info,
            frame_buffer_logger_status,
            serial_logger_status,
        )
    });
    log::set_logger(logger).expect("logger already set");
    log::set_max_level(convert_level(log_level));
    log::info!("Framebuffer info: {:?}", info);
}

fn convert_level(level: LevelFilter) -> log::LevelFilter {
    match level {
        LevelFilter::Off => log::LevelFilter::Off,
        LevelFilter::Error => log::LevelFilter::Error,
        LevelFilter::Warn => log::LevelFilter::Warn,
        LevelFilter::Info => log::LevelFilter::Info,
        LevelFilter::Debug => log::LevelFilter::Debug,
        LevelFilter::Trace => log::LevelFilter::Trace,
    }
}

/// Required system information that should be queried from the BIOS or UEFI firmware.
#[derive(Debug, Copy, Clone)]
pub struct SystemInfo {
    /// Information about the (still unmapped) framebuffer.
    pub framebuffer: Option<RawFrameBufferInfo>,
    /// Address of the _Root System Description Pointer_ structure of the ACPI standard.
    pub rsdp_addr: Option<PhysAddr>,
    pub ramdisk_addr: Option<u64>,
    pub ramdisk_len: u64,
}

/// The physical address of the framebuffer and information about the framebuffer.
#[derive(Debug, Copy, Clone)]
pub struct RawFrameBufferInfo {
    /// Start address of the pixel-based framebuffer.
    pub addr: PhysAddr,
    /// Information about the framebuffer, including layout and pixel format.
    pub info: FrameBufferInfo,
}

/// Parsed kernel ELF with embedded bootloader configuration.
pub struct Kernel<'a> {
    /// The bootloader configuration extracted from the kernel ELF.
    pub config: BootloaderConfig,
    /// Raw kernel ELF bytes.
    pub elf_bytes: &'a [u8],
}

impl<'a> Kernel<'a> {
    pub fn parse(kernel_slice: &'a [u8]) -> Self {
        // Use the `elf` crate to extract the .bootloader-config section.
        let elf = elf::ElfBytes::<elf::endian::AnyEndian>::minimal_parse(kernel_slice)
            .expect("failed to parse kernel ELF");
        let config = {
            let section = elf
                .section_header_by_name(".bootloader-config")
                .expect("failed to find .bootloader-config section")
                .expect("kernel must be compiled against bootloader_api");
            let (raw, _compression) = elf
                .section_data(&section)
                .expect("failed to read .bootloader-config data");
            BootloaderConfig::deserialize(raw)
                .expect("kernel was compiled with incompatible bootloader_api version")
        };
        Kernel {
            config,
            elf_bytes: kernel_slice,
        }
    }
}

/// Convert a `bootloader_api::config::Mapping` to a `kernel_elf_loader::RegionPlacement`.
fn mapping_to_placement(mapping: Mapping) -> RegionPlacement {
    match mapping {
        Mapping::Dynamic => RegionPlacement::Auto,
        Mapping::FixedAddress(addr) => RegionPlacement::Fixed(kel::VirtAddr::new(addr)),
    }
}

/// Convert a `bootloader_api::config::Mapping` to a `kernel_elf_loader::KernelPlacement`.
fn mapping_to_kernel_placement(mapping: Mapping) -> KernelPlacement {
    match mapping {
        Mapping::Dynamic => KernelPlacement::Auto,
        Mapping::FixedAddress(addr) => KernelPlacement::Fixed(kel::VirtAddr::new(addr)),
    }
}

/// Loads the kernel ELF executable into memory and switches to it.
///
/// This function is a convenience function that first calls [`set_up_mappings`], then
/// [`create_boot_info`], and finally [`switch_to_kernel`]. The given arguments are passed
/// directly to these functions, so see their docs for more info.
pub fn load_and_switch_to_kernel<I, D>(
    kernel: Kernel,
    boot_config: BootConfig,
    mut frame_allocator: LegacyFrameAllocator<I, D>,
    mut page_tables: PageTables,
    system_info: SystemInfo,
) -> !
where
    I: ExactSizeIterator<Item = D> + Clone,
    D: LegacyMemoryRegion,
{
    let config = kernel.config;
    let mut mappings = set_up_mappings(
        kernel,
        &mut frame_allocator,
        &mut page_tables,
        system_info.framebuffer.as_ref(),
        &config,
        &system_info,
    );
    let boot_info = create_boot_info(
        &config,
        &boot_config,
        frame_allocator,
        &mut page_tables,
        &mut mappings,
        system_info,
    );
    switch_to_kernel(page_tables, mappings, boot_info);
}

/// Sets up mappings for the kernel, kernel stack, framebuffer, ramdisk, and physical memory.
///
/// Uses `kernel_elf_loader::Loader` for the kernel ELF loading and region mapping,
/// while x86_64-specific operations (context switch identity mapping, GDT, recursive
/// page table) are done directly.
pub fn set_up_mappings<I, D>(
    kernel: Kernel,
    frame_allocator: &mut LegacyFrameAllocator<I, D>,
    page_tables: &mut PageTables,
    framebuffer: Option<&RawFrameBufferInfo>,
    _config: &BootloaderConfig,
    system_info: &SystemInfo,
) -> Mappings
where
    I: ExactSizeIterator<Item = D> + Clone,
    D: LegacyMemoryRegion,
{
    // Enable support for the no-execute bit in page tables.
    enable_nxe_bit();
    // Make the kernel respect the write-protection bits even when in ring 0 by default
    enable_write_protect_bit();

    let config = kernel.config;
    let kernel_slice_start = PhysAddr::new(kernel.elf_bytes.as_ptr() as u64);
    let kernel_slice_len = u64::try_from(kernel.elf_bytes.len()).unwrap();

    // Build RNG for ASLR if enabled.
    let mut rng = if config.mappings.aslr {
        Some(entropy::build_rng())
    } else {
        None
    };

    // Query frame allocator info before borrowing it through the bridge.
    let max_phys = frame_allocator.max_phys_addr();
    let frame_allocator_len = frame_allocator.len();

    // Create the kernel-elf-loader bridge types.
    let phys_mem = IdentityMappedAccess;
    let mut x86_pt = X86PageTable::new(&mut page_tables.kernel);
    let mut x86_alloc = X86FrameAllocator::new(frame_allocator);

    let rng_ref: Option<&mut dyn rand_core::RngCore> = match rng {
        Some(ref mut r) => Some(r),
        None => None,
    };

    let mut loader = Loader::<X86PageSize>::new(&mut x86_pt, &mut x86_alloc, &phys_mem, rng_ref);
    // Mark identity-mapped physical memory as used.
    // We must round up to an L4-entry boundary (512 GiB) because the bootloader
    // identity-maps physical memory using 2MiB pages, which populates L2/L3
    // page table entries. The boot info is mapped into both the kernel and
    // bootloader page tables, so its virtual address must not share any page
    // table entries with the identity mapping.
    let l4_entry_size: u64 = 4096 * 512 * 512 * 512; // 512 GiB
    let identity_map_end = ((max_phys.as_u64() + l4_entry_size - 1) / l4_entry_size) * l4_entry_size;
    loader.mark_used(kel::VirtAddr::new(0), identity_map_end);

    // Mark framebuffer physical address range as used.
    if let Some(fb) = framebuffer {
        loader.mark_used(
            kel::VirtAddr::new(fb.addr.as_u64()),
            fb.info.byte_len as u64,
        );
    }

    // Mark fixed-address config ranges as used.
    if let Some(Mapping::FixedAddress(addr)) = config.mappings.physical_memory {
        loader.mark_used(kel::VirtAddr::new(addr), max_phys.as_u64());
    }
    if let Some(Mapping::FixedAddress(addr)) = config.mappings.page_table_recursive {
        // A recursive mapping occupies a full L4 entry (512 GiB).
        let l4_entry_size: u64 = 4096 * 512 * 512 * 512;
        let aligned = addr / l4_entry_size * l4_entry_size;
        loader.mark_used(kel::VirtAddr::new(aligned), l4_entry_size);
    }
    if let Mapping::FixedAddress(addr) = config.mappings.kernel_stack {
        loader.mark_used(kel::VirtAddr::new(addr), config.kernel_stack_size + PAGE_SIZE);
    }
    if let Mapping::FixedAddress(addr) = config.mappings.boot_info {
        let boot_info_layout = Layout::new::<BootInfo>();
        let regions = frame_allocator_len + 1;
        let memory_regions_layout = Layout::array::<MemoryRegion>(regions).unwrap();
        let (combined, _) = boot_info_layout.extend(memory_regions_layout).unwrap();
        loader.mark_used(kel::VirtAddr::new(addr), combined.size() as u64);
    }
    if let Mapping::FixedAddress(addr) = config.mappings.framebuffer {
        if let Some(fb) = framebuffer {
            loader.mark_used(kel::VirtAddr::new(addr), fb.info.byte_len as u64);
        }
    }

    // Mark dynamic range boundaries.
    if let Some(dynamic_range_start) = config.mappings.dynamic_range_start {
        // Everything before this is unusable for dynamic allocation.
        if dynamic_range_start > 0 {
            loader.mark_used(kel::VirtAddr::new(0), dynamic_range_start);
        }
    }
    if let Some(dynamic_range_end) = config.mappings.dynamic_range_end {
        // Everything after this is unusable.
        let end = dynamic_range_end;
        let remaining = 0xFFFF_FFFF_FFFF_0000u64.saturating_sub(end);
        if remaining > 0 {
            loader.mark_used(kel::VirtAddr::new(end), remaining);
        }
    }

    // Load the kernel ELF.
    let loaded = loader
        .load_kernel(
            kernel.elf_bytes,
            kel::PhysAddr::new(kernel_slice_start.as_u64()),
            mapping_to_kernel_placement(config.mappings.kernel_base),
            false, // no huge pages for kernel segments
        )
        .expect("failed to load kernel ELF");

    log::info!("Entry point at: {:#x}", loaded.entry_point.as_u64());

    // Convert TLS template from kernel-elf-loader type to bootloader_api type.
    let tls_template = loaded.tls_template.map(|tls| TlsTemplate {
        start_addr: tls.start_addr.as_u64(),
        mem_size: tls.mem_size,
        file_size: tls.file_size,
    });

    let entry_point = VirtAddr::new(loaded.entry_point.as_u64());
    let kernel_image_offset = VirtAddr::new(loaded.load_base as u64);

    // Create kernel stack: guard page + stack pages.
    // We need them contiguous: guard page first, then stack pages.
    let stack_flags = kel::PageFlags {
        writable: true,
        executable: false,
    };
    let guard_and_stack_size = PAGE_SIZE + config.kernel_stack_size;

    let guard_page_addr = match config.mappings.kernel_stack {
        Mapping::Dynamic => {
            // Find one contiguous region for guard + stack.
            let region_start = loader
                .address_space_mut()
                .find_free(guard_and_stack_size, PAGE_SIZE, None)
                .expect("failed to find free region for kernel stack");
            region_start
        }
        Mapping::FixedAddress(addr) => kel::VirtAddr::new(addr),
    };

    // Add guard page (unmapped) at the start.
    loader
        .add_guard_page(RegionPlacement::Fixed(guard_page_addr))
        .expect("failed to add stack guard page");

    // Allocate and map stack pages right after the guard page.
    let stack_start_addr = kel::VirtAddr::new(guard_page_addr.as_u64() + PAGE_SIZE);
    loader
        .allocate_and_map(
            config.kernel_stack_size,
            stack_flags,
            RegionPlacement::Fixed(stack_start_addr),
        )
        .expect("failed to allocate kernel stack");

    let stack_bottom = VirtAddr::new(stack_start_addr.as_u64());
    let stack_end_addr = VirtAddr::new(stack_start_addr.as_u64() + config.kernel_stack_size);
    let stack_top = stack_end_addr.align_down(16u8);

    // Map framebuffer.
    let framebuffer_virt_addr = if let Some(fb) = framebuffer {
        log::info!("Map framebuffer");
        let fb_size = fb.info.byte_len as u64;
        let fb_flags = kel::PageFlags {
            writable: true,
            executable: false,
        };
        let virt = loader
            .map_physical_region(
                kel::PhysAddr::new(fb.addr.as_u64()),
                fb_size,
                fb_flags,
                mapping_to_placement(config.mappings.framebuffer),
                false,
            )
            .expect("failed to map framebuffer");
        Some(VirtAddr::new(virt.as_u64()))
    } else {
        None
    };

    // Map ramdisk.
    let ramdisk_slice_len = system_info.ramdisk_len;
    let ramdisk_slice_phys_start = system_info.ramdisk_addr.map(PhysAddr::new);
    let ramdisk_slice_start = if let Some(phys_addr) = system_info.ramdisk_addr {
        let ramdisk_flags = kel::PageFlags {
            writable: true,
            executable: false,
        };
        let virt = loader
            .map_physical_region(
                kel::PhysAddr::new(phys_addr),
                system_info.ramdisk_len,
                ramdisk_flags,
                mapping_to_placement(config.mappings.ramdisk_memory),
                false,
            )
            .expect("failed to map ramdisk");
        Some(VirtAddr::new(virt.as_u64()))
    } else {
        None
    };

    // Map physical memory.
    let physical_memory_offset = if let Some(mapping) = config.mappings.physical_memory {
        log::info!("Map physical memory");
        let phys_mem_flags = kel::PageFlags {
            writable: true,
            executable: false,
        };
        let size = max_phys.as_u64();
        let virt = loader
            .map_physical_region(
                kel::PhysAddr::new(0),
                size,
                phys_mem_flags,
                mapping_to_placement(mapping),
                true, // use huge pages for physical memory mapping
            )
            .expect("failed to map physical memory");
        Some(VirtAddr::new(virt.as_u64()))
    } else {
        None
    };

    // Extract the address space before dropping the loader.
    let address_space = {
        // We need to move the address space out. Since Loader borrows everything,
        // we need to extract it from the internal state.
        // Actually we can't easily move it out. Instead, let's drop the loader
        // and do the remaining x86-specific ops directly on the page table.
        // But first get the address space.
        core::mem::replace(loader.address_space_mut(), AddressSpace::new())
    };

    // Drop the loader — remaining operations are x86_64-specific and use
    // the page table + frame allocator directly.
    drop(loader);

    let kernel_page_table = &mut page_tables.kernel;

    // Identity-map context switch function, so that we don't get an immediate pagefault
    // after switching the active page table.
    let context_switch_function = PhysAddr::new(context_switch as *const () as u64);
    let context_switch_function_start_frame: PhysFrame =
        PhysFrame::containing_address(context_switch_function);
    for frame in PhysFrame::range_inclusive(
        context_switch_function_start_frame,
        context_switch_function_start_frame + 1,
    ) {
        let page = Page::containing_address(VirtAddr::new(frame.start_address().as_u64()));
        match unsafe {
            kernel_page_table.map_to_with_table_flags(
                page,
                frame,
                PageTableFlags::PRESENT,
                PageTableFlags::PRESENT | PageTableFlags::WRITABLE,
                frame_allocator,
            )
        } {
            Ok(tlb) => tlb.flush(),
            Err(err) => panic!("failed to identity map frame {:?}: {:?}", frame, err),
        }
    }

    // Create, load, and identity-map GDT (required for working `iretq`).
    let gdt_frame = frame_allocator
        .allocate_frame()
        .expect("failed to allocate GDT frame");
    gdt::create_and_load(gdt_frame);
    let gdt_page = Page::containing_address(VirtAddr::new(gdt_frame.start_address().as_u64()));
    match unsafe {
        kernel_page_table.map_to_with_table_flags(
            gdt_page,
            gdt_frame,
            PageTableFlags::PRESENT,
            PageTableFlags::PRESENT | PageTableFlags::WRITABLE,
            frame_allocator,
        )
    } {
        Ok(tlb) => tlb.flush(),
        Err(err) => panic!("failed to identity map frame {:?}: {:?}", gdt_frame, err),
    }

    // Set up recursive page table mapping.
    let recursive_index = if let Some(mapping) = config.mappings.page_table_recursive {
        log::info!("Map page table recursively");
        let index = match mapping {
            Mapping::Dynamic => {
                // Find a free L4 index. Use the address space to find an aligned region.
                // Each L4 entry covers 512 GiB.
                // We just need one free L4 entry. Pick a simple approach:
                // scan the L4 table for an unused entry.
                let l4_table = kernel_page_table.level_4_table_mut();
                let mut found = None;
                for i in 0u16..512 {
                    let idx = PageTableIndex::new(i);
                    if l4_table[idx].is_unused() {
                        found = Some(idx);
                        break;
                    }
                }
                found.expect("no free level 4 entry for recursive mapping")
            }
            Mapping::FixedAddress(offset) => {
                let offset = VirtAddr::new(offset);
                let table_level = PageTableLevel::Four;
                if !offset.is_aligned(table_level.entry_address_space_alignment()) {
                    panic!(
                        "Offset for recursive mapping must be properly aligned (must be \
                        a multiple of {:#x})",
                        table_level.entry_address_space_alignment()
                    );
                }
                offset.p4_index()
            }
        };

        let entry = &mut kernel_page_table.level_4_table_mut()[index];
        if !entry.is_unused() {
            panic!(
                "Could not set up recursive mapping: index {} already in use",
                u16::from(index)
            );
        }
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE;
        entry.set_frame(page_tables.kernel_level_4_frame, flags);

        Some(index)
    } else {
        None
    };

    Mappings {
        framebuffer: framebuffer_virt_addr,
        entry_point,
        stack_bottom,
        stack_top,
        address_space,
        physical_memory_offset,
        recursive_index,
        tls_template,

        kernel_slice_start,
        kernel_slice_len,
        kernel_image_offset,

        ramdisk_slice_phys_start,
        ramdisk_slice_start,
        ramdisk_slice_len,
    }
}

/// Contains the addresses of all memory mappings set up by [`set_up_mappings`].
pub struct Mappings {
    /// The entry point address of the kernel.
    pub entry_point: VirtAddr,
    pub stack_bottom: VirtAddr,
    /// The (exclusive) end address of the kernel stack.
    pub stack_top: VirtAddr,
    /// Tracks used virtual address ranges for finding free regions.
    pub address_space: AddressSpace,
    /// The start address of the framebuffer, if any.
    pub framebuffer: Option<VirtAddr>,
    /// The start address of the physical memory mapping, if enabled.
    pub physical_memory_offset: Option<VirtAddr>,
    /// The level 4 page table index of the recursive mapping, if enabled.
    pub recursive_index: Option<PageTableIndex>,
    /// The thread local storage template of the kernel executable, if it contains one.
    pub tls_template: Option<TlsTemplate>,

    /// Start address of the kernel slice allocation in memory.
    pub kernel_slice_start: PhysAddr,
    /// Size of the kernel slice allocation in memory.
    pub kernel_slice_len: u64,
    /// Relocation offset of the kernel image in virtual memory.
    pub kernel_image_offset: VirtAddr,
    pub ramdisk_slice_phys_start: Option<PhysAddr>,
    pub ramdisk_slice_start: Option<VirtAddr>,
    pub ramdisk_slice_len: u64,
}

/// Allocates and initializes the boot info struct and the memory map.
///
/// The boot info and memory map are mapped to both the kernel and bootloader
/// address space at the same address. This makes it possible to return a Rust
/// reference that is valid in both address spaces. The necessary physical frames
/// are taken from the given `frame_allocator`.
pub fn create_boot_info<I, D>(
    config: &BootloaderConfig,
    boot_config: &BootConfig,
    mut frame_allocator: LegacyFrameAllocator<I, D>,
    page_tables: &mut PageTables,
    mappings: &mut Mappings,
    system_info: SystemInfo,
) -> &'static mut BootInfo
where
    I: ExactSizeIterator<Item = D> + Clone,
    D: LegacyMemoryRegion,
{
    log::info!("Allocate bootinfo");

    // allocate and map space for the boot info
    let (boot_info, memory_regions) = {
        let boot_info_layout = Layout::new::<BootInfo>();
        let regions = frame_allocator.memory_map_max_region_count();
        let memory_regions_layout = Layout::array::<MemoryRegion>(regions).unwrap();
        let (combined, memory_regions_offset) =
            boot_info_layout.extend(memory_regions_layout).unwrap();

        let boot_info_addr = match config.mappings.boot_info {
            Mapping::FixedAddress(addr) => {
                let addr = VirtAddr::new(addr);
                assert!(
                    addr.is_aligned(combined.align() as u64),
                    "boot info addr is not properly aligned"
                );
                addr
            }
            Mapping::Dynamic => {
                let addr = mappings
                    .address_space
                    .find_free(
                        combined.size() as u64,
                        combined.align() as u64,
                        None, // No ASLR for boot info
                    )
                    .expect("no free virtual address space for boot info");
                VirtAddr::new(addr.as_u64())
            }
        };

        let memory_map_regions_addr = boot_info_addr + memory_regions_offset as u64;
        let memory_map_regions_end = boot_info_addr + combined.size() as u64;

        let start_page = Page::containing_address(boot_info_addr);
        let end_page = Page::containing_address(memory_map_regions_end - 1u64);
        for page in Page::range_inclusive(start_page, end_page) {
            let flags =
                PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE;
            let frame = frame_allocator
                .allocate_frame()
                .expect("frame allocation for boot info failed");
            match unsafe {
                page_tables
                    .kernel
                    .map_to(page, frame, flags, &mut frame_allocator)
            } {
                Ok(tlb) => tlb.flush(),
                Err(err) => panic!("failed to map page {:?}: {:?}", page, err),
            }
            // we need to be able to access it too
            match unsafe {
                page_tables
                    .bootloader
                    .map_to(page, frame, flags, &mut frame_allocator)
            } {
                Ok(tlb) => tlb.flush(),
                Err(err) => panic!("failed to map page {:?}: {:?}", page, err),
            }
        }

        let boot_info: &'static mut MaybeUninit<BootInfo> =
            unsafe { &mut *boot_info_addr.as_mut_ptr() };
        let memory_regions: &'static mut [MaybeUninit<MemoryRegion>] =
            unsafe { slice::from_raw_parts_mut(memory_map_regions_addr.as_mut_ptr(), regions) };
        (boot_info, memory_regions)
    };

    log::info!("Create Memory Map");

    // build memory map
    let memory_regions = frame_allocator.construct_memory_map(
        memory_regions,
        mappings.kernel_slice_start,
        mappings.kernel_slice_len,
        mappings.ramdisk_slice_phys_start,
        mappings.ramdisk_slice_len,
    );

    log::info!("Create bootinfo");

    // create boot info
    let boot_info = boot_info.write({
        let mut info = BootInfo::new(memory_regions.into());
        info.framebuffer = mappings
            .framebuffer
            .map(|addr| unsafe {
                FrameBuffer::new(
                    addr.as_u64(),
                    system_info
                        .framebuffer
                        .expect(
                            "there shouldn't be a mapping for the framebuffer if there is \
                            no framebuffer",
                        )
                        .info,
                )
            })
            .into();
        info.physical_memory_offset = mappings.physical_memory_offset.map(VirtAddr::as_u64).into();
        info.recursive_index = mappings.recursive_index.map(Into::into).into();
        info.rsdp_addr = system_info.rsdp_addr.map(|addr| addr.as_u64()).into();
        info.tls_template = mappings.tls_template.into();
        info.ramdisk_addr = mappings
            .ramdisk_slice_start
            .map(|addr| addr.as_u64())
            .into();
        info.ramdisk_len = mappings.ramdisk_slice_len;
        info.kernel_addr = mappings.kernel_slice_start.as_u64();
        info.kernel_len = mappings.kernel_slice_len as _;
        info.kernel_image_offset = mappings.kernel_image_offset.as_u64();
        info.kernel_stack_bottom = mappings.stack_bottom.as_u64();
        info.kernel_stack_len = config.kernel_stack_size;
        info._test_sentinel = boot_config._test_sentinel;
        info
    });

    boot_info
}

/// Switches to the kernel address space and jumps to the kernel entry point.
pub fn switch_to_kernel(
    page_tables: PageTables,
    mappings: Mappings,
    boot_info: &'static mut BootInfo,
) -> ! {
    let PageTables {
        kernel_level_4_frame,
        ..
    } = page_tables;
    let addresses = Addresses {
        page_table: kernel_level_4_frame,
        stack_top: mappings.stack_top,
        entry_point: mappings.entry_point,
        boot_info,
    };

    log::info!(
        "Jumping to kernel entry point at {:?}",
        addresses.entry_point
    );

    unsafe {
        context_switch(addresses);
    }
}

/// Provides access to the page tables of the bootloader and kernel address space.
pub struct PageTables {
    /// Provides access to the page tables of the bootloader address space.
    pub bootloader: OffsetPageTable<'static>,
    /// Provides access to the page tables of the kernel address space (not active).
    pub kernel: OffsetPageTable<'static>,
    /// The physical frame where the level 4 page table of the kernel address space is stored.
    ///
    /// Must be the page table that the `kernel` field of this struct refers to.
    ///
    /// This frame is loaded into the `CR3` register on the final context switch to the kernel.
    pub kernel_level_4_frame: PhysFrame,
}

/// Performs the actual context switch.
unsafe fn context_switch(addresses: Addresses) -> ! {
    unsafe {
        asm!(
            r#"
            xor rbp, rbp
            mov cr3, {}
            mov rsp, {}
            push 0
            jmp {}
            "#,
            in(reg) addresses.page_table.start_address().as_u64(),
            in(reg) addresses.stack_top.as_u64(),
            in(reg) addresses.entry_point.as_u64(),
            in("rdi") addresses.boot_info as *const _ as usize,
        );
    }
    unreachable!();
}

/// Memory addresses required for the context switch.
struct Addresses {
    page_table: PhysFrame,
    stack_top: VirtAddr,
    entry_point: VirtAddr,
    boot_info: &'static mut BootInfo,
}

fn enable_nxe_bit() {
    use x86_64::registers::control::{Efer, EferFlags};
    unsafe { Efer::update(|efer| *efer |= EferFlags::NO_EXECUTE_ENABLE) }
}

fn enable_write_protect_bit() {
    use x86_64::registers::control::{Cr0, Cr0Flags};
    unsafe { Cr0::update(|cr0| *cr0 |= Cr0Flags::WRITE_PROTECT) };
}
