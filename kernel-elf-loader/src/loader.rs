//! The main `Loader` type that orchestrates kernel ELF loading and
//! additional memory region mapping.

use elf::{
    abi::{ET_DYN, ET_EXEC, PT_LOAD},
    endian::AnyEndian,
    ElfBytes,
};
use rand_core::RngCore;

use crate::{
    LoadedKernel,
    address_space::AddressSpace,
    elf_loading::load_and_relocate_elf,
    error::LoadError,
    traits::{FrameAllocator, PageFlags, PageSize, PageTable, PhysAddr, PhysicalMemoryAccess, VirtAddr},
};

/// How to place the kernel in virtual memory.
pub enum KernelPlacement {
    /// For ET_EXEC: use ELF addresses as-is.
    /// For ET_DYN: loader finds a free region automatically.
    Auto,
    /// Place the kernel at a specific virtual base (ET_DYN only).
    Fixed(VirtAddr),
}

/// How to place a mapped region in virtual memory.
pub enum RegionPlacement {
    /// Let the loader choose a free region.
    Auto,
    /// Place at a specific virtual address.
    Fixed(VirtAddr),
}

/// The ELF loader — generic over the platform's page size type `S`.
///
/// Holds references to the page table, frame allocator, and physical
/// memory accessor. Tracks the kernel's virtual address space to
/// support both kernel loading and additional data mapping.
pub struct Loader<'a, S: PageSize> {
    page_table: &'a mut dyn PageTable<S>,
    allocator: &'a mut dyn FrameAllocator<S>,
    phys_mem: &'a dyn PhysicalMemoryAccess,
    rng: Option<&'a mut dyn RngCore>,
    address_space: AddressSpace,
}

impl<'a, S: PageSize> Loader<'a, S> {
    /// Create a new loader.
    ///
    /// - `page_table`: The kernel's page table (not the active one).
    /// - `allocator`: Frame allocator for physical memory.
    /// - `phys_mem`: Physical memory access (e.g. identity-mapped).
    /// - `rng`: Optional RNG for ASLR. Pass `None` to disable.
    pub fn new(
        page_table: &'a mut dyn PageTable<S>,
        allocator: &'a mut dyn FrameAllocator<S>,
        phys_mem: &'a dyn PhysicalMemoryAccess,
        rng: Option<&'a mut dyn RngCore>,
    ) -> Self {
        Self {
            page_table,
            allocator,
            phys_mem,
            rng,
            address_space: AddressSpace::new(),
        }
    }

    /// Find a free region in the address space and mark it as used.
    fn find_free_region(&mut self, size: u64, alignment: u64) -> Option<VirtAddr> {
        let rng: Option<&mut dyn RngCore> = match self.rng {
            Some(ref mut r) => Some(*r),
            None => None,
        };
        self.address_space.find_free(size, alignment, rng)
    }

    /// Mark a virtual address range as used (e.g. identity-mapped region).
    pub fn mark_used(&mut self, start: VirtAddr, size: u64) {
        self.address_space.mark_used(start, size);
    }

    /// Load a kernel ELF into the page table.
    ///
    /// `kernel_bytes` is the raw ELF file contents.
    /// `kernel_phys_base` is the physical address of `kernel_bytes[0]`
    /// (must be aligned to `S::BASE.bytes()`).
    ///
    /// If `use_huge_pages` is true, the loader will try the sizes
    /// from `S::HUGE_PAGE_SIZES` for large aligned regions, falling
    /// back to `S::BASE` at the edges.
    pub fn load_kernel(
        &mut self,
        kernel_bytes: &[u8],
        kernel_phys_base: PhysAddr,
        placement: KernelPlacement,
        use_huge_pages: bool,
    ) -> Result<LoadedKernel, LoadError> {
        log::info!(
            "Loading kernel ELF at phys {:#x} ({} bytes)",
            kernel_phys_base.as_u64(),
            kernel_bytes.len()
        );

        let base_size = S::BASE.bytes();
        if !kernel_phys_base.is_aligned(base_size) {
            return Err(LoadError::NotPageAligned);
        }

        let elf = ElfBytes::<AnyEndian>::minimal_parse(kernel_bytes)
            .map_err(|_| LoadError::InvalidElf("failed to parse ELF file"))?;

        let elf_type = elf.ehdr.e_type;
        log::info!(
            "ELF type: {}, machine: {:#x}, entry: {:#x}",
            elf_type,
            elf.ehdr.e_machine,
            elf.ehdr.e_entry
        );

        let load_base: i64 = match elf_type {
            ET_EXEC => {
                match placement {
                    KernelPlacement::Auto => {
                        // ET_EXEC uses addresses as-is (no relocation).
                        0
                    }
                    KernelPlacement::Fixed(_) => {
                        return Err(LoadError::InvalidPlacement);
                    }
                }
            }
            ET_DYN => {
                let (size, align, min_addr) = calc_elf_memory_requirements(&elf)?;
                let base_size = S::BASE.bytes();
                let page_aligned_size = align_up(size, base_size);

                match placement {
                    KernelPlacement::Auto => {
                        let virt = self
                            .find_free_region(page_aligned_size, align)
                            .ok_or(LoadError::AddressSpaceFull)?;
                        log::info!("ET_DYN kernel placed at {:#x} (auto)", virt.as_u64());
                        virt.as_u64() as i64 - min_addr as i64
                    }
                    KernelPlacement::Fixed(base) => {
                        log::info!("ET_DYN kernel placed at {:#x} (fixed)", base.as_u64());
                        base.as_u64() as i64
                    }
                }
            }
            _ => return Err(LoadError::UnsupportedElfType(elf_type)),
        };

        log::info!("Load base offset: {:#x}", load_base);

        // Mark the kernel's virtual address range as used (page-aligned).
        if let Ok((size, _, min_addr)) = calc_elf_memory_requirements(&elf) {
            let base_size = S::BASE.bytes();
            let start = (min_addr as i128 + load_base as i128) as u64;
            let aligned_start = start / base_size * base_size;
            let end = start + size;
            let aligned_end = align_up(end, base_size);
            self.address_space
                .mark_used(VirtAddr::new(aligned_start), aligned_end - aligned_start);
        }

        let tls_template = load_and_relocate_elf(
            &elf,
            kernel_bytes,
            kernel_phys_base,
            load_base,
            use_huge_pages,
            self.page_table,
            self.allocator,
            self.phys_mem,
        )?;

        let entry_point = (elf.ehdr.e_entry as i128 + load_base as i128) as u64;
        log::info!("Kernel entry point: {:#x}", entry_point);

        Ok(LoadedKernel {
            entry_point: VirtAddr::new(entry_point),
            load_base,
            tls_template,
        })
    }

    /// Map a contiguous physical region into the kernel's address space.
    ///
    /// Use for: framebuffer, ramdisk, physical memory mapping, etc.
    pub fn map_physical_region(
        &mut self,
        phys_start: PhysAddr,
        size: u64,
        flags: PageFlags,
        placement: RegionPlacement,
        use_huge_pages: bool,
    ) -> Result<VirtAddr, LoadError> {
        if size == 0 {
            return Err(LoadError::InvalidPlacement);
        }

        let base_size = S::BASE.bytes();
        let alignment = if use_huge_pages {
            S::HUGE_PAGE_SIZES
                .first()
                .map(|s| s.bytes())
                .unwrap_or(base_size)
        } else {
            base_size
        };

        // Round up to page boundaries for address space tracking,
        // since page mappings always consume full pages.
        let page_aligned_size = align_up(size, base_size);

        let virt_start = match placement {
            RegionPlacement::Auto => {
                self.find_free_region(page_aligned_size, alignment)
                    .ok_or(LoadError::AddressSpaceFull)?
            }
            RegionPlacement::Fixed(addr) => {
                self.address_space.mark_used(addr, page_aligned_size);
                addr
            }
        };

        log::info!(
            "Mapping physical region: phys={:#x}, size={:#x}, virt={:#x}, huge={}",
            phys_start.as_u64(),
            size,
            virt_start.as_u64(),
            use_huge_pages
        );

        // Map pages.
        let mut offset: u64 = 0;
        while offset < size {
            let virt = VirtAddr::new(virt_start.as_u64() + offset);
            let phys = PhysAddr::new(phys_start.as_u64() + offset);
            let remaining = size - offset;

            let mut mapped = false;
            if use_huge_pages {
                for &huge_size in S::HUGE_PAGE_SIZES {
                    let sz = huge_size.bytes();
                    if sz <= remaining && virt.is_aligned(sz) && phys.is_aligned(sz) {
                        self.page_table
                            .map(virt, phys, huge_size, flags, self.allocator)
                            .map_err(LoadError::MappingFailed)?;
                        offset += sz;
                        mapped = true;
                        break;
                    }
                }
            }

            if !mapped {
                self.page_table
                    .map(virt, phys, S::BASE, flags, self.allocator)
                    .map_err(LoadError::MappingFailed)?;
                offset += base_size;
            }
        }

        Ok(virt_start)
    }

    /// Allocate fresh zeroed frames and map them.
    ///
    /// Use for: kernel stack, boot info region, etc.
    /// Always uses base page size.
    pub fn allocate_and_map(
        &mut self,
        size: u64,
        flags: PageFlags,
        placement: RegionPlacement,
    ) -> Result<VirtAddr, LoadError> {
        let base_size = S::BASE.bytes();
        let aligned_size = align_up(size, base_size);

        let virt_start = match placement {
            RegionPlacement::Auto => {
                self.find_free_region(aligned_size, base_size)
                    .ok_or(LoadError::AddressSpaceFull)?
            }
            RegionPlacement::Fixed(addr) => {
                self.address_space.mark_used(addr, aligned_size);
                addr
            }
        };

        log::info!(
            "Allocating and mapping: size={:#x}, virt={:#x}",
            aligned_size,
            virt_start.as_u64()
        );

        let num_pages = aligned_size / base_size;
        for i in 0..num_pages {
            let virt = VirtAddr::new(virt_start.as_u64() + i * base_size);
            let frame = self
                .allocator
                .allocate_frame(S::BASE)
                .ok_or(LoadError::FrameAllocationFailed)?;

            // Zero the frame.
            unsafe {
                self.phys_mem.zero_phys(frame, base_size as usize);
            }

            self.page_table
                .map(virt, frame, S::BASE, flags, self.allocator)
                .map_err(LoadError::MappingFailed)?;
        }

        Ok(virt_start)
    }

    /// Create a guard page: reserves one base-page-sized virtual
    /// address range that is deliberately left **unmapped**.
    ///
    /// Any access to this address will trigger a page fault.
    /// Use below a kernel stack to catch stack overflows.
    /// Returns the virtual address of the guard page.
    pub fn add_guard_page(
        &mut self,
        placement: RegionPlacement,
    ) -> Result<VirtAddr, LoadError> {
        let base_size = S::BASE.bytes();

        let virt = match placement {
            RegionPlacement::Auto => {
                self.find_free_region(base_size, base_size)
                    .ok_or(LoadError::AddressSpaceFull)?
            }
            RegionPlacement::Fixed(addr) => {
                self.address_space.mark_used(addr, base_size);
                addr
            }
        };

        log::info!("Guard page at {:#x}", virt.as_u64());

        // Deliberately do NOT map this page — that's the point.
        Ok(virt)
    }

    /// Get a reference to the address space tracker.
    pub fn address_space(&self) -> &AddressSpace {
        &self.address_space
    }

    /// Get a mutable reference to the address space tracker.
    pub fn address_space_mut(&mut self) -> &mut AddressSpace {
        &mut self.address_space
    }

    /// Get a mutable reference to the page table.
    ///
    /// Use for platform-specific operations like recursive page table
    /// entries or identity mapping the context switch function.
    pub fn page_table_mut(&mut self) -> &mut dyn PageTable<S> {
        self.page_table
    }

    /// Get a mutable reference to the frame allocator.
    pub fn frame_allocator_mut(&mut self) -> &mut dyn FrameAllocator<S> {
        self.allocator
    }
}

/// Calculate ELF memory requirements: (total_size, alignment, min_addr).
fn calc_elf_memory_requirements(elf: &ElfBytes<AnyEndian>) -> Result<(u64, u64, u64), LoadError> {
    let segments = elf
        .segments()
        .ok_or(LoadError::InvalidElf("no program headers"))?;

    let mut min_addr = u64::MAX;
    let mut max_addr = 0u64;
    let mut max_align = 1u64;

    for phdr in segments.iter() {
        if phdr.p_type == PT_LOAD {
            let start = phdr.p_vaddr;
            let end = phdr.p_vaddr + phdr.p_memsz;
            if start < min_addr {
                min_addr = start;
            }
            if end > max_addr {
                max_addr = end;
            }
            if phdr.p_align > max_align {
                max_align = phdr.p_align;
            }
        }
    }

    if min_addr == u64::MAX {
        return Err(LoadError::InvalidElf("no LOAD segments"));
    }

    let size = max_addr - min_addr;
    Ok((size, max_align, min_addr))
}

fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) / alignment * alignment
}
