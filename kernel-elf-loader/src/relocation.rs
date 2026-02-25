//! Relocation handling for the kernel ELF loader.
//!
//! Supports RELA and RELR (compact) relocations with copy-on-write
//! page handling to avoid aliasing the original kernel ELF file.

use crate::{
    error::LoadError,
    traits::{FrameAllocator, PageFlags, PageSize, PageTable, PhysAddr, PhysicalMemoryAccess, VirtAddr},
};

/// Maximum number of pages that can be copy-on-write'd during relocation.
const MAX_COPIED_PAGES: usize = 1024;

/// Tracks pages that have been copied for mutation during relocation.
pub(crate) struct CopiedPages {
    pages: [u64; MAX_COPIED_PAGES],
    count: usize,
}

impl CopiedPages {
    pub fn new() -> Self {
        Self {
            pages: [0; MAX_COPIED_PAGES],
            count: 0,
        }
    }

    /// Check if a page (identified by its aligned virtual address) has been copied.
    pub fn contains(&self, page_addr: u64) -> bool {
        self.pages[..self.count].contains(&page_addr)
    }

    /// Record that a page has been copied.
    pub fn insert(&mut self, page_addr: u64) -> Result<(), LoadError> {
        if self.count >= MAX_COPIED_PAGES {
            return Err(LoadError::TooManyCopiedPages);
        }
        self.pages[self.count] = page_addr;
        self.count += 1;
        Ok(())
    }
}

/// Ensure the page containing `virt` is privately owned (copied from the
/// original ELF mapping if necessary).
///
/// Returns the physical address of the (now-writable) frame.
pub(crate) fn ensure_page_writable<S: PageSize>(
    virt: VirtAddr,
    page_table: &mut dyn PageTable<S>,
    allocator: &mut dyn FrameAllocator<S>,
    phys_mem: &dyn PhysicalMemoryAccess,
    copied: &mut CopiedPages,
) -> Result<PhysAddr, LoadError> {
    let base_size = S::BASE.bytes();
    let page_addr = virt.as_u64() / base_size * base_size;

    if copied.contains(page_addr) {
        // Already copied — just look up the current physical address.
        let (phys, _, _) = page_table
            .translate(VirtAddr::new(page_addr))
            .ok_or(LoadError::InvalidElf("page not mapped during relocation"))?;
        return Ok(PhysAddr::new(
            phys.as_u64() + (virt.as_u64() - page_addr),
        ));
    }

    // Look up the current mapping.
    let (old_phys, old_flags, _old_size) = page_table
        .translate(VirtAddr::new(page_addr))
        .ok_or(LoadError::InvalidElf("page not mapped during relocation"))?;

    // Allocate a new frame.
    let new_phys = allocator
        .allocate_frame(S::BASE)
        .ok_or(LoadError::FrameAllocationFailed)?;

    // Copy the old frame's contents to the new frame.
    unsafe {
        phys_mem.copy_phys(old_phys, new_phys, base_size as usize);
    }

    // Remap the page to the new frame with writable flags.
    page_table
        .unmap(VirtAddr::new(page_addr))
        .map_err(|_| LoadError::InvalidElf("failed to unmap page during CoW"))?;

    let new_flags = PageFlags {
        writable: true,
        executable: old_flags.executable,
    };
    page_table
        .map(
            VirtAddr::new(page_addr),
            new_phys,
            S::BASE,
            new_flags,
            allocator,
        )
        .map_err(|e| LoadError::MappingFailed(e))?;

    copied.insert(page_addr)?;

    Ok(PhysAddr::new(
        new_phys.as_u64() + (virt.as_u64() - page_addr),
    ))
}

/// Look up the expected RELATIVE relocation type for a given ELF machine.
pub(crate) fn relative_reloc_type(machine: u16) -> Option<u32> {
    match machine {
        elf::abi::EM_X86_64 => Some(elf::abi::R_X86_64_RELATIVE),
        elf::abi::EM_AARCH64 => Some(elf::abi::R_AARCH64_RELATIVE),
        elf::abi::EM_RISCV => Some(elf::abi::R_RISCV_RELATIVE),
        _ => None,
    }
}

/// Read a u64 value from a virtual address in the kernel's address space.
///
/// Translates the virtual address through the page table, then reads
/// from the corresponding physical address.
pub(crate) fn read_u64_at<S: PageSize>(
    virt: VirtAddr,
    page_table: &dyn PageTable<S>,
    phys_mem: &dyn PhysicalMemoryAccess,
) -> Result<u64, LoadError> {
    let base_size = S::BASE.bytes();
    let page_addr = virt.as_u64() / base_size * base_size;
    let offset = virt.as_u64() - page_addr;

    let (phys, _, _) = page_table
        .translate(VirtAddr::new(page_addr))
        .ok_or(LoadError::InvalidElf("address not mapped"))?;

    let phys_target = PhysAddr::new(phys.as_u64() + offset);
    let mut buf = [0u8; 8];
    unsafe {
        phys_mem.read_phys(phys_target, &mut buf);
    }
    Ok(u64::from_ne_bytes(buf))
}

/// Write a u64 value to a virtual address in the kernel's address space,
/// using CoW to ensure the page is privately owned.
pub(crate) fn write_u64_at<S: PageSize>(
    virt: VirtAddr,
    value: u64,
    page_table: &mut dyn PageTable<S>,
    allocator: &mut dyn FrameAllocator<S>,
    phys_mem: &dyn PhysicalMemoryAccess,
    copied: &mut CopiedPages,
) -> Result<(), LoadError> {
    let phys = ensure_page_writable(virt, page_table, allocator, phys_mem, copied)?;
    unsafe {
        phys_mem.write_phys(phys, &value.to_ne_bytes());
    }
    Ok(())
}

/// Process RELA relocations from the dynamic segment.
///
/// `rela_offset`: virtual address (pre-load-base) of the relocation table
/// `rela_size`: total size in bytes of the relocation table
/// `rela_ent`: size of each entry
/// `load_base`: the virtual address offset applied to the kernel
/// `e_machine`: ELF machine type from the header
pub(crate) fn process_rela<S: PageSize>(
    rela_offset: u64,
    rela_size: u64,
    rela_ent: u64,
    load_base: i64,
    e_machine: u16,
    page_table: &mut dyn PageTable<S>,
    allocator: &mut dyn FrameAllocator<S>,
    phys_mem: &dyn PhysicalMemoryAccess,
    copied: &mut CopiedPages,
) -> Result<(), LoadError> {
    let expected_type = relative_reloc_type(e_machine)
        .ok_or(LoadError::UnsupportedMachine(e_machine))?;

    if rela_ent != 24 {
        return Err(LoadError::InvalidElf("unexpected RELA entry size (expected 24)"));
    }

    let num_entries = rela_size / rela_ent;
    log::info!(
        "Processing {} RELA relocations (table at vaddr {:#x})",
        num_entries,
        rela_offset
    );

    for idx in 0..num_entries {
        let entry_vaddr = (rela_offset as i128 + load_base as i128) as u64 + idx * rela_ent;

        // Read the three fields of the Rela entry: offset, info, addend.
        let r_offset = read_u64_at(VirtAddr::new(entry_vaddr), page_table, phys_mem)?;
        let r_info = read_u64_at(VirtAddr::new(entry_vaddr + 8), page_table, phys_mem)?;
        let r_addend = read_u64_at(VirtAddr::new(entry_vaddr + 16), page_table, phys_mem)?;

        let r_type = (r_info & 0xFFFF_FFFF) as u32;
        let r_sym = (r_info >> 32) as u32;

        if r_sym != 0 {
            return Err(LoadError::InvalidElf(
                "relocations using the symbol table are not supported",
            ));
        }

        if r_type != expected_type {
            return Err(LoadError::UnsupportedRelocation(r_type));
        }

        // R_*_RELATIVE: value = load_base + addend
        let value = (load_base as i128 + r_addend as i64 as i128) as u64;
        let dest_vaddr = (r_offset as i128 + load_base as i128) as u64;

        write_u64_at(
            VirtAddr::new(dest_vaddr),
            value,
            page_table,
            allocator,
            phys_mem,
            copied,
        )?;
    }

    Ok(())
}

/// Process RELR (compact relative) relocations.
///
/// RELR is a compact encoding that only stores offsets (no addend/type).
/// The value at each offset is read, adjusted by load_base, and written back.
///
/// Encoding: a sequence of u64 entries:
/// - If entry is even: it's a base address. Apply one relocation there.
/// - If entry is odd: it's a bitmap. For each set bit i (bits 1..63),
///   apply a relocation at base + i * 8. Then advance base += 63 * 8.
pub(crate) fn process_relr<S: PageSize>(
    relr_vaddr: u64,
    relr_size: u64,
    relr_ent: u64,
    load_base: i64,
    page_table: &mut dyn PageTable<S>,
    allocator: &mut dyn FrameAllocator<S>,
    phys_mem: &dyn PhysicalMemoryAccess,
    copied: &mut CopiedPages,
) -> Result<(), LoadError> {
    if relr_ent != 8 {
        return Err(LoadError::InvalidElf("unexpected RELR entry size (expected 8)"));
    }

    let num_entries = relr_size / relr_ent;
    log::info!(
        "Processing {} RELR entries (table at vaddr {:#x})",
        num_entries,
        relr_vaddr
    );

    let mut base: u64 = 0;
    let mut reloc_count: u64 = 0;

    for idx in 0..num_entries {
        let entry_phys_vaddr = (relr_vaddr as i128 + load_base as i128) as u64 + idx * relr_ent;
        let entry = read_u64_at(VirtAddr::new(entry_phys_vaddr), page_table, phys_mem)?;

        if entry & 1 == 0 {
            // Even entry: base address. Apply one relocation here.
            base = (entry as i128 + load_base as i128) as u64;
            apply_relative_reloc(base, load_base, page_table, allocator, phys_mem, copied)?;
            reloc_count += 1;
            base += 8;
        } else {
            // Odd entry: bitmap. Each set bit i (starting from bit 1) means
            // a relocation at base + (i-1) * 8.
            // Actually the standard encoding: bit 0 is the marker (always 1).
            // Bits 1..63 are the bitmap. Bit j (1 <= j <= 63) means
            // a relocation at base + (j-1) * 8.
            let mut bitmap = entry >> 1;
            let mut offset = base;
            while bitmap != 0 {
                if bitmap & 1 != 0 {
                    apply_relative_reloc(
                        offset, load_base, page_table, allocator, phys_mem, copied,
                    )?;
                    reloc_count += 1;
                }
                bitmap >>= 1;
                offset += 8;
            }
            // Advance base past the 63 slots covered by this bitmap entry.
            base += 63 * 8;
        }
    }

    log::info!("Applied {} RELR relocations", reloc_count);
    Ok(())
}

/// Apply a single RELATIVE relocation at the given virtual address:
/// read the current value, add load_base, write it back.
fn apply_relative_reloc<S: PageSize>(
    virt_addr: u64,
    load_base: i64,
    page_table: &mut dyn PageTable<S>,
    allocator: &mut dyn FrameAllocator<S>,
    phys_mem: &dyn PhysicalMemoryAccess,
    copied: &mut CopiedPages,
) -> Result<(), LoadError> {
    let current = read_u64_at(VirtAddr::new(virt_addr), page_table, phys_mem)?;
    let relocated = (current as i128 + load_base as i128) as u64;
    write_u64_at(
        VirtAddr::new(virt_addr),
        relocated,
        page_table,
        allocator,
        phys_mem,
        copied,
    )
}
