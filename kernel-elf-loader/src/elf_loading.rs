//! Internal ELF loading logic: LOAD segment mapping, BSS handling, TLS extraction.

use elf::{
    abi::{PT_DYNAMIC, PT_GNU_RELRO, PT_LOAD, PT_TLS, PF_W, PF_X},
    endian::AnyEndian,
    ElfBytes,
};

// RELR constants not yet defined in elf 0.8.
const DT_RELR: i64 = 36;
const DT_RELRSZ: i64 = 35;
const DT_RELRENT: i64 = 37;

use crate::{
    TlsTemplate,
    align_up,
    error::LoadError,
    relocation::{CopiedPages, process_rela, process_relr},
    traits::{FrameAllocator, PageFlags, PageSize, PageTable, PhysAddr, PhysicalMemoryAccess, VirtAddr},
};

/// Map all LOAD segments of the kernel ELF into the page table.
///
/// Returns the TLS template if a TLS segment is present.
pub(crate) fn load_and_relocate_elf<S: PageSize>(
    elf: &ElfBytes<AnyEndian>,
    kernel_phys_base: PhysAddr,
    load_base: i64,
    use_huge_pages: bool,
    page_table: &mut dyn PageTable<S>,
    allocator: &mut dyn FrameAllocator<S>,
    phys_mem: &dyn PhysicalMemoryAccess,
) -> Result<Option<TlsTemplate>, LoadError> {
    let segments = elf.segments().ok_or(LoadError::InvalidElf("no program headers"))?;
    let base_size = S::BASE.bytes();

    // Phase 1: Map LOAD segments, extract TLS.
    let mut tls_template = None;
    for phdr in segments.iter() {
        match phdr.p_type {
            PT_LOAD => {
                map_load_segment(
                    &phdr,
                    kernel_phys_base,
                    load_base,
                    base_size,
                    use_huge_pages,
                    page_table,
                    allocator,
                    phys_mem,
                )?;
            }
            PT_TLS => {
                if tls_template.is_some() {
                    return Err(LoadError::InvalidElf("multiple TLS segments"));
                }
                let start_addr = (phdr.p_vaddr as i128 + load_base as i128) as u64;
                tls_template = Some(TlsTemplate {
                    start_addr: VirtAddr::new(start_addr),
                    file_size: phdr.p_filesz,
                    mem_size: phdr.p_memsz,
                });
                log::info!(
                    "TLS template: vaddr={:#x}, file_size={:#x}, mem_size={:#x}",
                    start_addr,
                    phdr.p_filesz,
                    phdr.p_memsz
                );
            }
            _ => {}
        }
    }

    // Phase 2: Apply relocations from Dynamic segment.
    let mut copied = CopiedPages::new();
    for phdr in segments.iter() {
        if phdr.p_type == PT_DYNAMIC {
            process_dynamic_segment(
                elf,
                load_base,
                page_table,
                allocator,
                phys_mem,
                &mut copied,
            )?;
        }
    }

    // Phase 3: Mark RELRO regions as read-only.
    for phdr in segments.iter() {
        if phdr.p_type == PT_GNU_RELRO {
            handle_relro_segment(&phdr, load_base, base_size, page_table)?;
        }
    }

    Ok(tls_template)
}

/// Map a single LOAD segment into the page table.
fn map_load_segment<S: PageSize>(
    phdr: &elf::segment::ProgramHeader,
    kernel_phys_base: PhysAddr,
    load_base: i64,
    base_size: u64,
    use_huge_pages: bool,
    page_table: &mut dyn PageTable<S>,
    allocator: &mut dyn FrameAllocator<S>,
    phys_mem: &dyn PhysicalMemoryAccess,
) -> Result<(), LoadError> {
    let phys_start = PhysAddr::new(kernel_phys_base.as_u64() + phdr.p_offset);
    let virt_start_raw = (phdr.p_vaddr as i128 + load_base as i128) as u64;
    let virt_start = VirtAddr::new(virt_start_raw);

    let flags = elf_flags_to_page_flags(phdr.p_flags);

    log::info!(
        "LOAD segment: vaddr={:#x}, phys={:#x}, filesz={:#x}, memsz={:#x}, flags={{w={}, x={}}}",
        virt_start_raw,
        phys_start.as_u64(),
        phdr.p_filesz,
        phdr.p_memsz,
        flags.writable,
        flags.executable
    );

    // Map the file-backed portion of the segment.
    let file_size = phdr.p_filesz;
    if file_size > 0 {
        let start_page = virt_start.as_u64() / base_size * base_size;
        let end_addr = virt_start_raw + file_size;
        let end_page_start = (end_addr.saturating_sub(1)) / base_size * base_size;

        let phys_page_start = phys_start.as_u64() / base_size * base_size;

        let num_pages = (end_page_start - start_page) / base_size + 1;
        map_pages(
            VirtAddr::new(start_page),
            PhysAddr::new(phys_page_start),
            num_pages,
            flags,
            use_huge_pages,
            page_table,
            allocator,
        )?;
    }

    // Handle BSS (mem_size > file_size).
    if phdr.p_memsz > phdr.p_filesz {
        handle_bss_section(
            virt_start,
            phdr.p_filesz,
            phdr.p_memsz,
            flags,
            base_size,
            page_table,
            allocator,
            phys_mem,
        )?;
    }

    Ok(())
}

/// Map `num_pages` pages starting at `virt_start` to physical frames at `phys_start`.
///
/// If `use_huge_pages` is true, tries larger page sizes from `S::HUGE_PAGE_SIZES`
/// where alignment permits.
fn map_pages<S: PageSize>(
    virt_start: VirtAddr,
    phys_start: PhysAddr,
    num_base_pages: u64,
    flags: PageFlags,
    use_huge_pages: bool,
    page_table: &mut dyn PageTable<S>,
    allocator: &mut dyn FrameAllocator<S>,
) -> Result<(), LoadError> {
    let base_size = S::BASE.bytes();
    let total_size = num_base_pages * base_size;
    let mut offset: u64 = 0;

    while offset < total_size {
        let virt = VirtAddr::new(virt_start.as_u64() + offset);
        let phys = PhysAddr::new(phys_start.as_u64() + offset);
        let remaining = total_size - offset;

        let mut mapped = false;
        if use_huge_pages {
            for &huge_size in S::HUGE_PAGE_SIZES {
                let size_bytes = huge_size.bytes();
                if size_bytes <= remaining
                    && virt.is_aligned(size_bytes)
                    && phys.is_aligned(size_bytes)
                {
                    page_table
                        .map(virt, phys, huge_size, flags, allocator)
                        .map_err(LoadError::MappingFailed)?;
                    offset += size_bytes;
                    mapped = true;
                    break;
                }
            }
        }

        if !mapped {
            page_table
                .map(virt, phys, S::BASE, flags, allocator)
                .map_err(LoadError::MappingFailed)?;
            offset += base_size;
        }
    }

    Ok(())
}

/// Handle the BSS portion of a LOAD segment (memory beyond the file-backed region).
fn handle_bss_section<S: PageSize>(
    virt_start: VirtAddr,
    file_size: u64,
    mem_size: u64,
    flags: PageFlags,
    base_size: u64,
    page_table: &mut dyn PageTable<S>,
    allocator: &mut dyn FrameAllocator<S>,
    phys_mem: &dyn PhysicalMemoryAccess,
) -> Result<(), LoadError> {
    log::info!("Mapping BSS section");

    let zero_start = virt_start.as_u64() + file_size;
    let zero_end = virt_start.as_u64() + mem_size;

    // Handle partial page at the boundary between file data and BSS.
    // If zero_start is not page-aligned, we need to copy the last file page
    // and zero the remainder (to avoid aliasing the original ELF data).
    let data_bytes_before_zero = zero_start % base_size;
    if data_bytes_before_zero != 0 {
        let last_page_vaddr = zero_start / base_size * base_size;

        // Translate the current mapping to find the physical frame.
        let (old_phys, _old_flags, _old_size) = page_table
            .translate(VirtAddr::new(last_page_vaddr))
            .ok_or(LoadError::InvalidElf("BSS partial page not mapped"))?;

        // Allocate a new frame and copy the old contents.
        let new_phys = allocator
            .allocate_frame(S::BASE)
            .ok_or(LoadError::FrameAllocationFailed)?;
        unsafe {
            phys_mem.copy_phys(old_phys, new_phys, base_size as usize);
        }

        // Zero the BSS portion of the new frame.
        let zero_offset = data_bytes_before_zero as usize;
        let zero_len = (base_size - data_bytes_before_zero) as usize;
        unsafe {
            phys_mem.zero_phys(
                PhysAddr::new(new_phys.as_u64() + zero_offset as u64),
                zero_len,
            );
        }

        // Remap to the new frame.
        page_table
            .unmap(VirtAddr::new(last_page_vaddr))
            .map_err(|_| LoadError::InvalidElf("failed to unmap page during BSS handling"))?;
        page_table
            .map(VirtAddr::new(last_page_vaddr), new_phys, S::BASE, flags, allocator)
            .map_err(LoadError::MappingFailed)?;
    }

    // Map additional zeroed frames for the rest of the BSS.
    let alloc_start = align_up(zero_start, base_size);
    if alloc_start < zero_end {
        let alloc_end_page = (zero_end.saturating_sub(1)) / base_size * base_size;
        let num_pages = (alloc_end_page - alloc_start) / base_size + 1;

        for i in 0..num_pages {
            let virt = VirtAddr::new(alloc_start + i * base_size);
            let frame = allocator
                .allocate_frame(S::BASE)
                .ok_or(LoadError::FrameAllocationFailed)?;

            // Zero the frame.
            unsafe {
                phys_mem.zero_phys(frame, base_size as usize);
            }

            page_table
                .map(virt, frame, S::BASE, flags, allocator)
                .map_err(LoadError::MappingFailed)?;
        }
    }

    Ok(())
}

/// Process the PT_DYNAMIC segment to find and apply relocations.
fn process_dynamic_segment<S: PageSize>(
    elf: &ElfBytes<AnyEndian>,
    load_base: i64,
    page_table: &mut dyn PageTable<S>,
    allocator: &mut dyn FrameAllocator<S>,
    phys_mem: &dyn PhysicalMemoryAccess,
    copied: &mut CopiedPages,
) -> Result<(), LoadError> {
    let e_machine = elf.ehdr.e_machine;

    // Parse the dynamic section from the raw ELF bytes.
    let dynamic = elf
        .dynamic()
        .map_err(|_| LoadError::InvalidElf("failed to parse dynamic section"))?;
    let dynamic = match dynamic {
        Some(d) => d,
        None => return Ok(()),
    };

    let mut rela_offset = None;
    let mut rela_size = None;
    let mut rela_ent = None;
    let mut relr_offset = None;
    let mut relr_size = None;
    let mut relr_ent = None;

    for entry in dynamic.iter() {
        match entry.d_tag {
            elf::abi::DT_RELA => rela_offset = Some(entry.d_val()),
            elf::abi::DT_RELASZ => rela_size = Some(entry.d_val()),
            elf::abi::DT_RELAENT => rela_ent = Some(entry.d_val()),
            DT_RELR => relr_offset = Some(entry.d_val()),
            DT_RELRSZ => relr_size = Some(entry.d_val()),
            DT_RELRENT => relr_ent = Some(entry.d_val()),
            _ => {}
        }
    }

    // Process RELA relocations if present.
    if let Some(offset) = rela_offset {
        let size = rela_size.ok_or(LoadError::InvalidElf("DT_RELA without DT_RELASZ"))?;
        let ent = rela_ent.ok_or(LoadError::InvalidElf("DT_RELA without DT_RELAENT"))?;
        process_rela(
            offset, size, ent, load_base, e_machine, page_table, allocator, phys_mem, copied,
        )?;
    } else if rela_size.is_some() || rela_ent.is_some() {
        return Err(LoadError::InvalidElf(
            "DT_RELASZ or DT_RELAENT without DT_RELA",
        ));
    }

    // Process RELR relocations if present.
    if let Some(offset) = relr_offset {
        let size = relr_size.ok_or(LoadError::InvalidElf("DT_RELR without DT_RELRSZ"))?;
        let ent = relr_ent.ok_or(LoadError::InvalidElf("DT_RELR without DT_RELRENT"))?;
        process_relr(
            offset, size, ent, load_base, page_table, allocator, phys_mem, copied,
        )?;
    }

    Ok(())
}

/// Mark a RELRO segment as read-only after relocations have been applied.
fn handle_relro_segment<S: PageSize>(
    phdr: &elf::segment::ProgramHeader,
    load_base: i64,
    base_size: u64,
    page_table: &mut dyn PageTable<S>,
) -> Result<(), LoadError> {
    let start = (phdr.p_vaddr as i128 + load_base as i128) as u64;
    let end = start + phdr.p_memsz;

    let start_page = start / base_size * base_size;
    let end_page = (end.saturating_sub(1)) / base_size * base_size;

    log::info!(
        "RELRO: marking pages {:#x}..={:#x} as read-only",
        start_page,
        end_page
    );

    let mut page = start_page;
    while page <= end_page {
        if let Some((_phys, flags, _size)) = page_table.translate(VirtAddr::new(page)) {
            if flags.writable {
                let new_flags = PageFlags {
                    writable: false,
                    executable: flags.executable,
                };
                page_table
                    .update_flags(VirtAddr::new(page), new_flags)
                    .map_err(LoadError::MappingFailed)?;
            }
        }
        page += base_size;
    }

    Ok(())
}

/// Convert ELF segment flags to PageFlags.
pub(crate) fn elf_flags_to_page_flags(elf_flags: u32) -> PageFlags {
    PageFlags {
        writable: elf_flags & PF_W != 0,
        executable: elf_flags & PF_X != 0,
    }
}

