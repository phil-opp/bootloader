//! x86_64 trait implementations for the kernel ELF loader.
//!
//! Implements the loader's [`PageSize`](crate::PageSize) and
//! [`PageTable`](crate::PageTable) traits for the `x86_64` crate's
//! page table types ([`OffsetPageTable`] and [`MappedPageTable`]).
//!
//! Enable the `x86_64` Cargo feature to use this module.

use crate::{
    error::{MapError, UnmapError},
    FrameAllocator, PageFlags, PageSize, PageTable, PhysAddr, VirtAddr,
};
use x86_64::structures::paging::{
    self,
    FrameAllocator as X86_64FrameAllocatorTrait, Mapper, MappedPageTable, OffsetPageTable, Page,
    PageTableFlags, PhysFrame, Size2MiB, Size4KiB, Translate,
    mapper::{MappedFrame, MapToError, PageTableFrameMapping, TranslateResult},
};

/// x86_64 page sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum X86_64PageSize {
    /// 4 KiB base page.
    Size4KiB,
    /// 2 MiB huge page.
    Size2MiB,
}

impl PageSize for X86_64PageSize {
    fn bytes(self) -> u64 {
        match self {
            Self::Size4KiB => 4096,
            Self::Size2MiB => 2 * 1024 * 1024,
        }
    }

    const BASE: Self = Self::Size4KiB;
    const HUGE_PAGE_SIZES: &[Self] = &[Self::Size2MiB];
}

/// Convert [`PageFlags`] to x86_64 [`PageTableFlags`].
fn to_x86_64_flags(flags: PageFlags) -> PageTableFlags {
    let mut f = PageTableFlags::PRESENT;
    if flags.writable {
        f |= PageTableFlags::WRITABLE;
    }
    if !flags.executable {
        f |= PageTableFlags::NO_EXECUTE;
    }
    f
}

/// Convert x86_64 [`PageTableFlags`] to [`PageFlags`].
fn from_x86_64_flags(flags: PageTableFlags) -> PageFlags {
    PageFlags {
        writable: flags.contains(PageTableFlags::WRITABLE),
        executable: !flags.contains(PageTableFlags::NO_EXECUTE),
    }
}

fn map_to_error<S: paging::PageSize>(e: MapToError<S>) -> MapError {
    match e {
        MapToError::FrameAllocationFailed => MapError::FrameAllocationFailed,
        MapToError::ParentEntryHugePage => MapError::ParentEntryHugePage,
        MapToError::PageAlreadyMapped(_) => MapError::AlreadyMapped,
    }
}

/// Shared implementation of [`PageTable`] for any type that implements
/// the x86_64 crate's [`Mapper`] and [`Translate`] traits.
fn map_impl(
    mapper: &mut (impl Mapper<Size4KiB> + Mapper<Size2MiB>),
    virt: VirtAddr,
    phys: PhysAddr,
    page_size: X86_64PageSize,
    flags: PageFlags,
    allocator: &mut dyn FrameAllocator<X86_64PageSize>,
) -> Result<(), MapError> {
    let x86_64_flags = to_x86_64_flags(flags);
    // Parent table flags need to be both readable and writable to
    // support recursive page tables.
    // See https://github.com/rust-osdev/bootloader/issues/443#issuecomment-2130010621
    let parent_flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;

    let mut adapter = FrameAllocAdapter(allocator);

    match page_size {
        X86_64PageSize::Size4KiB => {
            let page = Page::<Size4KiB>::from_start_address(
                x86_64::VirtAddr::new(virt.as_u64()),
            )
            .map_err(|_| MapError::InvalidAddress)?;
            let frame = PhysFrame::<Size4KiB>::from_start_address(
                x86_64::PhysAddr::new(phys.as_u64()),
            )
            .map_err(|_| MapError::InvalidAddress)?;

            unsafe {
                mapper
                    .map_to_with_table_flags(page, frame, x86_64_flags, parent_flags, &mut adapter)
                    .map_err(map_to_error)?
                    .ignore();
            }
        }
        X86_64PageSize::Size2MiB => {
            let page = Page::<Size2MiB>::from_start_address(
                x86_64::VirtAddr::new(virt.as_u64()),
            )
            .map_err(|_| MapError::InvalidAddress)?;
            let frame = PhysFrame::<Size2MiB>::from_start_address(
                x86_64::PhysAddr::new(phys.as_u64()),
            )
            .map_err(|_| MapError::InvalidAddress)?;

            unsafe {
                mapper
                    .map_to_with_table_flags(page, frame, x86_64_flags, parent_flags, &mut adapter)
                    .map_err(map_to_error)?
                    .ignore();
            }
        }
    }

    Ok(())
}

fn update_flags_impl(
    mapper: &mut impl Mapper<Size4KiB>,
    virt: VirtAddr,
    flags: PageFlags,
) -> Result<(), MapError> {
    let page = Page::<Size4KiB>::containing_address(x86_64::VirtAddr::new(virt.as_u64()));
    let x86_64_flags = to_x86_64_flags(flags);
    unsafe {
        mapper
            .update_flags(page, x86_64_flags)
            .map_err(|e| match e {
                paging::mapper::FlagUpdateError::PageNotMapped => MapError::NotMapped,
                paging::mapper::FlagUpdateError::ParentEntryHugePage => MapError::ParentEntryHugePage,
            })?
            .ignore();
    }
    Ok(())
}

fn unmap_impl(
    mapper: &mut impl Mapper<Size4KiB>,
    virt: VirtAddr,
) -> Result<(PhysAddr, X86_64PageSize), UnmapError> {
    let page = Page::<Size4KiB>::containing_address(x86_64::VirtAddr::new(virt.as_u64()));
    let (frame, flush) = mapper.unmap(page).map_err(|e| match e {
        paging::mapper::UnmapError::ParentEntryHugePage => UnmapError::ParentEntryHugePage,
        paging::mapper::UnmapError::PageNotMapped => UnmapError::NotMapped,
        paging::mapper::UnmapError::InvalidFrameAddress(_) => UnmapError::NotMapped,
    })?;
    flush.ignore();
    Ok((
        PhysAddr::new(frame.start_address().as_u64()),
        X86_64PageSize::Size4KiB,
    ))
}

fn translate_impl(
    translator: &impl Translate,
    virt: VirtAddr,
) -> Option<(PhysAddr, PageFlags, X86_64PageSize)> {
    let addr = x86_64::VirtAddr::new(virt.as_u64());
    match translator.translate(addr) {
        TranslateResult::Mapped { frame, offset: _, flags } => {
            let (phys, size) = match frame {
                MappedFrame::Size4KiB(f) => {
                    (f.start_address().as_u64(), X86_64PageSize::Size4KiB)
                }
                MappedFrame::Size2MiB(f) => {
                    (f.start_address().as_u64(), X86_64PageSize::Size2MiB)
                }
                MappedFrame::Size1GiB(f) => {
                    // We don't use 1GiB pages in our page size enum,
                    // but report the mapping correctly.
                    (f.start_address().as_u64(), X86_64PageSize::Size2MiB)
                }
            };
            Some((PhysAddr::new(phys), from_x86_64_flags(flags), size))
        }
        TranslateResult::NotMapped | TranslateResult::InvalidFrameAddress(_) => None,
    }
}

impl PageTable<X86_64PageSize> for OffsetPageTable<'_> {
    fn map(
        &mut self,
        virt: VirtAddr,
        phys: PhysAddr,
        page_size: X86_64PageSize,
        flags: PageFlags,
        allocator: &mut dyn FrameAllocator<X86_64PageSize>,
    ) -> Result<(), MapError> {
        map_impl(self, virt, phys, page_size, flags, allocator)
    }

    fn update_flags(&mut self, virt: VirtAddr, flags: PageFlags) -> Result<(), MapError> {
        update_flags_impl(self, virt, flags)
    }

    fn unmap(&mut self, virt: VirtAddr) -> Result<(PhysAddr, X86_64PageSize), UnmapError> {
        unmap_impl(self, virt)
    }

    fn translate(&self, virt: VirtAddr) -> Option<(PhysAddr, PageFlags, X86_64PageSize)> {
        translate_impl(self, virt)
    }
}

impl<F: PageTableFrameMapping> PageTable<X86_64PageSize> for MappedPageTable<'_, F> {
    fn map(
        &mut self,
        virt: VirtAddr,
        phys: PhysAddr,
        page_size: X86_64PageSize,
        flags: PageFlags,
        allocator: &mut dyn FrameAllocator<X86_64PageSize>,
    ) -> Result<(), MapError> {
        map_impl(self, virt, phys, page_size, flags, allocator)
    }

    fn update_flags(&mut self, virt: VirtAddr, flags: PageFlags) -> Result<(), MapError> {
        update_flags_impl(self, virt, flags)
    }

    fn unmap(&mut self, virt: VirtAddr) -> Result<(PhysAddr, X86_64PageSize), UnmapError> {
        unmap_impl(self, virt)
    }

    fn translate(&self, virt: VirtAddr) -> Option<(PhysAddr, PageFlags, X86_64PageSize)> {
        translate_impl(self, virt)
    }
}

/// Adapter to use a [`FrameAllocator`] as an x86_64
/// [`FrameAllocator<Size4KiB>`](X86_64FrameAllocatorTrait).
///
/// This is needed because the x86_64 crate's `map_to` methods require
/// an `impl FrameAllocator<Size4KiB>` for allocating intermediate page tables.
struct FrameAllocAdapter<'a>(&'a mut dyn FrameAllocator<X86_64PageSize>);

unsafe impl X86_64FrameAllocatorTrait<Size4KiB> for FrameAllocAdapter<'_> {
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        let addr = self.0.allocate_frame(X86_64PageSize::Size4KiB)?;
        Some(PhysFrame::from_start_address(x86_64::PhysAddr::new(addr.as_u64())).unwrap())
    }
}
