//! x86_64 trait implementations for the kernel ELF loader.
//!
//! Implements the loader's [`PageSize`](crate::PageSize) and
//! [`PageTable`](crate::PageTable) traits for the `x86_64` crate's
//! page table types ([`OffsetPageTable`] and [`MappedPageTable`]).
//!
//! Three page-size configurations are provided:
//!
//! - [`X86_64BasePageSize`] — 4 KiB only (no huge pages).
//! - [`X86_64PageSize`] — 4 KiB + 2 MiB huge pages.
//! - [`X86_64PageSizeWithGiB`] — 4 KiB + 2 MiB + 1 GiB huge pages.
//!
//! Enable the `x86_64` Cargo feature to use this module.

use crate::{
    error::{MapError, UnmapError},
    FrameAllocator, PageFlags, PageSize, PageTable, PhysAddr, VirtAddr,
};
use x86_64::structures::paging::{
    self,
    FrameAllocator as X86_64FrameAllocatorTrait, Mapper, MappedPageTable, OffsetPageTable, Page,
    PageTableFlags, PhysFrame, Size1GiB, Size2MiB, Size4KiB, Translate,
    mapper::{MappedFrame, MapToError, PageTableFrameMapping, TranslateResult},
};

// ---------------------------------------------------------------------------
// Page size enums
// ---------------------------------------------------------------------------

/// 4 KiB only — no huge pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum X86_64BasePageSize {
    /// 4 KiB base page.
    Size4KiB,
}

impl PageSize for X86_64BasePageSize {
    fn bytes(self) -> u64 {
        4096
    }

    const BASE: Self = Self::Size4KiB;
    const HUGE_PAGE_SIZES: &[Self] = &[];
}

/// 4 KiB + 2 MiB page sizes.
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

/// 4 KiB + 2 MiB + 1 GiB page sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum X86_64PageSizeWithGiB {
    /// 4 KiB base page.
    Size4KiB,
    /// 2 MiB huge page.
    Size2MiB,
    /// 1 GiB huge page.
    Size1GiB,
}

impl PageSize for X86_64PageSizeWithGiB {
    fn bytes(self) -> u64 {
        match self {
            Self::Size4KiB => 4096,
            Self::Size2MiB => 2 * 1024 * 1024,
            Self::Size1GiB => 1024 * 1024 * 1024,
        }
    }

    const BASE: Self = Self::Size4KiB;
    const HUGE_PAGE_SIZES: &[Self] = &[Self::Size1GiB, Self::Size2MiB];
}

// ---------------------------------------------------------------------------
// Flag conversion helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Generic per-x86_64-page-size mapping helper
// ---------------------------------------------------------------------------

/// Map a single page of compile-time size `S` (one of the x86_64 crate's
/// `Size4KiB`, `Size2MiB`, `Size1GiB`).
fn map_size<S: paging::PageSize>(
    mapper: &mut impl Mapper<S>,
    virt: VirtAddr,
    phys: PhysAddr,
    flags: PageTableFlags,
    parent_flags: PageTableFlags,
    allocator: &mut impl X86_64FrameAllocatorTrait<Size4KiB>,
) -> Result<(), MapError> {
    let page = Page::<S>::from_start_address(x86_64::VirtAddr::new(virt.as_u64()))
        .map_err(|_| MapError::InvalidAddress)?;
    let frame = PhysFrame::<S>::from_start_address(x86_64::PhysAddr::new(phys.as_u64()))
        .map_err(|_| MapError::InvalidAddress)?;

    unsafe {
        mapper
            .map_to_with_table_flags(page, frame, flags, parent_flags, allocator)
            .map_err(map_to_error)?
            .ignore();
    }
    Ok(())
}

/// Parent table flags: present + writable to support recursive page tables.
/// See <https://github.com/rust-osdev/bootloader/issues/443#issuecomment-2130010621>
const PARENT_FLAGS: PageTableFlags =
    PageTableFlags::PRESENT.union(PageTableFlags::WRITABLE);

// ---------------------------------------------------------------------------
// Shared implementations (update_flags, unmap, translate)
// ---------------------------------------------------------------------------

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

fn unmap_4kib(
    mapper: &mut impl Mapper<Size4KiB>,
    virt: VirtAddr,
) -> Result<PhysAddr, UnmapError> {
    let page = Page::<Size4KiB>::containing_address(x86_64::VirtAddr::new(virt.as_u64()));
    let (frame, flush) = mapper.unmap(page).map_err(|e| match e {
        paging::mapper::UnmapError::ParentEntryHugePage => UnmapError::ParentEntryHugePage,
        paging::mapper::UnmapError::PageNotMapped => UnmapError::NotMapped,
        paging::mapper::UnmapError::InvalidFrameAddress(_) => UnmapError::NotMapped,
    })?;
    flush.ignore();
    Ok(PhysAddr::new(frame.start_address().as_u64()))
}

fn translate_frame(
    translator: &impl Translate,
    virt: VirtAddr,
) -> Option<(PhysAddr, PageFlags, MappedFrame)> {
    let addr = x86_64::VirtAddr::new(virt.as_u64());
    match translator.translate(addr) {
        TranslateResult::Mapped { frame, offset: _, flags } => {
            let phys = match frame {
                MappedFrame::Size4KiB(f) => f.start_address().as_u64(),
                MappedFrame::Size2MiB(f) => f.start_address().as_u64(),
                MappedFrame::Size1GiB(f) => f.start_address().as_u64(),
            };
            Some((PhysAddr::new(phys), from_x86_64_flags(flags), frame))
        }
        TranslateResult::NotMapped | TranslateResult::InvalidFrameAddress(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Frame allocator adapter
// ---------------------------------------------------------------------------

/// Adapter to use a [`FrameAllocator`] as an x86_64
/// [`FrameAllocator<Size4KiB>`](X86_64FrameAllocatorTrait).
///
/// The x86_64 crate's `map_to` methods require an
/// `impl FrameAllocator<Size4KiB>` for allocating intermediate page tables.
struct FrameAllocAdapter<'a, S: PageSize>(&'a mut dyn FrameAllocator<S>);

unsafe impl<S: PageSize> X86_64FrameAllocatorTrait<Size4KiB> for FrameAllocAdapter<'_, S> {
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        let addr = self.0.allocate_frame(S::BASE)?;
        Some(PhysFrame::from_start_address(x86_64::PhysAddr::new(addr.as_u64())).unwrap())
    }
}

// ---------------------------------------------------------------------------
// PageTable impl for X86_64BasePageSize (4 KiB only)
// ---------------------------------------------------------------------------

macro_rules! impl_page_table_base {
    ($mapper:ty $(, $generics:tt : $bound:path)*) => {
        impl<$($generics: $bound),*> PageTable<X86_64BasePageSize> for $mapper {
            fn map(
                &mut self,
                virt: VirtAddr,
                phys: PhysAddr,
                _page_size: X86_64BasePageSize,
                flags: PageFlags,
                allocator: &mut dyn FrameAllocator<X86_64BasePageSize>,
            ) -> Result<(), MapError> {
                let mut adapter = FrameAllocAdapter(allocator);
                map_size::<Size4KiB>(
                    self, virt, phys,
                    to_x86_64_flags(flags), PARENT_FLAGS,
                    &mut adapter,
                )
            }

            fn update_flags(&mut self, virt: VirtAddr, flags: PageFlags) -> Result<(), MapError> {
                update_flags_impl(self, virt, flags)
            }

            fn unmap(&mut self, virt: VirtAddr) -> Result<(PhysAddr, X86_64BasePageSize), UnmapError> {
                unmap_4kib(self, virt).map(|p| (p, X86_64BasePageSize::Size4KiB))
            }

            fn translate(&self, virt: VirtAddr) -> Option<(PhysAddr, PageFlags, X86_64BasePageSize)> {
                let (phys, flags, _) = translate_frame(self, virt)?;
                Some((phys, flags, X86_64BasePageSize::Size4KiB))
            }
        }
    };
}

impl_page_table_base!(OffsetPageTable<'_>);
impl_page_table_base!(MappedPageTable<'_, F>, F: PageTableFrameMapping);

// ---------------------------------------------------------------------------
// PageTable impl for X86_64PageSize (4 KiB + 2 MiB)
// ---------------------------------------------------------------------------

macro_rules! impl_page_table_2m {
    ($mapper:ty $(, $generics:tt : $bound:path)*) => {
        impl<$($generics: $bound),*> PageTable<X86_64PageSize> for $mapper {
            fn map(
                &mut self,
                virt: VirtAddr,
                phys: PhysAddr,
                page_size: X86_64PageSize,
                flags: PageFlags,
                allocator: &mut dyn FrameAllocator<X86_64PageSize>,
            ) -> Result<(), MapError> {
                let x86_64_flags = to_x86_64_flags(flags);
                let mut adapter = FrameAllocAdapter(allocator);
                match page_size {
                    X86_64PageSize::Size4KiB => {
                        map_size::<Size4KiB>(self, virt, phys, x86_64_flags, PARENT_FLAGS, &mut adapter)
                    }
                    X86_64PageSize::Size2MiB => {
                        map_size::<Size2MiB>(self, virt, phys, x86_64_flags, PARENT_FLAGS, &mut adapter)
                    }
                }
            }

            fn update_flags(&mut self, virt: VirtAddr, flags: PageFlags) -> Result<(), MapError> {
                update_flags_impl(self, virt, flags)
            }

            fn unmap(&mut self, virt: VirtAddr) -> Result<(PhysAddr, X86_64PageSize), UnmapError> {
                unmap_4kib(self, virt).map(|p| (p, X86_64PageSize::Size4KiB))
            }

            fn translate(&self, virt: VirtAddr) -> Option<(PhysAddr, PageFlags, X86_64PageSize)> {
                let (phys, flags, frame) = translate_frame(self, virt)?;
                let size = match frame {
                    MappedFrame::Size4KiB(_) => X86_64PageSize::Size4KiB,
                    MappedFrame::Size2MiB(_) | MappedFrame::Size1GiB(_) => X86_64PageSize::Size2MiB,
                };
                Some((phys, flags, size))
            }
        }
    };
}

impl_page_table_2m!(OffsetPageTable<'_>);
impl_page_table_2m!(MappedPageTable<'_, F>, F: PageTableFrameMapping);

// ---------------------------------------------------------------------------
// PageTable impl for X86_64PageSizeWithGiB (4 KiB + 2 MiB + 1 GiB)
// ---------------------------------------------------------------------------

macro_rules! impl_page_table_1g {
    ($mapper:ty $(, $generics:tt : $bound:path)*) => {
        impl<$($generics: $bound),*> PageTable<X86_64PageSizeWithGiB> for $mapper {
            fn map(
                &mut self,
                virt: VirtAddr,
                phys: PhysAddr,
                page_size: X86_64PageSizeWithGiB,
                flags: PageFlags,
                allocator: &mut dyn FrameAllocator<X86_64PageSizeWithGiB>,
            ) -> Result<(), MapError> {
                let x86_64_flags = to_x86_64_flags(flags);
                let mut adapter = FrameAllocAdapter(allocator);
                match page_size {
                    X86_64PageSizeWithGiB::Size4KiB => {
                        map_size::<Size4KiB>(self, virt, phys, x86_64_flags, PARENT_FLAGS, &mut adapter)
                    }
                    X86_64PageSizeWithGiB::Size2MiB => {
                        map_size::<Size2MiB>(self, virt, phys, x86_64_flags, PARENT_FLAGS, &mut adapter)
                    }
                    X86_64PageSizeWithGiB::Size1GiB => {
                        map_size::<Size1GiB>(self, virt, phys, x86_64_flags, PARENT_FLAGS, &mut adapter)
                    }
                }
            }

            fn update_flags(&mut self, virt: VirtAddr, flags: PageFlags) -> Result<(), MapError> {
                update_flags_impl(self, virt, flags)
            }

            fn unmap(&mut self, virt: VirtAddr) -> Result<(PhysAddr, X86_64PageSizeWithGiB), UnmapError> {
                unmap_4kib(self, virt).map(|p| (p, X86_64PageSizeWithGiB::Size4KiB))
            }

            fn translate(&self, virt: VirtAddr) -> Option<(PhysAddr, PageFlags, X86_64PageSizeWithGiB)> {
                let (phys, flags, frame) = translate_frame(self, virt)?;
                let size = match frame {
                    MappedFrame::Size4KiB(_) => X86_64PageSizeWithGiB::Size4KiB,
                    MappedFrame::Size2MiB(_) => X86_64PageSizeWithGiB::Size2MiB,
                    MappedFrame::Size1GiB(_) => X86_64PageSizeWithGiB::Size1GiB,
                };
                Some((phys, flags, size))
            }
        }
    };
}

impl_page_table_1g!(OffsetPageTable<'_>);
impl_page_table_1g!(MappedPageTable<'_, F>, F: PageTableFrameMapping);
