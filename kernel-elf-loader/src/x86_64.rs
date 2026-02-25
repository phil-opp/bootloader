//! x86_64 trait implementations for the kernel ELF loader.
//!
//! Implements the loader's [`PageSize`](crate::PageSize) and
//! [`PageTable`](crate::PageTable) traits for the `x86_64` crate's
//! page table types ([`OffsetPageTable`] and [`MappedPageTable`]),
//! wrapped in [`InactivePageTable`] to enforce the safety invariant
//! that the page table is not currently active.
//!
//! Two page-size configurations are provided:
//!
//! - [`X86_64PageSize`] — 4 KiB + 2 MiB huge pages.
//! - [`X86_64PageSizeWithGiB`] — 4 KiB + 2 MiB + 1 GiB huge pages.
//!
//! Enable the `x86_64` Cargo feature to use this module.

use core::ops::{Deref, DerefMut};

use crate::{
    FrameAllocator, PageFlags, PageSize, PageTable, PhysAddr, VirtAddr,
    error::{MapError, UnmapError},
};
use x86_64::structures::paging::{
    self, FrameAllocator as X86_64FrameAllocatorTrait, MappedPageTable, Mapper, OffsetPageTable,
    Page, PageTableFlags, PhysFrame, Size1GiB, Size2MiB, Size4KiB, Translate,
    mapper::{MapToError, MappedFrame, PageTableFrameMapping, TranslateResult},
};

// ---------------------------------------------------------------------------
// Page size enums
// ---------------------------------------------------------------------------

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
// Inactive page table wrapper
// ---------------------------------------------------------------------------

/// A wrapper asserting that the contained page table is **not** the
/// currently active one.
///
/// This is the type that implements [`PageTable`]. Construct it via
/// [`InactivePageTable::new`] to assert the safety invariant.
///
/// `T` is typically [`OffsetPageTable`] or [`MappedPageTable`].
pub struct InactivePageTable<T>(T);

impl<T> InactivePageTable<T> {
    /// Wrap a page table mapper, asserting it is not currently active.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `inner` operates on a page table
    /// hierarchy that is **not** loaded into any CPU's CR3 register.
    /// Modifying the active page table can cause immediate undefined
    /// behavior.
    pub unsafe fn new(inner: T) -> Self {
        Self(inner)
    }

    /// Unwrap and return the inner mapper.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> Deref for InactivePageTable<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for InactivePageTable<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
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
        MapToError::ParentEntryHugePage => MapError::UnsupportedPageSize,
        MapToError::PageAlreadyMapped(_) => MapError::AlreadyMapped,
    }
}

fn convert_unmap_error(e: paging::mapper::UnmapError) -> UnmapError {
    match e {
        paging::mapper::UnmapError::ParentEntryHugePage => UnmapError::UnsupportedPageSize,
        paging::mapper::UnmapError::PageNotMapped => UnmapError::NotMapped,
        paging::mapper::UnmapError::InvalidFrameAddress(_) => UnmapError::NotMapped,
    }
}

// ---------------------------------------------------------------------------
// Generic per-x86_64-page-size helpers
// ---------------------------------------------------------------------------

/// Parent table flags: present + writable to support recursive page tables.
/// See <https://github.com/rust-osdev/bootloader/issues/443#issuecomment-2130010621>
const PARENT_FLAGS: PageTableFlags = PageTableFlags::PRESENT.union(PageTableFlags::WRITABLE);

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

/// Unmap a single page of compile-time size `S`.
fn unmap_size<S: paging::PageSize>(
    mapper: &mut impl Mapper<S>,
    virt: VirtAddr,
) -> Result<PhysAddr, UnmapError> {
    let page = Page::<S>::containing_address(x86_64::VirtAddr::new(virt.as_u64()));
    let (frame, flush) = mapper.unmap(page).map_err(convert_unmap_error)?;
    flush.ignore();
    Ok(PhysAddr::new(frame.start_address().as_u64()))
}

// ---------------------------------------------------------------------------
// Shared translate helper
// ---------------------------------------------------------------------------

fn translate_frame(
    translator: &impl Translate,
    virt: VirtAddr,
) -> Option<(PhysAddr, PageFlags, MappedFrame)> {
    let addr = x86_64::VirtAddr::new(virt.as_u64());
    match translator.translate(addr) {
        TranslateResult::Mapped {
            frame,
            offset: _,
            flags,
        } => {
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
// Shared update_flags helper
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
                paging::mapper::FlagUpdateError::ParentEntryHugePage => {
                    MapError::UnsupportedPageSize
                }
            })?
            .ignore();
    }
    Ok(())
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
// PageTable impl for X86_64PageSize (4 KiB + 2 MiB)
// ---------------------------------------------------------------------------

macro_rules! impl_page_table_2m {
    ($mapper:ty $(, $generics:tt : $bound:path)*) => {
        // SAFETY: The `InactivePageTable` wrapper's constructor requires the
        // caller to guarantee that the page table is not currently active.
        unsafe impl<$($generics: $bound),*> PageTable<X86_64PageSize> for InactivePageTable<$mapper> {
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
                let mapper = &mut **self;
                match page_size {
                    X86_64PageSize::Size4KiB => {
                        map_size::<Size4KiB>(mapper, virt, phys, x86_64_flags, PARENT_FLAGS, &mut adapter)
                    }
                    X86_64PageSize::Size2MiB => {
                        map_size::<Size2MiB>(mapper, virt, phys, x86_64_flags, PARENT_FLAGS, &mut adapter)
                    }
                }
            }

            fn update_flags(&mut self, virt: VirtAddr, flags: PageFlags) -> Result<(), MapError> {
                update_flags_impl(&mut **self, virt, flags)
            }

            fn unmap(&mut self, virt: VirtAddr) -> Result<(PhysAddr, X86_64PageSize), UnmapError> {
                let (_, _, frame) = translate_frame(&**self, virt)
                    .ok_or(UnmapError::NotMapped)?;
                let mapper = &mut **self;
                match frame {
                    MappedFrame::Size4KiB(_) => {
                        unmap_size::<Size4KiB>(mapper, virt)
                            .map(|p| (p, X86_64PageSize::Size4KiB))
                    }
                    MappedFrame::Size2MiB(_) => {
                        unmap_size::<Size2MiB>(mapper, virt)
                            .map(|p| (p, X86_64PageSize::Size2MiB))
                    }
                    MappedFrame::Size1GiB(_) => Err(UnmapError::UnsupportedPageSize),
                }
            }

            fn translate(&self, virt: VirtAddr) -> Option<(PhysAddr, PageFlags, X86_64PageSize)> {
                let (phys, flags, frame) = translate_frame(&**self, virt)?;
                let size = match frame {
                    MappedFrame::Size4KiB(_) => X86_64PageSize::Size4KiB,
                    MappedFrame::Size2MiB(_) => X86_64PageSize::Size2MiB,
                    MappedFrame::Size1GiB(_) => return None,
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
        // SAFETY: The `InactivePageTable` wrapper's constructor requires the
        // caller to guarantee that the page table is not currently active.
        unsafe impl<$($generics: $bound),*> PageTable<X86_64PageSizeWithGiB> for InactivePageTable<$mapper> {
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
                let mapper = &mut **self;
                match page_size {
                    X86_64PageSizeWithGiB::Size4KiB => {
                        map_size::<Size4KiB>(mapper, virt, phys, x86_64_flags, PARENT_FLAGS, &mut adapter)
                    }
                    X86_64PageSizeWithGiB::Size2MiB => {
                        map_size::<Size2MiB>(mapper, virt, phys, x86_64_flags, PARENT_FLAGS, &mut adapter)
                    }
                    X86_64PageSizeWithGiB::Size1GiB => {
                        map_size::<Size1GiB>(mapper, virt, phys, x86_64_flags, PARENT_FLAGS, &mut adapter)
                    }
                }
            }

            fn update_flags(&mut self, virt: VirtAddr, flags: PageFlags) -> Result<(), MapError> {
                update_flags_impl(&mut **self, virt, flags)
            }

            fn unmap(&mut self, virt: VirtAddr) -> Result<(PhysAddr, X86_64PageSizeWithGiB), UnmapError> {
                let (_, _, frame) = translate_frame(&**self, virt)
                    .ok_or(UnmapError::NotMapped)?;
                let mapper = &mut **self;
                match frame {
                    MappedFrame::Size4KiB(_) => {
                        unmap_size::<Size4KiB>(mapper, virt)
                            .map(|p| (p, X86_64PageSizeWithGiB::Size4KiB))
                    }
                    MappedFrame::Size2MiB(_) => {
                        unmap_size::<Size2MiB>(mapper, virt)
                            .map(|p| (p, X86_64PageSizeWithGiB::Size2MiB))
                    }
                    MappedFrame::Size1GiB(_) => {
                        unmap_size::<Size1GiB>(mapper, virt)
                            .map(|p| (p, X86_64PageSizeWithGiB::Size1GiB))
                    }
                }
            }

            fn translate(&self, virt: VirtAddr) -> Option<(PhysAddr, PageFlags, X86_64PageSizeWithGiB)> {
                let (phys, flags, frame) = translate_frame(&**self, virt)?;
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
