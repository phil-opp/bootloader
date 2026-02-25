//! Bridge between x86_64 crate types and kernel-elf-loader traits.

use kernel_elf_loader::{
    self as kel,
    error::{MapError, UnmapError},
};
use x86_64::structures::paging::{
    FrameAllocator as X86FrameAllocatorTrait, Mapper, OffsetPageTable, Page,
    PageTableFlags, PhysFrame, Size2MiB, Size4KiB, Translate,
    mapper::{MappedFrame, TranslateResult},
};

/// x86_64 page sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum X86PageSize {
    Size4KiB,
    Size2MiB,
}

impl kel::PageSize for X86PageSize {
    fn bytes(self) -> u64 {
        match self {
            Self::Size4KiB => 4096,
            Self::Size2MiB => 2 * 1024 * 1024,
        }
    }

    const BASE: Self = Self::Size4KiB;
    const HUGE_PAGE_SIZES: &[Self] = &[Self::Size2MiB];
}

/// Wraps an `OffsetPageTable` to implement `kernel_elf_loader::PageTable`.
pub struct X86PageTable<'a> {
    inner: &'a mut OffsetPageTable<'static>,
}

impl<'a> X86PageTable<'a> {
    pub fn new(inner: &'a mut OffsetPageTable<'static>) -> Self {
        Self { inner }
    }

    /// Get access to the underlying `OffsetPageTable` for x86-specific operations.
    pub fn inner_mut(&mut self) -> &mut OffsetPageTable<'static> {
        self.inner
    }
}

/// Convert `kernel_elf_loader::PageFlags` to x86_64 `PageTableFlags`.
fn to_x86_flags(flags: kel::PageFlags) -> PageTableFlags {
    let mut f = PageTableFlags::PRESENT;
    if flags.writable {
        f |= PageTableFlags::WRITABLE;
    }
    if !flags.executable {
        f |= PageTableFlags::NO_EXECUTE;
    }
    f
}

/// Convert x86_64 `PageTableFlags` to `kernel_elf_loader::PageFlags`.
fn from_x86_flags(flags: PageTableFlags) -> kel::PageFlags {
    kel::PageFlags {
        writable: flags.contains(PageTableFlags::WRITABLE),
        executable: !flags.contains(PageTableFlags::NO_EXECUTE),
    }
}

impl kel::PageTable<X86PageSize> for X86PageTable<'_> {
    fn map(
        &mut self,
        virt: kel::VirtAddr,
        phys: kel::PhysAddr,
        page_size: X86PageSize,
        flags: kel::PageFlags,
        allocator: &mut dyn kel::FrameAllocator<X86PageSize>,
    ) -> Result<(), MapError> {
        let x86_flags = to_x86_flags(flags);
        // Parent table flags need to be both readable and writable to
        // support recursive page tables.
        // See https://github.com/rust-osdev/bootloader/issues/443#issuecomment-2130010621
        let parent_flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;

        let mut adapter = FrameAllocAdapter(allocator);

        match page_size {
            X86PageSize::Size4KiB => {
                let page = Page::<Size4KiB>::from_start_address(
                    x86_64::VirtAddr::new(virt.as_u64()),
                )
                .map_err(|_| MapError::AlreadyMapped)?;
                let frame = PhysFrame::<Size4KiB>::from_start_address(
                    x86_64::PhysAddr::new(phys.as_u64()),
                )
                .map_err(|_| MapError::FrameAllocationFailed)?;

                unsafe {
                    self.inner
                        .map_to_with_table_flags(page, frame, x86_flags, parent_flags, &mut adapter)
                        .map_err(|_| MapError::AlreadyMapped)?
                        .ignore();
                }
            }
            X86PageSize::Size2MiB => {
                let page = Page::<Size2MiB>::from_start_address(
                    x86_64::VirtAddr::new(virt.as_u64()),
                )
                .map_err(|_| MapError::AlreadyMapped)?;
                let frame = PhysFrame::<Size2MiB>::from_start_address(
                    x86_64::PhysAddr::new(phys.as_u64()),
                )
                .map_err(|_| MapError::FrameAllocationFailed)?;

                unsafe {
                    self.inner
                        .map_to_with_table_flags(page, frame, x86_flags, parent_flags, &mut adapter)
                        .map_err(|_| MapError::AlreadyMapped)?
                        .ignore();
                }
            }
        }

        Ok(())
    }

    fn update_flags(
        &mut self,
        virt: kel::VirtAddr,
        flags: kel::PageFlags,
    ) -> Result<(), MapError> {
        let page = Page::<Size4KiB>::containing_address(x86_64::VirtAddr::new(virt.as_u64()));
        let x86_flags = to_x86_flags(flags);
        unsafe {
            self.inner
                .update_flags(page, x86_flags)
                .map_err(|_| MapError::AlreadyMapped)?
                .ignore();
        }
        Ok(())
    }

    fn unmap(&mut self, virt: kel::VirtAddr) -> Result<(kel::PhysAddr, X86PageSize), UnmapError> {
        let page = Page::<Size4KiB>::containing_address(x86_64::VirtAddr::new(virt.as_u64()));
        let (frame, flush) = self.inner.unmap(page).map_err(|_| UnmapError::NotMapped)?;
        flush.ignore();
        Ok((
            kel::PhysAddr::new(frame.start_address().as_u64()),
            X86PageSize::Size4KiB,
        ))
    }

    fn translate(
        &self,
        virt: kel::VirtAddr,
    ) -> Option<(kel::PhysAddr, kel::PageFlags, X86PageSize)> {
        let addr = x86_64::VirtAddr::new(virt.as_u64());
        match self.inner.translate(addr) {
            TranslateResult::Mapped { frame, offset: _, flags } => {
                let (phys, size) = match frame {
                    MappedFrame::Size4KiB(f) => {
                        (f.start_address().as_u64(), X86PageSize::Size4KiB)
                    }
                    MappedFrame::Size2MiB(f) => {
                        (f.start_address().as_u64(), X86PageSize::Size2MiB)
                    }
                    MappedFrame::Size1GiB(f) => {
                        // We don't use 1GiB pages in our page size enum,
                        // but report the mapping correctly.
                        (f.start_address().as_u64(), X86PageSize::Size2MiB)
                    }
                };
                Some((kel::PhysAddr::new(phys), from_x86_flags(flags), size))
            }
            TranslateResult::NotMapped | TranslateResult::InvalidFrameAddress(_) => None,
        }
    }
}

/// Adapter to use a `kernel_elf_loader::FrameAllocator` as an x86_64 `FrameAllocator`.
///
/// This is needed because the x86_64 crate's `map_to` methods require
/// an `impl FrameAllocator<Size4KiB>` for allocating intermediate page tables.
struct FrameAllocAdapter<'a>(&'a mut dyn kel::FrameAllocator<X86PageSize>);

unsafe impl X86FrameAllocatorTrait<Size4KiB> for FrameAllocAdapter<'_> {
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        let addr = self.0.allocate_frame(X86PageSize::Size4KiB)?;
        Some(PhysFrame::from_start_address(x86_64::PhysAddr::new(addr.as_u64())).unwrap())
    }
}

/// Wraps a `LegacyFrameAllocator` to implement `kernel_elf_loader::FrameAllocator`.
pub struct X86FrameAllocator<'a, I, D> {
    inner: &'a mut crate::legacy_memory_region::LegacyFrameAllocator<I, D>,
}

impl<'a, I, D> X86FrameAllocator<'a, I, D> {
    pub fn new(inner: &'a mut crate::legacy_memory_region::LegacyFrameAllocator<I, D>) -> Self {
        Self { inner }
    }
}

impl<I, D> kel::FrameAllocator<X86PageSize> for X86FrameAllocator<'_, I, D>
where
    I: ExactSizeIterator<Item = D> + Clone,
    D: crate::legacy_memory_region::LegacyMemoryRegion,
{
    fn allocate_frame(&mut self, size: X86PageSize) -> Option<kel::PhysAddr> {
        match size {
            X86PageSize::Size4KiB => {
                let frame: PhysFrame<Size4KiB> =
                    X86FrameAllocatorTrait::<Size4KiB>::allocate_frame(self.inner)?;
                Some(kel::PhysAddr::new(frame.start_address().as_u64()))
            }
            X86PageSize::Size2MiB => {
                // For 2MiB pages we need to allocate 512 contiguous 4KiB frames.
                // Check alignment of the first frame.
                let first: PhysFrame<Size4KiB> =
                    X86FrameAllocatorTrait::<Size4KiB>::allocate_frame(self.inner)?;
                let first_addr = first.start_address().as_u64();

                if first_addr % (2 * 1024 * 1024) != 0 {
                    // Not aligned — we can't guarantee contiguous 2MiB.
                    // Fall back: allocate remaining frames to "consume" them,
                    // then try again. This is a simplification.
                    return None;
                }

                // Allocate the remaining 511 frames.
                for _ in 1..512 {
                    X86FrameAllocatorTrait::<Size4KiB>::allocate_frame(self.inner)?;
                }

                Some(kel::PhysAddr::new(first_addr))
            }
        }
    }
}

