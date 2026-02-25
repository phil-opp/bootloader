//! Bridge between bootloader-specific types and kernel-elf-loader's x86_64 module.
//!
//! Re-exports [`X86_64PageSize`] from kernel-elf-loader,
//! and provides [`X86_64FrameAllocator`] which wraps the bootloader's
//! [`LegacyFrameAllocator`](crate::legacy_memory_region::LegacyFrameAllocator).

pub use kernel_elf_loader::x86_64::X86_64PageSize;

use kernel_elf_loader::{self as kel, PhysAddr};
use x86_64::structures::paging::{
    FrameAllocator as X86_64FrameAllocatorTrait, PhysFrame, Size4KiB,
};

/// Wraps a [`LegacyFrameAllocator`](crate::legacy_memory_region::LegacyFrameAllocator)
/// to implement [`kernel_elf_loader::FrameAllocator`].
pub struct X86_64FrameAllocator<'a, I, D> {
    inner: &'a mut crate::legacy_memory_region::LegacyFrameAllocator<I, D>,
}

impl<'a, I, D> X86_64FrameAllocator<'a, I, D> {
    pub fn new(inner: &'a mut crate::legacy_memory_region::LegacyFrameAllocator<I, D>) -> Self {
        Self { inner }
    }
}

impl<I, D> kel::FrameAllocator<X86_64PageSize> for X86_64FrameAllocator<'_, I, D>
where
    I: ExactSizeIterator<Item = D> + Clone,
    D: crate::legacy_memory_region::LegacyMemoryRegion,
{
    fn allocate_frame(&mut self, size: X86_64PageSize) -> Option<PhysAddr> {
        match size {
            X86_64PageSize::Size4KiB => {
                let frame: PhysFrame<Size4KiB> =
                    X86_64FrameAllocatorTrait::<Size4KiB>::allocate_frame(self.inner)?;
                Some(PhysAddr::new(frame.start_address().as_u64()))
            }
            X86_64PageSize::Size2MiB => {
                // For 2MiB pages we need to allocate 512 contiguous 4KiB frames.
                // Check alignment of the first frame.
                let first: PhysFrame<Size4KiB> =
                    X86_64FrameAllocatorTrait::<Size4KiB>::allocate_frame(self.inner)?;
                let first_addr = first.start_address().as_u64();

                if first_addr % (2 * 1024 * 1024) != 0 {
                    // Not aligned — we can't guarantee contiguous 2MiB.
                    // Fall back: allocate remaining frames to "consume" them,
                    // then try again. This is a simplification.
                    return None;
                }

                // Allocate the remaining 511 frames.
                for _ in 1..512 {
                    X86_64FrameAllocatorTrait::<Size4KiB>::allocate_frame(self.inner)?;
                }

                Some(PhysAddr::new(first_addr))
            }
        }
    }
}
