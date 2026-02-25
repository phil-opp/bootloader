//! Trait definitions for platform-agnostic ELF loading.
//!
//! Each architecture implements these traits to bridge between the loader
//! and the platform's page table format, frame allocator, and physical
//! memory access model.

use crate::error::{MapError, UnmapError};

/// A physical address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct PhysAddr(pub u64);

impl PhysAddr {
    /// Create a new physical address.
    pub fn new(addr: u64) -> Self {
        Self(addr)
    }

    /// Return the raw address value.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Check whether this address is aligned to `alignment`.
    pub fn is_aligned(self, alignment: u64) -> bool {
        self.0 % alignment == 0
    }

    /// Round this address down to the nearest multiple of `alignment`.
    pub fn align_down(self, alignment: u64) -> Self {
        Self(self.0 / alignment * alignment)
    }

    /// Round this address up to the nearest multiple of `alignment`.
    pub fn align_up(self, alignment: u64) -> Self {
        Self((self.0 + alignment - 1) / alignment * alignment)
    }
}

/// A virtual address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct VirtAddr(pub u64);

impl VirtAddr {
    /// Create a new virtual address.
    pub fn new(addr: u64) -> Self {
        Self(addr)
    }

    /// Return the raw address value.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Check whether this address is aligned to `alignment`.
    pub fn is_aligned(self, alignment: u64) -> bool {
        self.0 % alignment == 0
    }

    /// Round this address down to the nearest multiple of `alignment`.
    pub fn align_down(self, alignment: u64) -> Self {
        Self(self.0 / alignment * alignment)
    }

    /// Round this address up to the nearest multiple of `alignment`.
    pub fn align_up(self, alignment: u64) -> Self {
        Self((self.0 + alignment - 1) / alignment * alignment)
    }
}

/// Flags for page table entries.
#[derive(Clone, Copy, Debug, Default)]
pub struct PageFlags {
    /// Whether the page is writable.
    pub writable: bool,
    /// Whether the page is executable.
    pub executable: bool,
}

/// Platform-defined page size.
///
/// Each architecture implements this for its own enum of supported
/// page sizes (e.g. x86_64: 4KiB/2MiB/1GiB, AArch64 4K granule:
/// 4KiB/2MiB/1GiB, RISC-V Sv39: 4KiB/2MiB/1GiB, etc.).
pub trait PageSize: Copy + Eq + 'static {
    /// Size of this page in bytes. Must be a power of two.
    fn bytes(self) -> u64;

    /// The smallest page size supported by the platform.
    /// The loader uses this as the base granularity for partial-page
    /// BSS handling, copy-on-write, etc.
    const BASE: Self;

    /// All page sizes **in descending order** that the platform supports
    /// for huge-page mappings.
    ///
    /// Only consulted when the caller opts into huge pages. The loader
    /// walks this list and uses the largest size whose alignment is
    /// satisfied by both the virtual and physical address and that
    /// fits in the remaining region.
    ///
    /// Set to `&[Self::BASE]` if huge pages are not desired.
    const HUGE_PAGE_SIZES: &[Self];
}

/// Allocates physical memory frames.
///
/// The caller (BIOS/UEFI bootloader) constructs this from the firmware
/// memory map, ensuring only usable physical memory is handed out.
pub trait FrameAllocator<S: PageSize> {
    /// Allocate a physical frame of the given size.
    /// The returned address must be aligned to `size.bytes()`.
    fn allocate_frame(&mut self, size: S) -> Option<PhysAddr>;
}

/// Abstraction over a page table hierarchy.
///
/// All operations work on the **kernel's** page table (not the currently
/// active one). Generic over the platform's page size type.
///
/// # Safety
///
/// Implementors must guarantee that the page table represented by
/// this type is **not** the currently active page table. Modifying
/// the active page table can cause immediate undefined behavior
/// (e.g. invalidating the mapping the CPU is executing from).
pub unsafe trait PageTable<S: PageSize> {
    /// Map a virtual page to a physical frame.
    ///
    /// `virt` and `phys` must be aligned to `page_size.bytes()`.
    fn map(
        &mut self,
        virt: VirtAddr,
        phys: PhysAddr,
        page_size: S,
        flags: PageFlags,
        allocator: &mut dyn FrameAllocator<S>,
    ) -> Result<(), MapError>;

    /// Change the flags of an existing mapping without changing the
    /// physical frame.
    fn update_flags(
        &mut self,
        virt: VirtAddr,
        flags: PageFlags,
    ) -> Result<(), MapError>;

    /// Unmap a virtual page. Returns the physical frame and page size
    /// it was mapped to.
    fn unmap(&mut self, virt: VirtAddr) -> Result<(PhysAddr, S), UnmapError>;

    /// Translate a virtual address to (physical address, flags, page size).
    fn translate(&self, virt: VirtAddr) -> Option<(PhysAddr, PageFlags, S)>;
}

/// Access to physical memory from the bootloader environment.
///
/// In a typical identity-mapped bootloader, `phys_addr == virt_addr`,
/// so these are trivial pointer dereferences. This trait exists so
/// the loader doesn't hard-code that assumption.
///
/// # Safety
///
/// Implementors must guarantee that **all valid physical addresses**
/// (i.e., addresses returned by [`FrameAllocator::allocate_frame`] or
/// obtained via [`PageTable::translate`]) are accessible through this
/// trait's methods. The loader will only ever pass such addresses to
/// these methods.
pub unsafe trait PhysicalMemoryAccess {
    /// Read `buf.len()` bytes from physical address `addr` into `buf`.
    fn read_phys(&self, addr: PhysAddr, buf: &mut [u8]);

    /// Write `buf.len()` bytes from `buf` to physical address `addr`.
    fn write_phys(&self, addr: PhysAddr, buf: &[u8]);

    /// Zero `len` bytes at physical address `addr`.
    fn zero_phys(&self, addr: PhysAddr, len: usize);

    /// Copy `len` bytes from physical address `src` to `dst`.
    ///
    /// The source and destination ranges must not overlap.
    fn copy_phys(&self, src: PhysAddr, dst: PhysAddr, len: usize);
}
