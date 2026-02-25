//! Platform-agnostic ELF kernel loader for bootloaders.
//!
//! This crate provides the [`Loader`] type that can load a kernel ELF
//! binary into a new page table hierarchy via generic traits. It supports:
//!
//! - ET_EXEC and ET_DYN (PIE) kernel binaries
//! - RELA and RELR (compact) relocations
//! - TLS template extraction
//! - BSS zeroing with copy-on-write
//! - RELRO segment protection
//! - Mapping additional regions (framebuffer, ramdisk, physical memory, stack)
//! - Guard pages for stack overflow detection
//! - ASLR via optional RNG
//! - Huge page support via generic `PageSize` trait
//!
//! # Platform Integration
//!
//! To use this crate, the platform (e.g. x86_64 bootloader) must implement:
//!
//! - [`PageSize`] — enumerate the supported page sizes
//! - [`PageTable`] — abstract page table operations
//! - [`FrameAllocator`] — allocate physical frames
//! - [`PhysicalMemoryAccess`] — read/write/zero/copy physical memory
//!
//! # Cargo Features
//!
//! - **`x86_64`** — Enables the [`x86_64`] module with ready-made
//!   [`PageSize`] and [`PageTable`] implementations backed by the
//!   [`x86_64`](::x86_64) crate.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

// === Core types and traits ===

/// Platform-abstraction traits for page tables, frame allocation, and
/// physical memory access.
pub(crate) mod traits;

/// Error types for the loader.
pub(crate) mod error;

// === Internal implementation ===

mod elf_loading;
mod relocation;

// === Public modules ===

/// Virtual address space tracking.
pub(crate) mod address_space;

/// Identity-mapped physical memory access implementation.
pub(crate) mod identity_mapped;

/// The main [`Loader`] type for loading kernel ELFs and mapping memory
/// regions.
pub(crate) mod loader;

// === Platform-specific (feature-gated) ===

/// x86_64 platform implementation (requires the `x86_64` Cargo feature).
#[cfg(feature = "x86_64")]
pub mod x86_64;

// Re-export key types at the crate root.
pub use address_space::AddressSpace;
pub use error::{LoadError, MapError, UnmapError};
pub use identity_mapped::IdentityMappedAccess;
pub use loader::{KernelPlacement, Loader, RegionPlacement};
pub use traits::{
    FrameAllocator, PageFlags, PageSize, PageTable, PhysAddr, PhysicalMemoryAccess, VirtAddr,
};

/// Align `value` up to the next multiple of `alignment`.
///
/// `alignment` must be non-zero.
pub(crate) fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) / alignment * alignment
}

/// Result of loading a kernel ELF.
#[derive(Debug)]
pub struct LoadedKernel {
    /// The kernel's entry point virtual address.
    pub entry_point: VirtAddr,
    /// Offset applied to ELF virtual addresses (0 for ET_EXEC).
    pub load_base: i64,
    /// TLS template, if the kernel has a TLS segment.
    pub tls_template: Option<TlsTemplate>,
}

/// Thread Local Storage template information.
#[derive(Clone, Copy, Debug)]
pub struct TlsTemplate {
    /// Virtual address of the TLS template data.
    pub start_addr: VirtAddr,
    /// Size of initialized TLS data (from the file).
    pub file_size: u64,
    /// Total size of the TLS area in memory (includes uninitialized portion).
    pub mem_size: u64,
}
