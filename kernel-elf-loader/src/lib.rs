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

#![cfg_attr(not(test), no_std)]

pub mod address_space;
pub mod error;
pub mod traits;

mod elf_loading;
mod relocation;

pub mod loader;

// Re-export key types at the crate root.
pub use address_space::AddressSpace;
pub use error::{LoadError, MapError, UnmapError};
pub use loader::{KernelPlacement, Loader, RegionPlacement};
pub use traits::{FrameAllocator, PageFlags, PageSize, PageTable, PhysAddr, PhysicalMemoryAccess, VirtAddr};

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
