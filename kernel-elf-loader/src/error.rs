//! Error types for the kernel ELF loader.

/// Errors that can occur during kernel loading or region mapping.
#[derive(Debug)]
pub enum LoadError {
    /// ELF parsing failed.
    InvalidElf(&'static str),
    /// ELF type not supported (not ET_EXEC or ET_DYN).
    UnsupportedElfType(u16),
    /// The ELF's machine type is not supported for relocations.
    UnsupportedMachine(u16),
    /// Relocation type not supported.
    UnsupportedRelocation(u32),
    /// Frame allocator returned None.
    FrameAllocationFailed,
    /// Page table mapping operation failed.
    MappingFailed(MapError),
    /// Kernel ELF not page-aligned in physical memory.
    NotPageAligned,
    /// No free virtual address space for the requested mapping.
    AddressSpaceFull,
    /// Fixed placement for ET_EXEC kernel is not valid.
    InvalidPlacement,
    /// Too many copied pages during relocation handling.
    TooManyCopiedPages,
}

impl core::fmt::Display for LoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidElf(msg) => write!(f, "invalid ELF: {msg}"),
            Self::UnsupportedElfType(ty) => write!(f, "unsupported ELF type: {ty}"),
            Self::UnsupportedMachine(m) => write!(f, "unsupported ELF machine type: {m}"),
            Self::UnsupportedRelocation(ty) => write!(f, "unsupported relocation type: {ty:#x}"),
            Self::FrameAllocationFailed => write!(f, "frame allocation failed"),
            Self::MappingFailed(e) => write!(f, "page table mapping failed: {e:?}"),
            Self::NotPageAligned => write!(f, "kernel ELF not page-aligned in physical memory"),
            Self::AddressSpaceFull => write!(f, "no free virtual address space"),
            Self::InvalidPlacement => write!(f, "invalid kernel placement"),
            Self::TooManyCopiedPages => write!(f, "too many CoW pages during relocations"),
        }
    }
}

/// Errors from page table map operations.
#[derive(Debug)]
pub enum MapError {
    /// The page is already mapped.
    AlreadyMapped,
    /// Frame allocation failed while creating intermediate page tables.
    FrameAllocationFailed,
}

/// Errors from page table unmap operations.
#[derive(Debug)]
pub enum UnmapError {
    /// The page is not mapped.
    NotMapped,
}

impl From<MapError> for LoadError {
    fn from(e: MapError) -> Self {
        Self::MappingFailed(e)
    }
}
