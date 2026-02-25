//! [`PhysicalMemoryAccess`] implementation for identity-mapped environments.

use crate::{PhysAddr, PhysicalMemoryAccess};

/// Identity-mapped physical memory access.
///
/// Treats physical addresses as virtual addresses directly. This is the
/// common case in bootloader environments where all physical memory is
/// identity-mapped.
pub struct IdentityMappedAccess;

impl PhysicalMemoryAccess for IdentityMappedAccess {
    unsafe fn read_phys(&self, addr: PhysAddr, buf: &mut [u8]) {
        let src = addr.as_u64() as *const u8;
        unsafe {
            core::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), buf.len());
        }
    }

    unsafe fn write_phys(&self, addr: PhysAddr, buf: &[u8]) {
        let dst = addr.as_u64() as *mut u8;
        unsafe {
            core::ptr::copy_nonoverlapping(buf.as_ptr(), dst, buf.len());
        }
    }

    unsafe fn zero_phys(&self, addr: PhysAddr, len: usize) {
        let dst = addr.as_u64() as *mut u8;
        unsafe {
            core::ptr::write_bytes(dst, 0, len);
        }
    }

    unsafe fn copy_phys(&self, src: PhysAddr, dst: PhysAddr, len: usize) {
        let src_ptr = src.as_u64() as *const u8;
        let dst_ptr = dst.as_u64() as *mut u8;
        unsafe {
            core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len);
        }
    }
}
