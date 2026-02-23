//! Virtual address space tracking for the kernel.
//!
//! Maintains a sorted list of used virtual address regions and provides
//! methods to find free regions with optional ASLR randomization.

use crate::traits::VirtAddr;
use rand_core::RngCore;

const MAX_REGIONS: usize = 128;

/// Tracks used virtual address regions to find free space for new mappings.
///
/// Uses a simple sorted flat array. `find_free` scans gaps between used
/// regions for candidates that fit the requested size + alignment.
pub struct AddressSpace {
    /// Sorted array of (start, end_exclusive) used regions.
    regions: [(u64, u64); MAX_REGIONS],
    count: usize,
}

impl AddressSpace {
    pub fn new() -> Self {
        Self {
            regions: [(0, 0); MAX_REGIONS],
            count: 0,
        }
    }

    /// Mark a virtual address range as used.
    ///
    /// The range `[start, start + size)` will be considered occupied.
    pub fn mark_used(&mut self, start: VirtAddr, size: u64) {
        if size == 0 {
            return;
        }
        let start = start.as_u64();
        let end = start.saturating_add(size);

        // Find insertion point (maintain sorted order by start address).
        let pos = self.regions[..self.count]
            .iter()
            .position(|&(s, _)| s > start)
            .unwrap_or(self.count);

        assert!(self.count < MAX_REGIONS, "address space region limit reached");

        // Shift elements right.
        self.regions.copy_within(pos..self.count, pos + 1);
        self.regions[pos] = (start, end);
        self.count += 1;

        // Merge overlapping/adjacent regions.
        self.merge();
    }

    /// Merge overlapping or adjacent regions.
    fn merge(&mut self) {
        if self.count <= 1 {
            return;
        }
        let mut write = 0;
        for read in 1..self.count {
            let next_start = self.regions[read].0;
            let next_end = self.regions[read].1;
            if next_start <= self.regions[write].1 {
                // Overlapping or adjacent — extend.
                if next_end > self.regions[write].1 {
                    self.regions[write].1 = next_end;
                }
            } else {
                write += 1;
                self.regions[write] = (next_start, next_end);
            }
        }
        self.count = write + 1;
    }

    /// Find a free region of `size` bytes with the given alignment.
    ///
    /// If `rng` is `Some`, the region is chosen randomly among all
    /// candidates (ASLR). If `None`, the first suitable gap is used.
    ///
    /// Returns the start address of the free region and marks it as used.
    pub fn find_free(
        &mut self,
        size: u64,
        alignment: u64,
        rng: Option<&mut dyn RngCore>,
    ) -> Option<VirtAddr> {
        assert!(alignment.is_power_of_two());
        assert!(size > 0);

        let candidates = self.collect_candidates(size, alignment);
        if candidates.is_empty() {
            return None;
        }

        let addr = if let Some(rng) = rng {
            self.pick_random_candidate(&candidates, size, alignment, rng)
        } else {
            // First fit: use the first candidate at offset 0.
            let (gap_start, _) = candidates.as_slice()[0];
            align_up(gap_start, alignment)
        };

        self.mark_used(VirtAddr::new(addr), size);
        Some(VirtAddr::new(addr))
    }

    /// Collect all gaps that can fit the requested size + alignment.
    ///
    /// Returns a vector of (gap_start, gap_end_exclusive) pairs.
    fn collect_candidates(&self, size: u64, alignment: u64) -> CandidateList {
        let mut candidates = CandidateList::new();

        // Check gap before first region (but skip address 0 — never allocate at NULL).
        let first_start = if self.count > 0 {
            self.regions[0].0
        } else {
            u64::MAX
        };
        if first_start > 0 {
            let gap_start = alignment; // Skip address 0
            let gap_end = first_start;
            if gap_end > gap_start {
                let aligned_start = align_up(gap_start, alignment);
                if aligned_start + size <= gap_end {
                    candidates.push((gap_start, gap_end));
                }
            }
        }

        // Check gaps between regions.
        for i in 0..self.count.saturating_sub(1) {
            let gap_start = self.regions[i].1;
            let gap_end = self.regions[i + 1].0;
            if gap_end > gap_start {
                let aligned_start = align_up(gap_start, alignment);
                if aligned_start + size <= gap_end {
                    candidates.push((gap_start, gap_end));
                }
            }
        }

        // Check gap after last region.
        if self.count > 0 {
            let gap_start = self.regions[self.count - 1].1;
            // Use a large sentinel as "end of address space"
            // (we don't want to go all the way to u64::MAX to avoid overflow).
            let gap_end = 0xFFFF_FFFF_FFFF_0000u64;
            let aligned_start = align_up(gap_start, alignment);
            if gap_end > aligned_start && aligned_start + size <= gap_end {
                candidates.push((gap_start, gap_end));
            }
        } else {
            // No used regions at all — the whole space (minus 0) is free.
            let gap_start = alignment;
            let gap_end = 0xFFFF_FFFF_FFFF_0000u64;
            candidates.push((gap_start, gap_end));
        }

        candidates
    }

    fn pick_random_candidate(
        &self,
        candidates: &CandidateList,
        size: u64,
        alignment: u64,
        rng: &mut dyn RngCore,
    ) -> u64 {
        // Count total number of aligned positions across all gaps.
        let mut total_positions: u64 = 0;
        for &(gap_start, gap_end) in candidates.as_slice() {
            let aligned_start = align_up(gap_start, alignment);
            let max_start = gap_end - size;
            if max_start >= aligned_start {
                let positions = (max_start - aligned_start) / alignment + 1;
                total_positions = total_positions.saturating_add(positions);
            }
        }

        // Pick a random position index.
        let chosen = if total_positions > 0 {
            rng.next_u64() % total_positions
        } else {
            0
        };

        // Map the chosen index back to an address.
        let mut remaining = chosen;
        for &(gap_start, gap_end) in candidates.as_slice() {
            let aligned_start = align_up(gap_start, alignment);
            let max_start = gap_end - size;
            if max_start >= aligned_start {
                let positions = (max_start - aligned_start) / alignment + 1;
                if remaining < positions {
                    return aligned_start + remaining * alignment;
                }
                remaining -= positions;
            }
        }

        // Fallback (shouldn't happen).
        let (gap_start, _) = candidates.as_slice()[0];
        align_up(gap_start, alignment)
    }

    /// Returns the number of tracked regions.
    pub fn region_count(&self) -> usize {
        self.count
    }
}

/// Small fixed-capacity list for candidate gaps (avoids heap allocation).
struct CandidateList {
    entries: [(u64, u64); MAX_REGIONS + 1],
    count: usize,
}

impl CandidateList {
    fn new() -> Self {
        Self {
            entries: [(0, 0); MAX_REGIONS + 1],
            count: 0,
        }
    }

    fn push(&mut self, entry: (u64, u64)) {
        if self.count < self.entries.len() {
            self.entries[self.count] = entry;
            self.count += 1;
        }
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn as_slice(&self) -> &[(u64, u64)] {
        &self.entries[..self.count]
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) / alignment * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mark_used_and_find_free() {
        let mut space = AddressSpace::new();
        space.mark_used(VirtAddr::new(0x1000), 0x1000);
        space.mark_used(VirtAddr::new(0x3000), 0x1000);

        // Should find the gap at 0x2000.
        let addr = space.find_free(0x1000, 0x1000, None).unwrap();
        assert_eq!(addr.as_u64(), 0x2000);
    }

    #[test]
    fn test_merge_adjacent() {
        let mut space = AddressSpace::new();
        space.mark_used(VirtAddr::new(0x1000), 0x1000);
        space.mark_used(VirtAddr::new(0x2000), 0x1000);
        assert_eq!(space.region_count(), 1);
    }

    #[test]
    fn test_merge_overlapping() {
        let mut space = AddressSpace::new();
        space.mark_used(VirtAddr::new(0x1000), 0x2000);
        space.mark_used(VirtAddr::new(0x2000), 0x2000);
        assert_eq!(space.region_count(), 1);
    }

    #[test]
    fn test_alignment() {
        let mut space = AddressSpace::new();
        space.mark_used(VirtAddr::new(0x0), 0x1001); // Ends at 0x1001

        // With 0x1000 alignment, should align up to 0x2000.
        let addr = space.find_free(0x1000, 0x1000, None).unwrap();
        assert_eq!(addr.as_u64(), 0x2000);
    }

    #[test]
    fn test_no_null_allocation() {
        let mut space = AddressSpace::new();
        // The whole space is free, but we should not get address 0.
        let addr = space.find_free(0x1000, 0x1000, None).unwrap();
        assert!(addr.as_u64() > 0);
    }
}
