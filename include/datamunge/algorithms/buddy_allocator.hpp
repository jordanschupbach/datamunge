#pragma once

/// \file buddy_allocator.hpp
/// \brief Buddy memory allocation: power-of-two blocks with fast split and coalesce.
///
/// The buddy system manages a power-of-two arena as blocks whose sizes are also powers of two.
/// A request is rounded up to the nearest block size; if no free block of that size exists, a
/// larger one is split in half repeatedly, each half the *buddy* of the other. On free, a block
/// is merged with its buddy whenever the buddy is also free, recursively -- and because a
/// block's buddy address is just its own address with one bit flipped, both operations are
/// \f$O(\log n)\f$. The trade-off is *internal fragmentation*: rounding up to a power of two
/// wastes up to nearly half of each allocation. This module implements allocation, freeing,
/// and fragmentation accounting.

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <vector>

namespace datamunge::algorithms {

/// A buddy-system allocator over an arena of \c total_size bytes (a power of two).
class BuddyAllocator {
  public:
    /// \param total_size  arena size (power of two).
    /// \param min_block   smallest allocatable block (power of two).
    BuddyAllocator(int total_size, int min_block)
        : total_(total_size), min_block_(min_block) {
        max_order_ = 0;
        while ((min_block_ << max_order_) < total_) ++max_order_;
        free_lists_.assign(max_order_ + 1, {});
        free_lists_[max_order_].insert(0);            // one free block spanning the arena
    }

    /// \brief Smallest order whose block fits \c size (block size = min_block * 2^order).
    int order_for_size(int size) const {
        int ord = 0;
        while ((min_block_ << ord) < size) ++ord;
        return ord;
    }

    /// \brief Allocate a block for \c size bytes; returns its offset, or -1 if none fits.
    int allocate(int size) {
        int ord = order_for_size(size);
        int o   = ord;
        while (o <= max_order_ && free_lists_[o].empty()) ++o;
        if (o > max_order_) return -1;                // no block large enough
        while (o > ord) {                             // split down to the requested order
            int off = *free_lists_[o].begin();
            free_lists_[o].erase(free_lists_[o].begin());
            int half = min_block_ << (o - 1);
            free_lists_[o - 1].insert(off);
            free_lists_[o - 1].insert(off + half);    // the two buddies
            --o;
        }
        int off = *free_lists_[ord].begin();
        free_lists_[ord].erase(free_lists_[ord].begin());
        alloc_order_[off] = ord;
        alloc_req_[off]   = size;
        return off;
    }

    /// \brief Free the block at \c offset (must be a live allocation), coalescing buddies.
    void free(int offset) {
        auto it = alloc_order_.find(offset);
        if (it == alloc_order_.end()) return;
        int ord = it->second;
        alloc_order_.erase(it);
        alloc_req_.erase(offset);
        while (ord < max_order_) {                    // coalesce with a free buddy
            int blocksz = min_block_ << ord;
            int buddy   = offset ^ blocksz;           // buddy address = flip the size bit
            auto bit    = free_lists_[ord].find(buddy);
            if (bit == free_lists_[ord].end()) break;  // buddy busy -> stop merging
            free_lists_[ord].erase(bit);
            offset = std::min(offset, buddy);
            ++ord;
        }
        free_lists_[ord].insert(offset);
    }

    /// \brief Total bytes currently handed out (block sizes, including internal fragmentation).
    int allocated_bytes() const {
        int t = 0;
        for (const auto& [off, ord] : alloc_order_) t += min_block_ << ord;
        return t;
    }

    /// \brief Bytes actually requested by live allocations (excludes rounding waste).
    int requested_bytes() const {
        int t = 0;
        for (const auto& [off, req] : alloc_req_) t += req;
        return t;
    }

    /// \brief Internal fragmentation = allocated block bytes minus requested bytes.
    int internal_fragmentation() const { return allocated_bytes() - requested_bytes(); }

    /// \brief Number of free blocks at each order (index = order).
    std::vector<int> free_blocks_per_order() const {
        std::vector<int> counts(max_order_ + 1);
        for (int o = 0; o <= max_order_; ++o) counts[o] = static_cast<int>(free_lists_[o].size());
        return counts;
    }

    int max_order() const { return max_order_; }
    int block_size(int order) const { return min_block_ << order; }

  private:
    int                        total_, min_block_, max_order_;
    std::vector<std::set<int>> free_lists_;   ///< free_lists_[order] = free block offsets.
    std::map<int, int>         alloc_order_;  ///< live offset -> order.
    std::map<int, int>         alloc_req_;    ///< live offset -> requested size.
};

}  // namespace datamunge::algorithms
