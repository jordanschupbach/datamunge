#pragma once
/// @file buddy_allocator.hpp
/// @brief Checked offset-based buddy allocation with actual split/coalesce paths and owned snapshots.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief One current free or allocated interval in an owned arena snapshot.
/// Offsets and sizes are bytes relative to arena zero. Free blocks have requested=0;
/// live blocks have positive requested<=size. No memory pointer or backing bytes
/// are owned by this record.
/// @par Example
/// @code
/// dm::algorithms::BuddyAllocator a(64,8);
/// (void)a.allocate(5);
/// const auto b=a.snapshot().blocks[0];
/// std::print("{} {} {}\n",b.offset,b.size,b.requested); // 0 8 5
/// @endcode
/// @see BuddySnapshot
struct BuddyBlock {
  int offset{0}; ///< Start byte offset, aligned to size.
  int order{0}; ///< Size class relative to the minimum block.
  int size{0}; ///< Reserved interval length in bytes.
  int requested{0}; ///< Live requested bytes, or zero for a free block.
  bool allocated{false}; ///< Whether the interval is a live allocation.
};

/// @brief Owned, address-sorted partition and free lists from one allocator state.
/// blocks covers [0,total_bytes) exactly once. Free blocks are maximally coalesced;
/// adjacent blocks need not be buddies. free_offsets is indexed by order and sorted
/// within each order. Copies remain valid after subsequent allocator changes.
/// Construction takes O((A+F) log(A+F+1)+H) time and O(A+F+H) storage for live
/// allocation count A, free block count F and number of orders H.
/// @par Example
/// @code
/// dm::algorithms::BuddyAllocator a(64,8);
/// (void)a.allocate(5);
/// const auto s=a.snapshot();
/// std::print("{} {} {}\n",s.free_bytes,s.largest_free_block,s.internal_fragmentation); // 56 32 3
/// @endcode
/// @see BuddyAllocator::snapshot()
struct BuddySnapshot {
  int total_bytes{0}; ///< Logical arena size; no payload storage is allocated.
  int min_block{0}; ///< Smallest block in bytes.
  int allocated_bytes{0}; ///< Sum of live reserved block sizes.
  int requested_bytes{0}; ///< Sum of live requested sizes.
  int internal_fragmentation{0}; ///< Reserved minus requested bytes.
  int free_bytes{0}; ///< Total unreserved bytes, possibly in separated intervals.
  int largest_free_block{0}; ///< Largest single allocatable free block, or zero if full.
  std::vector<BuddyBlock> blocks; ///< Complete free/live partition, sorted by offset.
  std::vector<std::vector<int>> free_offsets; ///< Actual free offsets for every order.
};

/// @brief One actual allocation split of a parent into equal-sized children.
/// Parent interval starts at offset and has order parent_order. The retained left
/// child starts at offset; the free right buddy starts at offset+half_size.
/// @par Example
/// @code
/// dm::algorithms::BuddyAllocator a(64,8);
/// const auto t=a.allocate_trace(5);
/// std::print("{} {} {}\n",t.splits[0].offset,t.splits[0].parent_order,t.splits[0].half_size); // 0 3 32
/// @endcode
/// @see BuddyAllocationTrace
struct BuddySplit {
  int offset{0}; ///< Parent and retained-left-child start offset.
  int parent_order{0}; ///< Parent order before splitting.
  int half_size{0}; ///< Each child's byte size.
};

/// @brief Actual order search, splits and returned offset from one allocation call.
/// offset=-1 means failure with unchanged state; a positive oversized request has
/// no eligible order and examines zero free lists. Other failures examine every
/// order from desired_order through the arena's maximum. Successful calls record
/// only actual splits, in decreasing parent order. O(H) trace storage, no snapshots.
/// @par Example
/// @code
/// dm::algorithms::BuddyAllocator a(64,8);
/// const auto t=a.allocate_trace(5);
/// std::print("{} {} {}\n",t.offset,t.orders_examined,t.splits.size()); // 0 4 3
/// @endcode
/// @see BuddyAllocator::allocate_trace()
struct BuddyAllocationTrace {
  int requested{0}; ///< Positive requested byte count.
  int desired_order{0}; ///< Smallest mathematical class fitting the request, even if beyond the arena.
  int source_order{-1}; ///< Found free-block order, or -1 on failure.
  int offset{-1}; ///< Allocated offset, or -1 on failure.
  std::size_t orders_examined{0}; ///< Actual free-list emptiness checks, including the found list.
  std::vector<BuddySplit> splits; ///< Actual split path; empty if no split was needed.
};

/// @brief One actual same-order buddy membership test during a free operation.
/// buddy_offset=offset XOR block_size(order), using arena-relative offsets. A
/// successful test triggers one merge; a false test stops the cascade. No test
/// occurs once the arena's maximum order has been reached.
/// @par Example
/// @code
/// dm::algorithms::BuddyAllocator a(64,8);
/// const int off=a.allocate(5);
/// const auto t=a.free_trace(off);
/// std::print("{} {}\n",t.probes[0].buddy_offset,t.probes[0].buddy_free); // 8 true
/// @endcode
/// @see BuddyFreeTrace
struct BuddyProbe {
  int offset{0}; ///< Current block offset before the test.
  int order{0}; ///< Current block order.
  int buddy_offset{0}; ///< Computed same-order buddy offset.
  bool buddy_free{false}; ///< Whether the actual free set contained that buddy.
};

/// @brief Actual release, buddy probes and resulting coalesced block from one free call.
/// Unknown, interior, negative and already-free offsets are no-ops: released=false,
/// final_offset=final_order=-1, no probes. A valid offset releases the current live
/// allocation at that offset; stale offsets cannot distinguish later offset reuse.
/// O(H) trace storage, with no arena snapshots or operation replay.
/// @par Example
/// @code
/// dm::algorithms::BuddyAllocator a(64,8);
/// const auto t=a.free_trace(a.allocate(5));
/// std::print("{} {} {}\n",t.released,t.merges,t.final_order); // true 3 3
/// @endcode
/// @see BuddyAllocator::free_trace()
struct BuddyFreeTrace {
  int requested_offset{-1}; ///< Offset supplied to free_trace().
  bool released{false}; ///< Whether a current live allocation was released.
  int final_offset{-1}; ///< Resulting free-block start, or -1 for a no-op.
  int final_order{-1}; ///< Resulting free-block order, or -1 for a no-op.
  std::size_t merges{0}; ///< Number of successful same-order buddy tests and merges.
  std::vector<BuddyProbe> probes; ///< Actual probes in ascending order, including a final false test if any.
};

/// @brief Model a binary buddy arena using offsets, ordered free sets and live maps.
/// Does not allocate arena payload bytes or return pointers. Both dimensions must
/// be positive powers of two, with minimum<=total. Allocation chooses the smallest
/// adequate nonempty order, then its lowest offset, retaining the left child when
/// splitting. Free coalesces only true same-order buddies. Unknown frees are no-ops.
/// Positive oversize/exhausted requests return -1; nonpositive requests throw.
/// Ordinary and traced calls share kernels. Metadata allocation failure leaves
/// state unchanged; snapshots own their data. Copies are independent offset models.
/// Concurrent mutations need external synchronization. For H orders and B current
/// blocks, operations have conservative O(H log(B+1)) ordered-container cost, not
/// a constant-cost-per-order guarantee. Metadata storage is O(H+B).
/// @par Example
/// @code
/// dm::algorithms::BuddyAllocator a(1024,32);
/// const int off=a.allocate(30);
/// std::print("{} {}\n",off,a.internal_fragmentation()); // 0 2
/// @endcode
/// @par References
/// - K. C. Knowlton (1965), A fast storage allocator, Communications of the ACM 8(10),
///   doi:10.1145/365628.365655; author bibliography: https://www.kenknowlton.com/pages/29papers.htm.
/// - C++ associative-container requirements, https://eel.is/c++draft/associative.reqmts.
/// @see BuddySnapshot, allocate_trace(), free_trace()
class BuddyAllocator {
 public:
  /// @brief Construct a fully free logical arena after validating both dimensions.
  /// Allocates only O(H) host metadata for H orders, not total_size payload bytes.
  /// @param total_size Positive power-of-two arena size in bytes, fitting int.
  /// @param min_block Positive power of two no larger than total_size, in bytes.
  /// @throws std::invalid_argument If either dimension is invalid.
  /// @throws std::bad_alloc If metadata cannot be allocated.
  /// @par Example
  /// @code
  /// dm::algorithms::BuddyAllocator a(1024,32);
  /// std::print("{} {}\n",a.max_order(),a.free_blocks_per_order()); // 5 [0, 0, 0, 0, 0, 1]
  /// @endcode
  /// @par References
  /// - K. C. Knowlton (1965), A fast storage allocator, doi:10.1145/365628.365655.
  /// @see BuddyAllocator
  BuddyAllocator(int total_size,int min_block):total_(total_size),min_block_(min_block),max_order_(0) {
    if (!power_of_two(total_size) || !power_of_two(min_block) || min_block>total_size)
      throw std::invalid_argument("buddy dimensions must be positive powers of two with minimum<=total");
    for (int ratio=total_/min_block_;ratio>1;ratio/=2) ++max_order_;
    free_lists_.resize(static_cast<std::size_t>(max_order_)+1);
    free_lists_[max_order_].insert(0);
  }

  /// @brief Find the smallest power-of-two class fitting a positive request safely.
  /// The order may exceed max_order() for an oversized request; block_size() only
  /// accepts arena orders. Uses widened multiplication, never an overflowing shift.
  /// @param size Positive requested byte count.
  /// @return Least order with min_block*2^order>=size, possibly beyond the arena.
  /// @throws std::invalid_argument If size is not positive.
  /// @par Example
  /// @code
  /// dm::algorithms::BuddyAllocator a(1024,32);
  /// std::print("{} {}\n",a.order_for_size(100),a.order_for_size(1025)); // 2 6
  /// @endcode
  /// @par References
  /// - K. C. Knowlton (1965), A fast storage allocator, doi:10.1145/365628.365655.
  /// @see block_size(), allocate()
  int order_for_size(int size) const {
    if (size<=0) throw std::invalid_argument("buddy request must be positive");
    std::uint64_t block=static_cast<unsigned int>(min_block_);
    int order=0;
    while (block<static_cast<unsigned int>(size)) {block*=2;++order;}
    return order;
  }

  /// @brief Allocate the lowest offset at the smallest adequate available order.
  /// Returns -1 with unchanged state if no block fits, including positive requests
  /// larger than the arena. Zero and negative sizes throw before mutation. Metadata
  /// failures leave state unchanged. No backing payload bytes are allocated.
  /// @param size Positive requested bytes.
  /// @return Live arena-relative offset, or -1 on capacity/fragmentation failure.
  /// @throws std::invalid_argument If size is not positive.
  /// @throws std::bad_alloc If host metadata cannot be allocated.
  /// @par Example
  /// @code
  /// dm::algorithms::BuddyAllocator a(64,8);
  /// std::print("{}\n",a.allocate(9)); // 0
  /// @endcode
  /// @par References
  /// - K. C. Knowlton (1965), A fast storage allocator, doi:10.1145/365628.365655.
  /// @see allocate_trace(), free()
  int allocate(int size) {return allocate_impl(size,nullptr);}

  /// @brief Allocate once and return actual order checks and split decisions.
  /// Mutates exactly as allocate(); the kernel is not replayed. Trace capacity is
  /// reserved before mutation, and metadata failures leave allocator state unchanged.
  /// @param size Positive requested bytes, possibly exceeding arena capacity.
  /// @return Owned actual path, with offset=-1 on capacity/fragmentation failure.
  /// @throws std::invalid_argument If size is not positive.
  /// @throws std::bad_alloc If trace or host metadata cannot be allocated.
  /// @par Example
  /// @code
  /// dm::algorithms::BuddyAllocator a(64,8);
  /// const auto t=a.allocate_trace(9);
  /// std::print("{} {} {}\n",t.desired_order,t.source_order,t.splits.size()); // 1 3 2
  /// @endcode
  /// @par References
  /// - K. C. Knowlton (1965), A fast storage allocator, doi:10.1145/365628.365655.
  /// @see allocate(), BuddyAllocationTrace
  BuddyAllocationTrace allocate_trace(int size) {
    BuddyAllocationTrace trace;trace.requested=size;
    (void)allocate_impl(size,&trace);
    return trace;
  }

  /// @brief Free a current live offset and maximally coalesce its same-order buddies.
  /// Unknown/interior/negative/already-free offsets are no-ops, preserving historical
  /// behavior. A reused live offset names its current allocation. Metadata failure
  /// leaves state unchanged. Does not access or release any backing payload bytes.
  /// @param offset Current allocation start to release, or an ignored unknown offset.
  /// @throws std::bad_alloc If the final free-block metadata cannot be allocated.
  /// @par Example
  /// @code
  /// dm::algorithms::BuddyAllocator a(64,8);
  /// a.free(a.allocate(5));a.free(-1);
  /// std::print("{}\n",a.free_blocks_per_order()); // [0, 0, 0, 1]
  /// @endcode
  /// @par References
  /// - K. C. Knowlton (1965), A fast storage allocator, doi:10.1145/365628.365655.
  /// @see free_trace(), allocate()
  void free(int offset) {free_impl(offset,nullptr);}

  /// @brief Free once and record actual XOR buddy tests and successful merges.
  /// Shares the free() kernel without replay or whole-arena snapshots. Unknown
  /// offsets return released=false and no probes. Metadata failure leaves state
  /// unchanged, including failures while reserving the trace.
  /// @param offset Current allocation start, or an ignored unknown offset.
  /// @return Owned actual probe path and resulting free block, or an explicit no-op.
  /// @throws std::bad_alloc If trace or host metadata cannot be allocated.
  /// @par Example
  /// @code
  /// dm::algorithms::BuddyAllocator a(64,8);
  /// const auto t=a.free_trace(a.allocate(5));
  /// std::print("{} {}\n",t.final_offset,t.merges); // 0 3
  /// @endcode
  /// @par References
  /// - K. C. Knowlton (1965), A fast storage allocator, doi:10.1145/365628.365655.
  /// @see free(), BuddyFreeTrace
  BuddyFreeTrace free_trace(int offset) {
    BuddyFreeTrace trace;trace.requested_offset=offset;
    free_impl(offset,&trace);
    return trace;
  }

  /// @brief Sum live reserved block sizes in bytes, including rounding waste.
  /// O(A) time for A live allocations, no allocation or mutation.
  /// @return Value in [0,total_size].
  int allocated_bytes() const {
    int total=0;for (const auto& [offset,order]:alloc_order_) {(void)offset;total+=block_size(order);}return total;
  }

  /// @brief Sum requested bytes in live allocations, excluding rounding waste.
  /// O(A) time, no allocation or mutation; requests are always positive.
  /// @return Value in [0,allocated_bytes()].
  int requested_bytes() const {
    int total=0;for (const auto& [offset,requested]:alloc_req_) {(void)offset;total+=requested;}return total;
  }

  /// @brief Compute live internal waste as reserved minus requested bytes.
  /// O(A) time, no allocation or mutation. Requests below min_block can waste more
  /// than half a block; larger-than-minimum requests waste strictly less than half.
  /// @return Nonnegative unused bytes inside live blocks.
  int internal_fragmentation() const {return allocated_bytes()-requested_bytes();}

  /// @brief Copy the actual free-block count at each order.
  /// O(H) time/storage; index zero is min_block and the last index spans the arena.
  /// @return H=max_order()+1 owned integer counts.
  /// @throws std::bad_alloc If the result vector cannot be allocated.
  /// @par Example
  /// @code
  /// dm::algorithms::BuddyAllocator a(64,8);(void)a.allocate(5);
  /// std::print("{}\n",a.free_blocks_per_order()); // [1, 1, 1, 0]
  /// @endcode
  /// @see snapshot()
  std::vector<int> free_blocks_per_order() const {
    std::vector<int> result(free_lists_.size());
    for (std::size_t order=0;order<free_lists_.size();++order) result[order]=static_cast<int>(free_lists_[order].size());
    return result;
  }

  /// @brief Return the highest valid arena order without mutation or allocation.
  /// @return log2(total_size/min_block), at least zero.
  int max_order() const {return max_order_;}

  /// @brief Return a checked arena block size without overflowing a signed shift.
  /// @param order Arena order in [0,max_order()].
  /// @return min_block*2^order bytes, at most total_size.
  /// @throws std::out_of_range If order is outside the arena's order range.
  /// @par Example
  /// @code
  /// dm::algorithms::BuddyAllocator a(1024,32);
  /// std::print("{}\n",a.block_size(2)); // 128
  /// @endcode
  /// @see order_for_size()
  int block_size(int order) const {
    if (order<0 || order>max_order_) throw std::out_of_range("buddy block order outside arena");
    return min_block_<<order;
  }

  /// @brief Copy actual free/live metadata into an address-sorted arena partition.
  /// No simulation, payload inspection or mutation; returned values own all storage.
  /// See BuddySnapshot for fields, byte units, complexity and interval invariants.
  /// @return Complete independent snapshot of the current state.
  /// @throws std::bad_alloc If snapshot storage cannot be allocated.
  /// @par Example
  /// @code
  /// dm::algorithms::BuddyAllocator a(64,8);(void)a.allocate(5);
  /// const auto s=a.snapshot();
  /// std::print("{} {}\n",s.blocks.size(),s.free_offsets[2]); // 4 [32]
  /// @endcode
  /// @see BuddySnapshot, free_blocks_per_order()
  BuddySnapshot snapshot() const {
    BuddySnapshot result;
    result.total_bytes=total_;result.min_block=min_block_;
    result.allocated_bytes=allocated_bytes();result.requested_bytes=requested_bytes();
    result.internal_fragmentation=result.allocated_bytes-result.requested_bytes;
    result.free_bytes=total_-result.allocated_bytes;
    result.free_offsets.resize(free_lists_.size());
    for (int order=0;order<=max_order_;++order) {
      const int size=block_size(order);
      result.free_offsets[order].assign(free_lists_[order].begin(),free_lists_[order].end());
      for (const int offset:free_lists_[order]) result.blocks.push_back({offset,order,size,0,false});
      if (!free_lists_[order].empty()) result.largest_free_block=size;
    }
    for (const auto& [offset,order]:alloc_order_)
      result.blocks.push_back({offset,order,block_size(order),alloc_req_.at(offset),true});
    /// @brief Order free/live interval records by their unique start offsets.
    /// @param left First record.
    /// @param right Second record.
    /// @return Whether left starts earlier.
    std::sort(result.blocks.begin(),result.blocks.end(),[](const BuddyBlock& left,const BuddyBlock& right){return left.offset<right.offset;});
    return result;
  }

 private:
  /// @brief Test a positive integer's single-bit power-of-two property safely.
  /// @param value Dimension to validate.
  /// @return Whether value is a positive power of two.
  static bool power_of_two(int value) {return value>0 && (value&(value-1))==0;}

  /// @brief Shared smallest-order/lowest-offset allocation with rollback on metadata failure.
  /// All lower candidate free sets are empty when source_order is found, so every
  /// split retains its left child. Prepare right-buddy and live nodes before removing
  /// the source; rollback erases only nodes that did not exist before this call.
  /// @param size Positive request, validated before any mutation.
  /// @param trace Optional owned-path receiver with capacity reserved before mutation.
  /// @return Allocated offset, or -1 if no block fits.
  /// @throws std::invalid_argument If size is not positive.
  /// @throws std::bad_alloc If trace or metadata allocation fails; state is unchanged.
  int allocate_impl(int size,BuddyAllocationTrace* trace) {
    const int desired=order_for_size(size);
    if (trace) trace->desired_order=desired;
    int source=desired;
    while (source<=max_order_) {
      if (trace) ++trace->orders_examined;
      if (!free_lists_[source].empty()) break;
      ++source;
    }
    if (source>max_order_) return -1;
    if (trace) trace->splits.reserve(static_cast<std::size_t>(source-desired));
    const auto original=free_lists_[source].begin();
    const int offset=*original;
    try {
      for (int order=source;order>desired;--order) {
        const int half=block_size(order-1);
        free_lists_[order-1].insert(offset+half);
        if (trace) trace->splits.push_back({offset,order,half});
      }
      alloc_order_.emplace(offset,desired);
      alloc_req_.emplace(offset,size);
    } catch (...) {
      for (int order=source;order>desired;--order) free_lists_[order-1].erase(offset+block_size(order-1));
      alloc_order_.erase(offset);alloc_req_.erase(offset);
      throw;
    }
    free_lists_[source].erase(original);
    if (trace) {trace->source_order=source;trace->offset=offset;}
    return offset;
  }

  /// @brief Shared live lookup and actual buddy-probe path with failure-safe commit.
  /// Probe without deleting children, then insert the final parent before committing
  /// nonthrowing erases. A fixed stack array holds at most one buddy iterator per
  /// representable order. No trace is replayed to discover or count operations.
  /// @param offset Offset supplied to free/free_trace; unknown values are ignored.
  /// @param trace Optional actual-path receiver, reserved before mutation.
  /// @throws std::bad_alloc If trace/final-parent allocation fails; state is unchanged.
  void free_impl(int offset,BuddyFreeTrace* trace) {
    const auto allocation=alloc_order_.find(offset);
    if (allocation==alloc_order_.end()) return;
    if (trace) trace->probes.reserve(static_cast<std::size_t>(max_order_));
    std::array<std::pair<int,std::set<int>::iterator>,std::numeric_limits<int>::digits> path;
    std::size_t merges=0;
    int start=offset,order=allocation->second;
    while (order<max_order_) {
      const int buddy=start^block_size(order);
      const auto found=free_lists_[order].find(buddy);
      const bool available=found!=free_lists_[order].end();
      if (trace) trace->probes.push_back({start,order,buddy,available});
      if (!available) break;
      path[merges++]={order,found};start=std::min(start,buddy);++order;
    }
    free_lists_[order].insert(start);
    for (std::size_t i=0;i<merges;++i) free_lists_[path[i].first].erase(path[i].second);
    alloc_req_.erase(offset);alloc_order_.erase(allocation);
    if (trace) {trace->released=true;trace->final_offset=start;trace->final_order=order;trace->merges=merges;}
  }

  int total_; ///< Positive logical arena byte count, a power of two.
  int min_block_; ///< Positive minimum block byte count, a power of two.
  int max_order_; ///< Highest valid order, with min_block*2^max_order=total.
  std::vector<std::set<int>> free_lists_; ///< Actual ordered free offsets, indexed by order.
  std::map<int,int> alloc_order_; ///< Current live offset to block order.
  std::map<int,int> alloc_req_; ///< Current live offset to positive requested byte count.
};

} // namespace datamunge::algorithms
