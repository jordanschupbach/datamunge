/// @file buddy_allocator_allocations.cpp
/// @brief Isolated host-allocation fault injection for buddy metadata transactions.
#include <datamunge/algorithms/buddy_allocator.hpp>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace {
bool failing=false; ///< Whether to inject a failure in the measured call.
std::size_t remaining=0; ///< Successful allocations allowed before the injected failure.
}
/// @brief Fail one selected ordinary host allocation during a measured operation.
/// @param size Requested host allocation length.
/// @return Owned memory allocated by malloc.
/// @throws std::bad_alloc At the injected position or on actual host allocation failure.
void* operator new(std::size_t size) {
  if (failing) {
    if (!remaining) throw std::bad_alloc();
    --remaining;
  }
  if (void* memory=std::malloc(size?size:1)) return memory;
  throw std::bad_alloc();
}
/// @brief Release intercepted ordinary memory.
/// @param memory Owned host allocation.
void operator delete(void* memory) noexcept {std::free(memory);}
/// @brief Release intercepted sized memory.
/// @param memory Owned host allocation.
/// @param size Unused host length.
void operator delete(void* memory,std::size_t size) noexcept {(void)size;std::free(memory);}

namespace {
/// @brief Compare complete arena partitions after an injected failure.
/// @param a First owned snapshot.
/// @param b Second owned snapshot.
/// @return Whether metadata and accounting match exactly.
bool same(const datamunge::algorithms::BuddySnapshot& a,const datamunge::algorithms::BuddySnapshot& b) {
  if (a.total_bytes!=b.total_bytes || a.min_block!=b.min_block || a.allocated_bytes!=b.allocated_bytes ||
      a.requested_bytes!=b.requested_bytes || a.free_bytes!=b.free_bytes ||
      a.internal_fragmentation!=b.internal_fragmentation || a.largest_free_block!=b.largest_free_block ||
      a.free_offsets!=b.free_offsets || a.blocks.size()!=b.blocks.size()) return false;
  for (std::size_t i=0;i<a.blocks.size();++i) {
    const auto& x=a.blocks[i];const auto& y=b.blocks[i];
    if (x.offset!=y.offset || x.order!=y.order || x.size!=y.size || x.requested!=y.requested || x.allocated!=y.allocated) return false;
  }
  return true;
}
/// @brief Fail each host allocation in a valid allocation or free until one full call succeeds.
/// @param mode Zero/one choose plain/traced allocate; two/three plain/traced coalescing free;
///             four/five plain/traced free with a busy buddy; six/seven allocate without a split.
/// @return Whether all failures preserved state and the successful call was valid.
bool scenario(int mode) {
  std::size_t failures=0;
  for (std::size_t budget=0;budget<100;++budget) {
    datamunge::algorithms::BuddyAllocator arena(1024,32);
    int off=-1;
    if (mode>=2 && mode<=5) off=arena.allocate(30);
    if (mode>=4 && mode<=5) (void)arena.allocate(30);
    if (mode>=6) (void)arena.allocate(30); // its same-order right buddy is already free
    const auto before=arena.snapshot();
    bool threw=false;
    remaining=budget;failing=true;
    try {
      if (mode==0 || mode==6) off=arena.allocate(30);
      else if (mode==1 || mode==7) off=arena.allocate_trace(30).offset;
      else if (mode==2 || mode==4) arena.free(off);
      else (void)arena.free_trace(off);
    } catch (const std::bad_alloc&) {threw=true;}
    failing=false;
    if (threw) {
      ++failures;
      if (!same(before,arena.snapshot())) return false;
      // The object remains usable after rollback, not merely inspectable.
      const int recovery=arena.allocate(1);
      if (recovery<0) return false;
      arena.free(recovery);
      if (!same(before,arena.snapshot())) return false;
    } else {
      if (!failures) return false;
      const int expected=(mode<=1)?32:(mode<=3)?0:(mode<=5)?32:64;
      if (arena.allocated_bytes()!=expected) return false;
      std::printf("Buddy metadata scenario %d: %zu injected failure positions, then success\n",mode,failures);
      return true;
    }
  }
  return false;
}
}
/// @brief Verify strong state preservation for every host-allocation failure position.
/// @return Zero on success, otherwise one.
int main() {
  for (int mode=0;mode<8;++mode) if (!scenario(mode)) return 1;
}
