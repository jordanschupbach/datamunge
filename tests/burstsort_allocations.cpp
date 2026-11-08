/// @file burstsort_allocations.cpp
/// @brief Isolated host-allocation failure checks for transactional burstsort output.
#include <datamunge/algorithms/burstsort.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace {
bool failing=false; ///< Whether the measured operation should inject a host-allocation failure.
std::size_t remaining=0; ///< Successful host allocations before the chosen failure position.
}
/// @brief Inject one failure at a selected ordinary allocation within the measured sort.
/// @param size Requested host allocation length.
/// @return Owned malloc allocation.
/// @throws std::bad_alloc At the selected failure or on actual host allocation failure.
void* operator new(std::size_t size) {
  if (failing) {if (!remaining) throw std::bad_alloc();--remaining;}
  if (void* memory=std::malloc(size?size:1)) return memory;
  throw std::bad_alloc();
}
/// @brief Free intercepted ordinary memory.
/// @param memory Owned host allocation.
void operator delete(void* memory) noexcept {std::free(memory);}
/// @brief Free intercepted sized memory.
/// @param memory Owned host allocation.
/// @param size Unused allocation length.
void operator delete(void* memory,std::size_t size) noexcept {(void)size;std::free(memory);}

/// @brief Fail every allocation position until each ordinary/configured sort succeeds.
/// @return Zero only if every exception preserves values and a subsequent sort succeeds.
int main() {
  const std::vector<std::string> saved{"common9","common1","common4","common3","common7","common2","common1","common5","common6","common0",""};
  auto expected=saved;std::sort(expected.begin(),expected.end());
  for (int mode=0;mode<2;++mode) {
    bool complete=false;std::size_t failures=0;
    for (std::size_t budget=0;budget<1000;++budget) {
      auto input=saved;bool threw=false;remaining=budget;failing=true;
      try {
        if (!mode) datamunge::algorithms::burstsort(input);
        else {datamunge::algorithms::BurstSortOptions options;options.bucket_capacity=1;datamunge::algorithms::burstsort_with_options(input,options);}
      } catch (const std::bad_alloc&) {threw=true;}
      failing=false;
      if (threw) {
        ++failures;if (input!=saved) return 1;
        datamunge::algorithms::burstsort(input);if (input!=expected) return 1;
      } else {
        if (!failures || input!=expected) return 1;
        std::printf("Burstsort mode %d: %zu injected failure positions, then success\n",mode,failures);complete=true;break;
      }
    }
    if (!complete) return 1;
  }
}
