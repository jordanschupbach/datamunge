#pragma once
#include <datamunge/algorithms/strand_sort.hpp>
#include <datamunge/algorithms/bitonic_sort.hpp>
#include <datamunge/algorithms/flashsort.hpp>
#include <datamunge/algorithms/patience_sort.hpp>

// A quartet of sorting algorithms with distinct mechanisms:
//   - bitonic_sort  : a data-oblivious sorting network (Batcher), general n
//   - strand_sort   : repeatedly pull out an increasing run and merge it
//   - patience_sort : deal into piles, then k-way merge the pile tops
//   - flashsort     : distribution sort by value classification + permutation
// Each sorts the vector in place into ascending order.

#include <algorithm>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

} // namespace datamunge::algorithms
