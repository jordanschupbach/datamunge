/// @file heap_permutations.cpp
/// @brief Shared legacy-order Heap recursion with complete count/capture preflight.
#include <datamunge/algorithms/heap_permutations.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private checked arithmetic, admission and producing recursion.
namespace heap_detail {
/// @brief Multiply logical counts without wraparound.
/// @param a First count.
/// @param b Second count.
/// @return Exact product.
/// @throws std::length_error On size_t overflow.
std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("Heap count overflow");
  return a * b;
}
/// @brief Increment logical counts without wraparound.
/// @param a Current count.
/// @return Exact successor.
/// @throws std::length_error On size_t overflow.
std::size_t successor(std::size_t a) {
  if (a == std::numeric_limits<std::size_t>::max())
    throw std::length_error("Heap count increment overflow");
  return a + 1;
}
/// @brief Generate one recursive prefix in the original order, capturing actual swaps optionally.
/// @param a Borrowed mutable working values.
/// @param k Prefix length, validated and count-admitted by execute().
/// @param out Borrowed producing result, with reserved outer row capacity.
/// @param swaps Optional borrowed reserved event sink.
/// @throws std::bad_alloc On row allocation failure.
void generate(
    std::vector<int>&                 a,
    std::size_t                       k,
    HeapPermutationRun&               out,
    std::vector<HeapPermutationSwap>* swaps) {
  ++out.work.recursive_calls;
  if (k <= 1) {
    out.permutations.push_back(a);
    ++out.work.emissions;
    out.work.copied_values += a.size();
    return;
  }
  for (std::size_t i = 0; i + 1 < k; ++i) {
    generate(a, k - 1, out, swaps);
    const std::size_t left = k % 2 == 0 ? i : 0, right = k - 1;
    const int         lv = a[left], rv = a[right];
    std::swap(a[left], a[right]);
    ++out.work.swaps;
    if (k % 2 == 0)
      ++out.work.even_swaps;
    else
      ++out.work.odd_swaps;
    if (swaps)
      swaps->push_back({out.permutations.size(), k, i, left, right, lv, rv});
  }
  generate(a, k - 1, out, swaps);
}
/// @brief Preflight all known requirements before allocating or copying, then execute once.
/// @param items Borrowed arbitrary integer values.
/// @param o Complete output/history policies.
/// @param swaps Optional borrowed initially empty trace sink.
/// @return Owned actual rows and selected producing work.
/// @throws std::length_error On policy/count/container requirements.
/// @throws std::bad_alloc On allocation failure.
HeapPermutationRun execute(
    const std::vector<int>&           items,
    HeapPermutationOptions            o,
    std::vector<HeapPermutationSwap>* swaps) {
  const auto         s = heap_permutation_size(items.size());
  HeapPermutationRun r;
  if ((o.max_permutations && s.permutations > o.max_permutations) ||
      (o.max_values && s.copied_values > o.max_values) ||
      s.permutations > r.permutations.max_size())
    throw std::length_error("Heap complete output limit");
  const auto entries = s.permutations - 1;
  if (swaps &&
      ((o.max_trace_entries && entries > o.max_trace_entries) || entries > swaps->max_size()))
    throw std::length_error("Heap complete history limit");
  if (swaps)
    swaps->reserve(entries);
  r.permutations.reserve(s.permutations);
  auto a = items;
  generate(a, a.size(), r, swaps);
  return r;
}
} // namespace heap_detail
HeapPermutationSize heap_permutation_size(std::size_t n) {
  HeapPermutationSize s;
  for (std::size_t k = 2; k <= n; ++k) {
    s.permutations    = heap_detail::multiply(s.permutations, k);
    s.recursive_calls = heap_detail::successor(heap_detail::multiply(s.recursive_calls, k));
  }
  s.copied_values = heap_detail::multiply(n, s.permutations);
  return s;
}
std::vector<std::vector<int>> heap_permutations(const std::vector<int>& items) {
  return heap_detail::execute(items, {0, 0, 0}, nullptr).permutations;
}
HeapPermutationRun heap_permutations_run(
    const std::vector<int>& items,
    HeapPermutationOptions  options) {
  return heap_detail::execute(items, options, nullptr);
}
HeapPermutationTrace heap_permutations_trace(
    const std::vector<int>& items,
    HeapPermutationOptions  options) {
  HeapPermutationTrace t;
  t.run = heap_detail::execute(items, options, &t.swaps);
  return t;
}
} // namespace datamunge::algorithms
