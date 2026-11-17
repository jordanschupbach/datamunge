#pragma once
/// @file insertion_sort_analysis.hpp
/// @brief Original insertion-sort workload profile and bounded actual-step diagrams.
#include <algorithm>
#include <cstdint>
#include <datamunge/algorithms/insertion_sort.hpp>
#include <datamunge/plot/diagram.hpp>
#include <format>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief One actual original-workload input and its three producing work records.
/// Random means one permutation from an evolving std::mt19937/std::shuffle
/// stream, not an empirical mean or a portable permutation specification.
/// @par Example
/// @code
/// const dm::algorithms::InsertionSortProfileRow r;std::print("{}\n",r.n); // 0
/// @endcode
/// @see insertion_sort_example_profile()
struct InsertionSortProfileRow {
  std::size_t       n{0};     ///< Actual input size.
  std::vector<int>  shuffled; ///< Owned actual shuffled [0,n) input.
  InsertionSortWork random;   ///< Actual work sorting that permutation.
  InsertionSortWork sorted;   ///< Actual work sorting [0,n).
  InsertionSortWork reversed; ///< Actual work sorting its reversal.
};
/// @brief Reproduce the original evolving seed-one shuffled/sorted/reversed workload.
/// Sizes step,2*step,...<=maximum; defaults reproduce all40 original draws.
/// Complete retained input count sum(step*i) and int-key bounds are admitted
/// before allocation/RNG work. std::shuffle's mapping is toolchain-specific.
/// Each row calls the shared sorting kernel three times, no copied instrumented
/// algorithm. O(sum n²) worst work/O(sum n) owned inputs plus O(maximum) scratch;
/// no mutation, shared state or retained references. Counters exclude generation.
/// @param maximum Largest permitted size, default200; zero yields no rows, at most INT_MAX.
/// @param step Positive size increment, default5.
/// @param seed MT19937 initial seed, default1.
/// @param max_values Complete retained shuffled-value cap, default million; zero disables.
/// @return Owned actual inputs and three selected producing counters per size.
/// @throws std::invalid_argument On step zero.
/// @throws std::length_error On int-key/count overflow or exceeded complete value cap.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto p=dm::algorithms::insertion_sort_example_profile(5);
/// std::print("{} {}\n",p.size(),p[0].sorted.comparisons); // 1 4
/// @endcode
/// @see insertion_sort_run()
inline std::vector<InsertionSortProfileRow> insertion_sort_example_profile(
    std::size_t   maximum    = 200,
    std::size_t   step       = 5,
    std::uint32_t seed       = 1,
    std::size_t   max_values = 1000000) {
  if (!step)
    throw std::invalid_argument("insertion profile positive step");
  if (maximum > std::size_t(std::numeric_limits<int>::max()))
    throw std::length_error("insertion profile int keys");
  const auto  q = maximum / step, max = std::numeric_limits<std::size_t>::max();
  std::size_t a = q, b = q + 1;
  if (a % 2 == 0)
    a /= 2;
  else
    b /= 2;
  if (a && b > max / a)
    throw std::length_error("insertion profile count");
  const auto sum = a * b;
  if (sum && step > max / sum)
    throw std::length_error("insertion profile count");
  if (max_values && step * sum > max_values)
    throw std::length_error("insertion complete profile cap");
  std::vector<InsertionSortProfileRow> out;
  out.reserve(q);
  std::mt19937 rng(seed);
  for (std::size_t i = 1; i <= q; ++i) {
    const auto       n = i * step;
    std::vector<int> base(n);
    std::iota(base.begin(), base.end(), 0);
    auto reverse = base;
    std::reverse(reverse.begin(), reverse.end());
    auto random = base;
    std::shuffle(random.begin(), random.end(), rng);
    InsertionSortProfileRow row;
    row.n        = n;
    row.random   = insertion_sort_run(random, {0, 0}).work;
    row.sorted   = insertion_sort_run(base, {0, 0}).work;
    row.reversed = insertion_sort_run(reverse, {0, 0}).work;
    row.shuffled = std::move(random);
    out.push_back(std::move(row));
  }
  return out;
}
/// @brief Draw one actual insertion boundary without sorting or replaying prefixes.
/// Shows the preceding owned state, actual held key/destination/counts and actual
/// after state. Validates visited dimensions and complete original identity/value
/// bindings, not the mathematical origin of an arbitrarily edited history.
/// Complete display admission precedes validation scratch/layout; O(n) label and
/// validation storage. No mutation, retained references or shared state.
/// @param trace Borrowed original/result/actual insertion records.
/// @param step Zero-based actual insertion record, default0.
/// @param max_elements Complete displayed input cap, default16; zero disables.
/// @return Owned fitted native concept Diagram.
/// @throws std::out_of_range On absent step.
/// @throws std::invalid_argument On visited shape/key/identity errors.
/// @throws std::length_error On exceeded nonzero complete display cap.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::insertion_sort_trace({3,1,2});
/// std::print("{}\n",dm::algorithms::insertion_sort_step_diagram(t).to_svg().empty()); // false
/// @endcode
/// @see insertion_sort_trace()
inline graphics::Diagram insertion_sort_step_diagram(
    const InsertionSortTrace& trace,
    std::size_t               step         = 0,
    std::size_t               max_elements = 16) {
  if (step >= trace.steps.size())
    throw std::out_of_range("insertion absent step");
  const auto n = trace.original.size();
  if (max_elements && n > max_elements)
    throw std::length_error("insertion complete display cap");
  const auto& s = trace.steps[step];
  if (s.index != step + 1 || s.index >= n || s.destination > s.index ||
      s.shifts != s.index - s.destination || s.key_identity >= n ||
      trace.original[s.key_identity] != s.key)
    throw std::invalid_argument("insertion diagram step");
  const auto& before = step ? trace.steps[step - 1].values : trace.original;
  if (before.size() != n || s.values.size() != n || s.identities.size() != n)
    throw std::invalid_argument("insertion diagram dimensions");
  std::vector<unsigned char> seen(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    auto id = s.identities[i];
    if (id >= n || seen[id] || s.values[i] != trace.original[id])
      throw std::invalid_argument("insertion diagram identities");
    seen[id] = 1;
  }
  graphics::Diagram          d;
  graphics::DiagramNodeStyle style;
  style.width = 5.5;
  d.node("before", {0, 0}, std::format("Before insertion {}\n{}", s.index, before), style)
      .node(
          "key",
          {8, 0},
          std::format(
              "Held key {} (id {})\nslot {} -> {}\ncomparisons {}, shifts {}",
              s.key,
              s.key_identity,
              s.index,
              s.destination,
              s.comparisons,
              s.shifts),
          style)
      .node(
          "after",
          {16, 0},
          std::format("After placement\n{}\nids {}", s.values, s.identities),
          style)
      .edge("before", "key")
      .edge("key", "after");
  return d;
}
} // namespace datamunge::algorithms
