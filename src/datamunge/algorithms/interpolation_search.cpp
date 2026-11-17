/// @file interpolation_search.cpp
/// @brief Exact widened floor arithmetic in one shared sorted-input interpolation loop.
#include <datamunge/algorithms/interpolation_search.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked counters, complete admission and actual execution.
namespace {
/// @brief Increment one selected count without wrap.
/// @param n Mutable actual count.
/// @throws std::length_error On exhausted size_t.
void increment(std::size_t& n) {
  if (n == std::numeric_limits<std::size_t>::max())
    throw std::length_error("interpolation count");
  ++n;
}
/// @brief Execute shared exact-floor interpolation without validating sortedness.
/// @param a Borrowed nondecreasing complete input.
/// @param key Target int.
/// @param options Complete input and actual history policies.
/// @param capture Whether to retain each actual candidate boundary.
/// @return Actual result/work and optional chronological scalar records.
/// @throws std::length_error On ptrdiff_t length, enabled cap or count/capacity overflow.
/// @throws std::bad_alloc On optional capture allocation failure.
InterpolationSearchTrace
execute(const std::vector<int>& a, int key, InterpolationSearchOptions options, bool capture) {
  static_assert(std::numeric_limits<int>::digits <= 63);
  if (a.size() > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) ||
      (options.max_elements && a.size() > options.max_elements))
    throw std::length_error("interpolation complete input size");
  InterpolationSearchTrace out;
  auto&                    r  = out.run.result;
  auto&                    w  = out.run.work;
  std::ptrdiff_t           lo = 0, hi = static_cast<std::ptrdiff_t>(a.size()) - 1;
  while (lo <= hi) {
    increment(w.endpoint_guards);
    const int low = a[std::size_t(lo)];
    if (key < low)
      break;
    increment(w.endpoint_guards);
    const int high = a[std::size_t(hi)];
    if (key > high)
      break;
    increment(w.flat_tests);
    const bool    flat      = low == high;
    std::uint64_t numerator = 0, denominator = 0;
    std::size_t   offset = 0;
    if (!flat) {
      numerator   = static_cast<std::uint64_t>(static_cast<__int128>(key) - low);
      denominator = static_cast<std::uint64_t>(static_cast<__int128>(high) - low);
      offset      = static_cast<std::size_t>(
          (static_cast<unsigned __int128>(numerator) * static_cast<std::size_t>(hi - lo)) /
          denominator);
    }
    const auto pos = lo + static_cast<std::ptrdiff_t>(offset);
    if (capture && options.max_probes && out.probes.size() >= options.max_probes)
      throw std::length_error("interpolation actual probe cap");
    const int value = a[std::size_t(pos)];
    increment(r.probes);
    if (capture)
      out.probes.push_back({lo, hi, pos, low, high, value, numerator, denominator, offset, flat});
    if (flat) {
      r.found = true;
      r.index = pos;
      break;
    }
    increment(w.key_comparisons);
    if (value == key) {
      r.found = true;
      r.index = pos;
      break;
    }
    increment(w.key_comparisons);
    if (value < key)
      lo = pos + 1;
    else
      hi = pos - 1;
  }
  return out;
}
} // namespace
SearchResult interpolation_search(const std::vector<int>& a, int key) {
  return execute(a, key, {0, 0}, false).run.result;
}
InterpolationSearchRun
interpolation_search_run(const std::vector<int>& a, int key, InterpolationSearchOptions options) {
  return execute(a, key, options, false).run;
}
InterpolationSearchTrace
interpolation_search_trace(const std::vector<int>& a, int key, InterpolationSearchOptions options) {
  return execute(a, key, options, true);
}
} // namespace datamunge::algorithms
