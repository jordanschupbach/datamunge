/// @file postman_sort.cpp
/// @brief Shared actual byte-MSD recursion, stable distribution and insertion fallback.
#include <datamunge/algorithms/postman_sort.hpp>
#include <random>
#include <stdexcept>
#include <limits>
#include <utility>
namespace datamunge::algorithms {
/// @brief Local checked producer and optional actual-history implementation.
namespace postman_detail {
/// @brief Selected-work/history policy and owned producing result.
/// All recursion and accounting is local to a single run.
struct Context {
  PostmanOptions options;    ///< Borrow-free scalar policies.
  PostmanTrace   output;     ///< Owned actual output and optional nodes.
  bool           capture;    ///< Whether producing events retain node records.
  std::size_t    entries{0}; ///< Actual logical retained entries admitted so far.
};
/// @brief Admit one actual selected event before its producing action.
/// @param c Local mutable producing context.
/// @param counter Selected event counter to increment.
/// @throws std::length_error If the selected cap or size_t representation is exhausted.
void event(Context& c, std::size_t& counter) {
  if (c.output.run.operations == std::numeric_limits<std::size_t>::max() ||
      (c.options.max_operations && c.output.run.operations >= c.options.max_operations))
    throw std::length_error("postman work cap");
  ++c.output.run.operations;
  ++counter;
}
/// @brief Process one actual half-open range using the original recursion order.
/// @param c Local owning context.
/// @param lo Inclusive array index.
/// @param hi Exclusive array index.
/// @param shift Actual current byte shift.
/// @throws std::length_error If a selected-operation/history policy or representable storage is exceeded.
void recurse(Context& c, std::size_t lo, std::size_t hi, int shift) {
  auto& r = c.output.run;
  auto& a = r.sorted;
  ++r.nodes;
  bool        insertion = hi - lo <= c.options.insertion_cutoff || shift < 0;
  PostmanNode node{lo, hi, shift, insertion};
  if (c.capture) {
    std::size_t cost = insertion ? 1 : 514;
    if (cost > std::numeric_limits<std::size_t>::max() - c.entries ||
        (c.options.max_trace_entries && cost > c.options.max_trace_entries - c.entries))
      throw std::length_error("postman history cap");
    c.entries += cost;
  }
  if (insertion) {
    if (c.capture)
      c.output.nodes.push_back(node);
    for (std::size_t i = lo + 1; i < hi; ++i) {
      auto        key = a[i];
      std::size_t j   = i;
      while (j > lo) {
        event(c, r.comparisons);
        if (!(a[j - 1] > key))
          break;
        event(c, r.moves);
        a[j] = a[j - 1];
        --j;
      }
      event(c, r.moves);
      a[j] = key;
    }
    return;
  }
  ++r.radix_nodes;
  std::array<std::size_t, 257> count{};
  for (std::size_t i = lo; i < hi; ++i) {
    event(c, r.key_scans);
    ++count[((a[i] >> shift) & 255) + 1];
  }
  for (std::size_t i = 0; i < 256; ++i)
    node.counts[i] = count[i + 1];
  for (std::size_t i = 1; i < 257; ++i) {
    event(c, r.prefix_steps);
    count[i] += count[i - 1];
  }
  node.offsets = count;
  if (c.capture)
    c.output.nodes.push_back(node);
  std::vector<std::uint32_t>   tmp(hi - lo);
  std::array<std::size_t, 256> cursor{};
  for (std::size_t i = 0; i < 256; ++i)
    cursor[i] = count[i];
  for (std::size_t i = lo; i < hi; ++i) {
    event(c, r.distributions);
    tmp[cursor[(a[i] >> shift) & 255]++] = a[i];
  }
  for (std::size_t i = lo; i < hi; ++i) {
    event(c, r.copybacks);
    a[i] = tmp[i - lo];
  }
  for (std::size_t i = 0; i < 256; ++i)
    if (count[i + 1] - count[i] > 1)
      recurse(c, lo + count[i], lo + count[i + 1], shift - 8);
}
/// @brief Start the single producing recursion after taking ownership.
/// @param a Owned complete input.
/// @param options Selected policies.
/// @param capture Whether to retain actual nodes.
/// @return Owned output and optional histories.
/// @throws std::length_error If a selected-operation/history policy or representable storage is exceeded.
PostmanTrace produce(std::vector<std::uint32_t> a, PostmanOptions options, bool capture) {
  Context c{options, {}, capture};
  c.output.run.sorted = std::move(a);
  if (c.output.run.sorted.size() > 1)
    recurse(c, 0, c.output.run.sorted.size(), 24);
  return std::move(c.output);
}
} // namespace postman_detail
PostmanRun postman_sort_run(std::vector<std::uint32_t> a, PostmanOptions o) {
  return postman_detail::produce(std::move(a), o, false).run;
}
PostmanTrace postman_sort_trace(std::vector<std::uint32_t> a, PostmanOptions o) {
  return postman_detail::produce(std::move(a), o, true);
}
void postman_sort(std::vector<std::uint32_t>& a) {
  a = postman_sort_run(a).sorted;
}
std::vector<std::uint32_t>
postman_uniform_keys(std::size_t count, std::uint64_t seed, std::size_t max_keys) {
  if (max_keys && count > max_keys)
    throw std::length_error("postman fixture count");
  std::vector<std::uint32_t> keys(count);
  std::mt19937_64            rng(seed);
  for (auto& v : keys)
    v = static_cast<std::uint32_t>(rng());
  return keys;
}
} // namespace datamunge::algorithms
