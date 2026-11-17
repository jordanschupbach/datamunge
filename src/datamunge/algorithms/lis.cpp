/// @file lis.cpp
/// @brief One checked strict minimal-tail LIS producer with actual comparison/update/link capture and source-index witness.
#include <algorithm>
#include <bit>
#include <datamunge/algorithms/lis.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private vector/count admission and same-execution minimal-tail producer.
namespace lis_detail {
/// @brief Add an actual logical count without wrap.
/// @param value Mutable count.
/// @param amount Additional actual count, default1.
/// @throws std::length_error On size_t overflow.
void add(std::size_t& value, std::size_t amount = 1) {
  if (amount > std::numeric_limits<std::size_t>::max() - value)
    throw std::length_error("LIS count overflow");
  value += amount;
}
/// @brief Validate full predecessor/vector and conservative comparison geometry.
/// @param n Original full input length.
/// @throws std::length_error On complete container/count/sentinel geometry.
void geometry(std::size_t n) {
  if (n == std::numeric_limits<std::size_t>::max() || n > std::vector<std::size_t>{}.max_size() ||
      (n && std::bit_width(n) > std::numeric_limits<std::size_t>::max() / n))
    throw std::length_error("LIS complete vector/work geometry");
}
/// @brief Admit actual capture records before their copy/append.
/// @param trace Optional mutable complete capture.
/// @param amount Actual new logical count.
/// @param cap Optional logical cap,0 off.
/// @throws std::length_error On actual quota/count limits.
void admit(LISTrace* trace, std::size_t amount, std::size_t cap) {
  if (!trace)
    return;
  if (cap && (trace->captured_entries > cap || amount > cap - trace->captured_entries))
    throw std::length_error("LIS actual capture quota");
  add(trace->captured_entries, amount);
}
/// @brief Produce strict-LIS tail updates and original-index predecessor reconstruction once.
/// @param input Geometry-admitted original integer sequence.
/// @param trace Optional input/predecessor-admitted complete capture.
/// @param cap Actual logical cap,0 off.
/// @return Owned real selected witness/source indices and selected producing work.
/// @throws std::length_error On actual selected/capture count limits.
/// @throws std::bad_alloc On runtime/output/capture allocations.
LISAnalysis run(const std::vector<int>& input, LISTrace* trace, std::size_t cap) {
  LISAnalysis              out;
  std::vector<std::size_t> tails, prev(input.size(), std::numeric_limits<std::size_t>::max());
  for (std::size_t i = 0; i < input.size(); ++i) {
    std::size_t lo = 0, hi = tails.size();
    while (lo < hi) {
      const auto mid  = lo + (hi - lo) / 2;
      const bool less = input[tails[mid]] < input[i];
      add(out.work.comparisons);
      if (trace) {
        admit(trace, 1, cap);
        trace->probes.push_back({i, lo, hi, mid, tails[mid], input[tails[mid]], less});
      }
      if (less)
        lo = mid + 1;
      else
        hi = mid;
    }
    if (lo)
      prev[i] = tails[lo - 1];
    const bool   append = lo == tails.size();
    LISInsertion record;
    if (trace) {
      std::size_t records = 1;
      add(records, tails.size());
      add(records, tails.size() + append);
      admit(trace, records, cap);
      record.input_index  = i;
      record.length_class = lo;
      record.predecessor  = prev[i];
      record.before       = tails;
    }
    if (append) {
      tails.push_back(i);
      add(out.work.appends);
    } else {
      tails[lo] = i;
      add(out.work.replacements);
    }
    if (trace) {
      record.after = tails;
      trace->insertions.push_back(std::move(record));
    }
  }
  if (!tails.empty())
    for (auto k = tails.back(); k != std::numeric_limits<std::size_t>::max(); k = prev[k]) {
      out.sequence.push_back(input[k]);
      out.indices.push_back(k);
      add(out.work.backtrack);
    }
  std::reverse(out.sequence.begin(), out.sequence.end());
  std::reverse(out.indices.begin(), out.indices.end());
  if (trace) {
    admit(trace, tails.size(), cap);
    trace->predecessor = std::move(prev);
    trace->final_tails = std::move(tails);
  }
  return out;
}
} // namespace lis_detail
std::vector<int> longest_increasing_subsequence(const std::vector<int>& input) {
  return lis_analyze(input).sequence;
}
LISAnalysis lis_analyze(const std::vector<int>& input) {
  lis_detail::geometry(input.size());
  return lis_detail::run(input, nullptr, 0);
}
LISTrace lis_trace(const std::vector<int>& input, std::size_t max_entries) {
  lis_detail::geometry(input.size());
  if (input.size() > std::numeric_limits<std::size_t>::max() / 2)
    throw std::length_error("LIS input capture geometry");
  LISTrace out;
  lis_detail::admit(&out, 2 * input.size(), max_entries);
  out.input    = input;
  out.analysis = lis_detail::run(input, &out, max_entries);
  return out;
}
} // namespace datamunge::algorithms
