/// @file kmp.cpp
/// @brief Shared original-order prefix/text KMP producer with optional actual operand records.
#include <datamunge/algorithms/kmp.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Checked logical counters and optional same-execution observation state.
namespace kmp_detail {
/// @brief Add an actual logical count without wrap.
/// @param count Mutable producing counter.
/// @param amount Additional entries or operations.
/// @throws std::length_error On size_t overflow.
void add(std::size_t& count, std::size_t amount) {
  if (amount > std::numeric_limits<std::size_t>::max() - count)
    throw std::length_error("KMP count overflow");
  count += amount;
}
/// @brief Optional capture shared by the actual prefix builder and text scan.
struct Observer {
  KMPWork         work;  ///< Actual producing comparisons/fallbacks/resets.
  KMPSearchTrace* trace; ///< Borrowed output for this call only, null for scalar/count mode.
  std::size_t     cap;   ///< Optional retained logical-entry ceiling, zero off.

  /// @brief Admit retained entries before their producing allocation/evaluation.
  /// @param n Additional original-byte/table/test/match entries.
  /// @throws std::length_error On counter or enabled quota overflow.
  void admit(std::size_t n) {
    if (!trace)
      return;
    auto next = trace->captured_entries;
    add(next, n);
    if (cap && next > cap)
      throw std::length_error("KMP trace quota");
    trace->captured_entries = next;
  }
  /// @brief Perform and optionally capture one actual byte equality test.
  /// @param phase Actual table/text producer.
  /// @param site Actual while/extend site.
  /// @param i Original source-byte index.
  /// @param q Actual pre-test pattern index.
  /// @param source Actual source byte operand.
  /// @param pattern Actual pattern byte operand.
  /// @return Actual equality used by the producing branch.
  /// @throws std::length_error On capture/work limits.
  /// @throws std::bad_alloc On observation append failure.
  bool
  equal(KMPPhase phase, KMPSite site, std::size_t i, std::size_t q, char source, char pattern) {
    admit(1);
    const bool same = source == pattern;
    add(phase == KMPPhase::Prefix ? work.prefix_comparisons : work.text_comparisons, 1);
    if (trace)
      trace->comparisons.push_back(
          {phase,
           site,
           i,
           q,
           static_cast<unsigned char>(source),
           static_cast<unsigned char>(pattern),
           same});
    return same;
  }
};
/// @brief Build the actual prefix table using the preserved mismatch-loop then equality order.
/// @param pattern Borrowed immutable bytes.
/// @param o Local producing observation state.
/// @return Owned actual prefix table.
/// @throws std::length_error On capture/count/container limits.
/// @throws std::bad_alloc On table/observation allocation failure.
std::vector<std::size_t> prefix(const std::string& pattern, Observer& o) {
  const auto m = pattern.size();
  o.admit(m);
  std::vector<std::size_t> pi(m, 0);
  std::size_t              q = 0;
  for (std::size_t i = 1; i < m; ++i) {
    while (q > 0 && !o.equal(KMPPhase::Prefix, KMPSite::While, i, q, pattern[i], pattern[q])) {
      q = pi[q - 1];
      add(o.work.prefix_fallbacks, 1);
    }
    if (o.equal(KMPPhase::Prefix, KMPSite::Extend, i, q, pattern[i], pattern[q]))
      ++q;
    pi[i] = q;
  }
  return pi;
}
/// @brief Produce exact matches once, sharing actual prefix/scan observation logic.
/// @param text Borrowed complete original bytes.
/// @param pattern Borrowed complete original needle.
/// @param o Local producing observation state.
/// @return Owned actual ascending starts.
/// @throws std::length_error On capture/count/container limits.
/// @throws std::bad_alloc On table/output/observation allocation failure.
std::vector<std::size_t> run(const std::string& text, const std::string& pattern, Observer& o) {
  std::vector<std::size_t> matches;
  const auto               n = text.size(), m = pattern.size();
  if (o.trace) {
    o.admit(n);
    o.admit(m);
    o.trace->text    = text;
    o.trace->pattern = pattern;
  }
  if (m == 0 || m > n)
    return matches;
  const auto pi = prefix(pattern, o);
  if (o.trace)
    o.trace->prefix = pi;
  std::size_t q = 0;
  for (std::size_t i = 0; i < n; ++i) {
    while (q > 0 && !o.equal(KMPPhase::Text, KMPSite::While, i, q, text[i], pattern[q])) {
      q = pi[q - 1];
      add(o.work.text_fallbacks, 1);
    }
    if (o.equal(KMPPhase::Text, KMPSite::Extend, i, q, text[i], pattern[q]))
      ++q;
    if (q == m) {
      o.admit(1);
      matches.push_back(i - m + 1);
      q = pi[q - 1];
      add(o.work.match_resets, 1);
    }
  }
  return matches;
}
} // namespace kmp_detail
std::vector<std::size_t> kmp_prefix_function(const std::string& pattern) {
  kmp_detail::Observer o{{}, nullptr, 0};
  return kmp_detail::prefix(pattern, o);
}
std::vector<std::size_t> kmp_search(const std::string& text, const std::string& pattern) {
  kmp_detail::Observer o{{}, nullptr, 0};
  return kmp_detail::run(text, pattern, o);
}
KMPSearchAnalysis kmp_search_analyze(const std::string& text, const std::string& pattern) {
  kmp_detail::Observer o{{}, nullptr, 0};
  KMPSearchAnalysis    out;
  out.matches = kmp_detail::run(text, pattern, o);
  out.work    = o.work;
  return out;
}
KMPSearchTrace
kmp_search_trace(const std::string& text, const std::string& pattern, std::size_t max_entries) {
  KMPSearchTrace       out;
  kmp_detail::Observer o{{}, &out, max_entries};
  out.matches = kmp_detail::run(text, pattern, o);
  out.work    = o.work;
  return out;
}
} // namespace datamunge::algorithms
