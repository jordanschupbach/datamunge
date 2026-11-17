#pragma once
/// @file jaro_winkler_analysis.hpp
/// @brief Original name fixtures, actual match graphics and bounded native candidate ranking.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/jaro_winkler.hpp>
#include <datamunge/plot/diagram.hpp>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Return all five original name/string pairs in their original order.
/// Owned fixed data, constant bounded work/storage, no RNG/shared state or
/// retained references. ASCII spellings and case are unchanged.
/// @return Complete original five-pair fixture.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::jaro_winkler_name_pairs()[0].first); // MARTHA
/// @endcode
/// @see jaro_winkler_run()
inline std::vector<std::pair<std::string, std::string>> jaro_winkler_name_pairs() {
  return {
      {"MARTHA", "MARHTA"},
      {"DIXON", "DICKSONX"},
      {"DWAYNE", "DUANE"},
      {"MARTHA", "MARDHA"},
      {"abcde", "fghij"}};
}
/// @brief Draw actual accepted match partners from a recorded byte-matching execution.
/// Original first/second byte rows are aligned by index. Edges come only from
/// accepted Candidate events; this is different from Order-event pairing.
/// ASCII letters/digits are shown directly; other bytes use hexadecimal labels.
/// Complete input display admission precedes visited validation/graphics. Validates
/// visited accepted-edge indices, byte/source bindings and one-to-one use, not
/// complete provenance of caller-edited histories. No scoring/RNG replay, input
/// mutation or retained references. O(E+n+m) validation and O(n+m+matches)
/// native graphic primitives/storage; caps count bytes, not capacity/peak RAM.
/// @param trace Borrowed actual owned-byte/event history.
/// @param max_total_bytes Complete displayed input-byte cap, default64; zero disables.
/// @return Owned fitted native concept Diagram of actual accepted match partners.
/// @throws std::invalid_argument On empty inputs or inconsistent accepted-edge fields.
/// @throws std::length_error On complete byte/count cap overflow.
/// @throws std::bad_alloc On graphics allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::jaro_winkler_trace("MARTHA","MARHTA");
/// std::print("{}\n",dm::algorithms::jaro_winkler_match_diagram(t).to_svg().empty()); // false
/// @endcode
/// @see jaro_winkler_trace()
inline graphics::Diagram jaro_winkler_match_diagram(
    const JaroWinklerTrace& trace,
    std::size_t             max_total_bytes = 64) {
  const auto n = trace.first.size(), m = trace.second.size();
  if (n > std::numeric_limits<std::size_t>::max() - m ||
      (max_total_bytes && n + m > max_total_bytes))
    throw std::length_error("Jaro diagram complete byte cap");
  if (!n || !m)
    throw std::invalid_argument("Jaro diagram nonempty inputs");
  std::vector<bool>                                used_a(n), used_b(m);
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  for (const auto& e : trace.events)
    if (e.phase == JaroWinklerPhase::Candidate && e.compared && e.equal) {
      if (e.first_index >= n || e.second_index >= m || e.occupied || used_a[e.first_index] ||
          used_b[e.second_index] ||
          e.first_byte != static_cast<std::uint8_t>(trace.first[e.first_index]) ||
          e.second_byte != static_cast<std::uint8_t>(trace.second[e.second_index]) ||
          e.first_byte != e.second_byte)
        throw std::invalid_argument("Jaro diagram accepted match metadata");
      used_a[e.first_index]  = true;
      used_b[e.second_index] = true;
      edges.push_back({e.first_index, e.second_index});
    }
  if (edges.size() != trace.run.matches)
    throw std::invalid_argument("Jaro diagram actual match count");
  graphics::Diagram          d;
  graphics::DiagramNodeStyle style;
  style.width = 1.5;
  for (std::size_t row = 0; row < 2; ++row) {
    const auto& bytes = row ? trace.second : trace.first;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      const auto c     = static_cast<unsigned char>(bytes[i]);
      const auto label = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                             ? std::string(1, char(c))
                             : std::format("{:02X}", unsigned(c));
      const auto id    = std::format("{}{}", row ? 'b' : 'a', i);
      d.node(
          id, {2. * i, row ? 0. : 3.}, std::format("{}[{}]\n{}", row ? 'b' : 'a', i, label), style);
    }
  }
  for (const auto& [i, j] : edges)
    d.edge(std::format("a{}", i), std::format("b{}", j));
  return d;
}
/// @brief Complete candidate-count and input-byte policies for native name ranking.
/// All candidates and query bytes are admitted before scoring/copying. Zero
/// disables either optional cap. Excludes working scores/flags, returned name
/// copies, capacities and byte/peak RAM; no partial top-k prefix on error.
/// @par Example
/// @code
/// const dm::algorithms::JaroWinklerRankLimits l;
/// std::print("{} {}\n",l.max_candidates,l.max_total_bytes); // 10000 1000000
/// @endcode
/// @see jaro_winkler_rank()
struct JaroWinklerRankLimits {
  std::size_t max_candidates{10000}; ///< Complete candidate row count; zero disables.
  std::size_t max_total_bytes{
      1000000}; ///< Query bytes plus all candidate bytes, counted once; zero disables.
};
/// @brief One owned ranked name and its actual single-pass scoring result.
/// index is original input row. Name is copied only for returned rows. No input
/// lifetime dependency, identity/probability assertion or shared mutable state.
/// @par Example
/// @code
/// const auto r=dm::algorithms::jaro_winkler_rank({"MARHTA","DIXON"},"MARTHA",1);
/// std::print("{} {}\n",r[0].index,r[0].value); // 0 MARHTA
/// @endcode
/// @see jaro_winkler_rank()
struct JaroWinklerRankedName {
  std::size_t index{0}; ///< Original zero-based candidate row, also the score-tie breaker.
  std::string value;    ///< Owned complete candidate bytes, unnormalized.
  JaroWinklerRun
      score; ///< Actual query-first/candidate-second result, computed once before sorting.
};
/// @brief Rank all candidates by unconditional byte Winkler score, with original-row ties.
/// Validates finite scale[0,.25], complete candidate count and query+all-name
/// byte total before scoring/copying. Each candidate is scored exactly once with
/// query first; sorting uses decreasing score then increasing original index.
/// Returns min(k,N) owned rows; k=0 still validates complete inputs but performs
/// no scoring. Duplicate names remain distinct rows. No case folding, tokenization,
/// missing-value inference, Unicode normalization or entity-resolution calibration.
/// No mutation, retained references or shared state. O(N log N) ranking plus
/// all pair-score costs, O(N+longest pair length) working storage, plus returned
/// top-k name copies. A top-k return is not an early-stop input/work admission.
/// @param candidates Borrowed complete ordered candidate names, including empty/binary strings.
/// @param query Borrowed complete unnormalized query bytes.
/// @param k Maximum returned rows, default10; zero returns empty after validation.
/// @param prefix_scale Finite scale[0,.25], default .1.
/// @param limits Complete candidate and total-input policies.
/// @return Owned top-k actual scores/names, preserving original row indices and deterministic ties.
/// @throws std::invalid_argument On invalid scale before scoring/allocation.
/// @throws std::length_error On complete cap or size/count overflow.
/// @throws std::bad_alloc On score/flag/output allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::jaro_winkler_rank({"DIXON","MARHTA","MARTHA"},"MARTHA",2);
/// std::print("{} {}\n",r[0].index,r[1].index); // 2 1
/// @endcode
/// @see jaro_winkler_run(), JaroWinklerRankLimits
inline std::vector<JaroWinklerRankedName> jaro_winkler_rank(
    const std::vector<std::string>& candidates,
    const std::string&              query,
    std::size_t                     k            = 10,
    double                          prefix_scale = .1,
    JaroWinklerRankLimits           limits       = {}) {
  if (!std::isfinite(prefix_scale) || prefix_scale < 0 || prefix_scale > .25)
    throw std::invalid_argument("Winkler rank scale");
  if (limits.max_candidates && candidates.size() > limits.max_candidates)
    throw std::length_error("Winkler complete candidate cap");
  std::size_t total = query.size();
  for (const auto& value : candidates) {
    if (value.size() > std::numeric_limits<std::size_t>::max() - total)
      throw std::length_error("Winkler rank byte count");
    total += value.size();
  }
  if (limits.max_total_bytes && total > limits.max_total_bytes)
    throw std::length_error("Winkler complete ranking bytes");
  if (k == 0)
    return {};
  std::vector<JaroWinklerRankedName> scored;
  scored.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i)
    scored.push_back({i, {}, jaro_winkler_run(query, candidates[i], prefix_scale, {0, 0})});
  /// @brief Stable-by-index strict score ordering; every score is finite by construction.
  /// @param a First actual scoring row.
  /// @param b Second actual scoring row.
  /// @return Decreasing Winkler score, then increasing original row index.
  const auto order = [](const auto& a, const auto& b) {
    return a.score.winkler != b.score.winkler ? a.score.winkler > b.score.winkler
                                              : a.index < b.index;
  };
  std::sort(scored.begin(), scored.end(), order);
  scored.resize(std::min(k, scored.size()));
  for (auto& row : scored)
    row.value = candidates[row.index];
  return scored;
}
} // namespace datamunge::algorithms
