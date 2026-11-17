/// @file ruzzo_tompa.cpp
/// @brief One actual widened-prefix candidate list with honest direct-scan work and histories.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/ruzzo_tompa.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical scored-sequence implementation.
namespace datamunge::algorithms {
/// @brief Private actual integration state.
namespace {
/// @brief One actual internal candidate and its endpoint prefix scores.
struct Candidate {
  std::size_t begin; ///< Inclusive source index.
  std::size_t end;   ///< Exclusive source index.
  long double left;  ///< Actual prefix before begin.
  long double right; ///< Actual prefix at end.
};
/// @brief Convert one actual positive endpoint-score difference.
/// @param x Actual candidate.
/// @return Owned finite positive public segment.
/// @throws std::overflow_error If score is not representable as finite positive double.
ScoringSegment public_segment(const Candidate& x) {
  const auto score = x.right - x.left;
  if (!std::isfinite(score) || score <= 0 || score > std::numeric_limits<double>::max() ||
      static_cast<double>(score) == 0)
    throw std::overflow_error("Ruzzo: represented segment score");
  return {x.begin, x.end, static_cast<double>(score)};
}
/// @brief Execute one direct predecessor-scan stream.
/// @param scores Complete finite scores.
/// @param o Policies.
/// @param capture Retain actual list states.
/// @return Owned final intervals and actual work.
RuzzoRun solve(const std::vector<double>& scores, const RuzzoOptions& o, bool capture) {
  if (o.max_values && scores.size() > o.max_values)
    throw std::length_error("Ruzzo: source policy");
  for (double x : scores)
    if (!std::isfinite(x))
      throw std::invalid_argument("Ruzzo: nonfinite source");
  RuzzoRun               out;
  std::vector<Candidate> list;
  long double            prefix   = 0;
  std::size_t            retained = 0;
  /// @brief Count one actual prefix-order comparison before its branch.
  const auto compare = [&]() {
    if (out.comparisons == std::numeric_limits<std::size_t>::max() ||
        (o.max_comparisons && out.comparisons >= o.max_comparisons))
      throw std::length_error("Ruzzo: comparison policy");
    ++out.comparisons;
  };
  for (std::size_t i = 0; i < scores.size(); ++i) {
    const auto before = prefix;
    prefix += scores[i];
    if (!std::isfinite(prefix))
      throw std::overflow_error("Ruzzo: prefix range");
    if (scores[i] != 0 && prefix == before)
      throw std::underflow_error("Ruzzo: unresolved prefix increment");
    const auto old_comparisons = out.comparisons, old_absorbed = out.absorbed;
    ++out.values;
    if (scores[i] > 0) {
      ++out.positive_seeds;
      Candidate segment{i, i + 1, before, prefix};
      while (true) {
        std::size_t predecessor = list.size();
        bool        found       = false;
        for (std::size_t k = list.size(); k-- > 0;) {
          compare();
          if (list[k].left < segment.left) {
            predecessor = k;
            found       = true;
            break;
          }
        }
        if (!found)
          break;
        compare();
        if (list[predecessor].right >= segment.right)
          break;
        segment.begin = list[predecessor].begin;
        segment.left  = list[predecessor].left;
        out.absorbed += list.size() - predecessor;
        list.resize(predecessor);
      }
      list.push_back(segment);
      out.peak_candidates = std::max(out.peak_candidates, list.size());
    }
    if (capture) {
      if (list.size() == std::numeric_limits<std::size_t>::max() ||
          list.size() + 1 > std::numeric_limits<std::size_t>::max() - retained ||
          (o.max_trace_entries &&
           (retained > o.max_trace_entries || list.size() + 1 > o.max_trace_entries - retained)))
        throw std::length_error("Ruzzo: history policy");
      retained += list.size() + 1;
      RuzzoStep event{
          i,
          before,
          prefix,
          scores[i] > 0,
          out.comparisons - old_comparisons,
          out.absorbed - old_absorbed,
          {}};
      for (const auto& x : list)
        event.candidates.push_back(public_segment(x));
      out.steps.push_back(std::move(event));
    }
  }
  for (const auto& x : list)
    out.segments.push_back(public_segment(x));
  return out;
}
} // namespace
RuzzoRun ruzzo_tompa_run(const std::vector<double>& s, const RuzzoOptions& o) {
  return solve(s, o, false);
}
RuzzoRun ruzzo_tompa_trace(const std::vector<double>& s, const RuzzoOptions& o) {
  return solve(s, o, true);
}
std::vector<ScoringSegment> ruzzo_tompa(const std::vector<double>& s) {
  return solve(s, {}, false).segments;
}
std::vector<double> ruzzo_alternating_scores(std::size_t positive_count) {
  if (positive_count > 500000)
    throw std::length_error("Ruzzo: fixture policy");
  if (!positive_count)
    return {};
  std::vector<double> values(2 * positive_count - 1);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = i % 2 ? -1. : 1.;
  return values;
}
} // namespace datamunge::algorithms
