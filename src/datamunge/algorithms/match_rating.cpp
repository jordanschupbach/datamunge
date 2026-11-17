/// @file match_rating.cpp
/// @brief Canonical match rating API, producing histories and checked native contracts.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <datamunge/algorithms/match_rating.hpp>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Implementation-only helpers for canonical match rating production and capture.
namespace {
/// @brief Shared complete logical capture admission across all actual pair phases.
struct Budget {
  std::size_t count{0}; ///< Actual already admitted entries.
  std::size_t limit{0}; ///< Complete cap, zero off.

  /// @brief Admit actual next records/characters before copying or producing them.
  /// @param n Actual added entry count.
  /// @throws std::length_error On count overflow or exceeded optional cap.
  void admit(std::size_t n) {
    if (n > SIZE_MAX - count || (limit && (count > limit || n > limit - count)))
      throw std::length_error("MRA complete capture");
    count += n;
  }
};
/// @brief Canonical actual name-byte normalization, kept-letter filtering and final trimming.
/// @param name Original arbitrary bytes.
/// @param trace Optional owned actual capture target.
/// @param budget Shared capture admission, nonnull exactly when trace is nonnull.
/// @return Actual normalization/filter/trim phase strings.
/// @throws std::length_error On complete actual capture admission failure.
MatchRatingCodexAnalysis
codex(const std::string& name, MatchRatingCodexTrace* trace, Budget* budget) {
  MatchRatingCodexAnalysis out;
  std::size_t              start = budget ? budget->count : 0;
  if (trace) {
    budget->admit(name.size());
    trace->input = name;
  }
  unsigned char previous = 0;
  bool          first    = true;
  for (std::size_t i = 0; i < name.size(); ++i) {
    auto          raw         = static_cast<unsigned char>(name[i]);
    bool          letter      = raw != ' ' && std::isalpha(raw);
    unsigned char upper       = letter ? static_cast<unsigned char>(std::toupper(raw)) : 0;
    auto          oldprevious = previous;
    bool          oldfirst    = first;
    auto          before      = out.stripped.size();
    bool          kept        = false;
    if (trace)
      budget->admit(1);
    if (letter) {
      kept = first ||
             ((upper != 'A' && upper != 'E' && upper != 'I' && upper != 'O' && upper != 'U') &&
              upper != previous);
      if (trace)
        budget->admit(kept ? 2 : 1);
      out.normalized.push_back(static_cast<char>(upper));
      if (kept)
        out.stripped.push_back(static_cast<char>(upper));
      previous = upper;
      first    = false;
    }
    if (trace)
      trace->steps.push_back(
          {i, raw, letter, upper, oldprevious, oldfirst, kept, before, out.stripped.size()});
  }
  auto length = std::min(std::size_t{6}, out.stripped.size());
  if (trace)
    budget->admit(length);
  out.codex = out.stripped.size() > 6
                  ? out.stripped.substr(0, 3) + out.stripped.substr(out.stripped.size() - 3)
                  : out.stripped;
  if (trace)
    trace->captured_entries = budget->count - start;
  return out;
}
/// @brief Canonical actual codex length guard, threshold and aligned two-pass cancellation.
/// @param a Actual first final codex.
/// @param b Actual second final codex.
/// @param trace Optional pair capture target.
/// @param budget Shared complete admission when capturing.
/// @return Actual comparison phases and original-layout verdict.
/// @throws std::length_error On complete actual capture admission failure.
MatchRatingAnalysis compare(std::string a, std::string b, MatchRatingTrace* trace, Budget* budget) {
  MatchRatingAnalysis out;
  out.codex_a    = std::move(a);
  out.codex_b    = std::move(b);
  const auto& a1 = out.codex_a;
  const auto& b1 = out.codex_b;
  int         n = a1.size(), m = b1.size();
  if (std::abs(n - m) >= 3)
    return out;
  out.result.comparable = true;
  int total             = n + m;
  out.result.min_rating = total <= 4 ? 5 : total <= 7 ? 4 : total <= 11 ? 3 : 2;
  for (std::size_t i = 0; i < std::max(a1.size(), b1.size()); ++i) {
    unsigned char x = i < a1.size() ? static_cast<unsigned char>(a1[i]) : 0,
                  y = i < b1.size() ? static_cast<unsigned char>(b1[i]) : 0;
    auto beforea = out.residual_a.size(), beforeb = out.residual_b.size();
    bool equal = x == y;
    if (trace)
      budget->admit(1 + (!equal && x ? 1 : 0) + (!equal && y ? 1 : 0));
    if (!equal) {
      if (x)
        out.residual_a.push_back(static_cast<char>(x));
      if (y)
        out.residual_b.push_back(static_cast<char>(y));
    }
    if (trace)
      trace->cancellations.push_back(
          {false, i, x, y, equal, beforea, beforeb, out.residual_a.size(), out.residual_b.size()});
  }
  for (std::size_t i = 0; i < std::max(out.residual_a.size(), out.residual_b.size()); ++i) {
    unsigned char x     = i < out.residual_a.size() ? static_cast<unsigned char>(
                                                      out.residual_a[out.residual_a.size() - 1 - i])
                                                    : 0,
                  y     = i < out.residual_b.size() ? static_cast<unsigned char>(
                                                      out.residual_b[out.residual_b.size() - 1 - i])
                                                    : 0;
    std::size_t beforea = out.unmatched_a, beforeb = out.unmatched_b;
    bool        equal = x == y;
    if (trace)
      budget->admit(1);
    if (!equal) {
      out.unmatched_a += x != 0;
      out.unmatched_b += y != 0;
    }
    if (trace)
      trace->cancellations.push_back(
          {true,
           i,
           x,
           y,
           equal,
           beforea,
           beforeb,
           static_cast<std::size_t>(out.unmatched_a),
           static_cast<std::size_t>(out.unmatched_b)});
  }
  out.result.similarity = 6 - std::max(out.unmatched_a, out.unmatched_b);
  out.result.match      = out.result.similarity >= out.result.min_rating;
  return out;
}
} // namespace
std::string match_rating_codex(const std::string& name) {
  return codex(name, nullptr, nullptr).codex;
}
MatchRatingCodexAnalysis match_rating_codex_analyze(const std::string& name) {
  return codex(name, nullptr, nullptr);
}
MatchRatingCodexTrace match_rating_codex_trace(const std::string& name, std::size_t limit) {
  Budget                budget{0, limit};
  MatchRatingCodexTrace out;
  out.analysis = codex(name, &out, &budget);
  return out;
}
MatchRatingResult match_rating_comparison(const std::string& a, const std::string& b) {
  return match_rating_analyze(a, b).result;
}
MatchRatingAnalysis match_rating_analyze(const std::string& a, const std::string& b) {
  auto c1 = codex(a, nullptr, nullptr), c2 = codex(b, nullptr, nullptr);
  return compare(std::move(c1.codex), std::move(c2.codex), nullptr, nullptr);
}
MatchRatingTrace match_rating_trace(const std::string& a, const std::string& b, std::size_t limit) {
  Budget           budget{0, limit};
  MatchRatingTrace out;
  out.a.analysis       = codex(a, &out.a, &budget);
  out.b.analysis       = codex(b, &out.b, &budget);
  out.analysis         = compare(out.a.analysis.codex, out.b.analysis.codex, &out, &budget);
  out.captured_entries = budget.count;
  return out;
}
std::vector<std::pair<std::string, std::string>> match_rating_demo_pairs() {
  return {
      {"Byrne", "Boern"},
      {"Smith", "Smyth"},
      {"Catherine", "Kathryn"},
      {"Robert", "Rupert"},
      {"Steven", "Stephen"},
      {"Karen", "Caren"},
      {"Sean", "Shaun"},
      {"Margaret", "Marguerite"},
      {"Alice", "Zachary"},
      {"Jackson", "Johnson"},
      {"Baker", "Barker"},
      {"Nelson", "Wilson"}};
}
} // namespace datamunge::algorithms
