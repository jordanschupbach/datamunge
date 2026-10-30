/// @file wildcard.cpp
/// @brief Shared last-star matcher, exact work traces, prefix DP and string-column filtering.
#include <datamunge/algorithms/wildcard.hpp>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private scan state, trace storage and explanatory-grid helpers.
namespace {
/// @brief One actual greedy transition, stored only for a requested trace.
struct WildcardEvent {
  std::string action;        ///< Named branch or terminal decision.
  std::size_t before_i;      ///< Text byte offset before the transition.
  std::size_t before_j;      ///< Pattern byte offset before the transition.
  std::size_t after_i;       ///< Text byte offset after the transition.
  std::size_t after_j;       ///< Pattern byte offset after the transition.
  std::size_t star;          ///< Remembered pattern star, or npos when none is retained.
  std::size_t mark;          ///< Text endpoint assigned to the remembered star.
  std::size_t literal_tests; ///< Cumulative literal comparisons after this event.
};
/// @brief Canonical scan shared by decision, work counters and optional event recording.
/// @tparam Count Collect actual work counters when true.
/// @tparam Trace Retain transition records when true; used only with Count=true.
/// @param text Borrowed bytes.
/// @param pattern Restricted glob bytes.
/// @param events Optional trace output, non-null exactly when Trace=true.
/// @param max_events Maximum retained events.
/// @return Acceptance and requested counters.
/// @throws std::length_error If a requested trace exceeds its event budget.
template<bool Count, bool Trace>
WildcardMatchStats wildcard_scan(
    const std::string&          text,
    const std::string&          pattern,
    std::vector<WildcardEvent>* events,
    std::size_t                 max_events) {
  WildcardMatchStats result;
  std::size_t        i = 0, j = 0, star = std::string::npos, mark = 0;
  /// @brief Append a transition only for the trace instantiation.
  /// @param action Name of the branch or terminal decision.
  /// @param before_i Previous text offset.
  /// @param before_j Previous pattern offset.
  const auto emit = [&](const char* action, std::size_t before_i, std::size_t before_j) {
    if constexpr (Trace) {
      if (events->size() >= max_events)
        throw std::length_error("wildcard_match_trace: event budget exceeded");
      events->push_back({action, before_i, before_j, i, j, star, mark, result.literal_tests});
    }
  };
  while (i < text.size()) {
    const auto before_i = i, before_j = j;
    if constexpr (Count)
      ++result.steps;
    if (j < pattern.size() && pattern[j] == '*') {
      star = j;
      mark = i;
      ++j;
      if constexpr (Count)
        ++result.star_steps;
      emit("star-empty", before_i, before_j);
      continue;
    }
    bool equal = false;
    if (j < pattern.size()) {
      if (pattern[j] == '?')
        equal = true;
      else {
        if constexpr (Count)
          ++result.literal_tests;
        equal = pattern[j] == text[i];
      }
    }
    if (equal) {
      const bool question = pattern[j] == '?';
      ++i;
      ++j;
      if constexpr (Count)
        if (question)
          ++result.question_matches;
      emit(question ? "question" : "literal", before_i, before_j);
    } else if (star != std::string::npos) {
      j = star + 1;
      i = ++mark;
      if constexpr (Count)
        ++result.retries;
      emit("expand-star", before_i, before_j);
    } else {
      emit("reject", before_i, before_j);
      return result;
    }
  }
  while (j < pattern.size() && pattern[j] == '*') {
    const auto before_j = j;
    ++j;
    if constexpr (Count) {
      ++result.steps;
      ++result.star_steps;
    }
    emit("trailing-star", i, before_j);
  }
  result.matched = j == pattern.size();
  emit(result.matched ? "accept" : "reject-end", i, j);
  return result;
}
/// @brief Validate and count empty-prefix-inclusive DP cells before allocation.
/// @param n Text length.
/// @param m Pattern length.
/// @param limit Positive cell budget.
/// @return Safe product (n+1)*(m+1).
/// @throws std::invalid_argument If limit is zero.
/// @throws std::length_error If dimensions overflow or exceed the budget.
std::size_t wildcard_cells(std::size_t n, std::size_t m, std::size_t limit) {
  if (limit == 0)
    throw std::invalid_argument("wildcard_dp_table: zero cell budget");
  if (n == std::numeric_limits<std::size_t>::max() ||
      m == std::numeric_limits<std::size_t>::max() || n + 1 > limit / (m + 1))
    throw std::length_error("wildcard_dp_table: cell budget exceeded");
  return (n + 1) * (m + 1);
}
/// @brief Encode a byte as visible ASCII, the word space, or a hexadecimal byte label.
/// @param byte Arbitrary unsigned byte.
/// @return Owned safe printable label; no Unicode decoding.
std::string wildcard_byte_label(unsigned char byte) {
  if (byte == ' ')
    return "space";
  if (byte >= 33 && byte <= 126)
    return std::string(1, static_cast<char>(byte));
  std::ostringstream out;
  out << "0x" << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
  return out.str();
}
} // namespace

bool wildcard_match(const std::string& text, const std::string& pattern) {
  return wildcard_scan<false, false>(text, pattern, nullptr, 0).matched;
}
WildcardMatchStats wildcard_match_stats(const std::string& text, const std::string& pattern) {
  return wildcard_scan<true, false>(text, pattern, nullptr, 0);
}
dstruct::DataFrame
wildcard_match_trace(const std::string& text, const std::string& pattern, std::size_t max_events) {
  if (max_events == 0)
    throw std::invalid_argument("wildcard_match_trace: zero event budget");
  std::vector<WildcardEvent> events;
  (void)wildcard_scan<true, true>(text, pattern, &events, max_events);
  std::vector<std::string> action;
  std::vector<double>      id, bi, bj, ai, aj, star, mark, comparisons;
  for (const auto& event : events) {
    id.push_back(id.size());
    action.push_back(event.action);
    bi.push_back(event.before_i);
    bj.push_back(event.before_j);
    ai.push_back(event.after_i);
    aj.push_back(event.after_j);
    star.push_back(event.star == std::string::npos ? -1.0 : double(event.star));
    mark.push_back(event.mark);
    comparisons.push_back(event.literal_tests);
  }
  dstruct::DataFrame out;
  out.add_column("event", id);
  out.add_column("action", action);
  out.add_column("text_before", bi);
  out.add_column("pattern_before", bj);
  out.add_column("text_after", ai);
  out.add_column("pattern_after", aj);
  out.add_column("star_after", star);
  out.add_column("mark_after", mark);
  out.add_column("literal_tests", comparisons);
  return out;
}
dstruct::DataFrame
wildcard_dp_table(const std::string& text, const std::string& pattern, std::size_t max_cells) {
  const auto cells  = wildcard_cells(text.size(), pattern.size(), max_cells),
             stride = pattern.size() + 1;
  std::vector<unsigned char> reachable(cells, 0);
  reachable[0] = 1;
  for (std::size_t j = 1; j <= pattern.size(); ++j)
    reachable[j] = pattern[j - 1] == '*' && reachable[j - 1];
  for (std::size_t i = 1; i <= text.size(); ++i)
    for (std::size_t j = 1; j <= pattern.size(); ++j) {
      if (pattern[j - 1] == '*')
        reachable[i * stride + j] =
            reachable[i * stride + j - 1] || reachable[(i - 1) * stride + j];
      else if (pattern[j - 1] == '?' || pattern[j - 1] == text[i - 1])
        reachable[i * stride + j] = reachable[(i - 1) * stride + j - 1];
    }
  std::vector<double> i, j, state;
  i.reserve(cells);
  j.reserve(cells);
  state.reserve(cells);
  for (std::size_t r = 0; r <= text.size(); ++r)
    for (std::size_t c = 0; c <= pattern.size(); ++c) {
      i.push_back(r);
      j.push_back(c);
      state.push_back(reachable[r * stride + c]);
    }
  dstruct::DataFrame out;
  out.add_column("text_prefix", i);
  out.add_column("pattern_prefix", j);
  out.add_column("reachable", state);
  return out;
}
graphics::RPlot
wildcard_dp_plot(const std::string& text, const std::string& pattern, std::size_t max_cells) {
  const auto          grid = wildcard_dp_table(text, pattern, max_cells);
  std::vector<double> yes_x, yes_y, no_x, no_y;
  for (std::size_t r = 0; r < grid.nrows(); ++r) {
    const bool reachable = grid.double_at("reachable", r) != 0;
    (reachable ? yes_x : no_x).push_back(grid.double_at("pattern_prefix", r));
    (reachable ? yes_y : no_y).push_back(-grid.double_at("text_prefix", r));
  }
  const bool accepted = grid.double_at("reachable", grid.nrows() - 1) != 0;
  auto       plot     = graphics::RPlot::create();
  plot.points(no_x, no_y, "", {203, 213, 225}, 5)
      .points(yes_x, yes_y, "", {22, 163, 74}, 6)
      .points(
          {double(pattern.size())},
          {-double(text.size())},
          "",
          accepted ? graphics::RGB{202, 138, 4} : graphics::RGB{220, 38, 38},
          10);
  for (std::size_t j = 0; j < pattern.size(); ++j)
    plot.text(
        j + 1, 0.8, wildcard_byte_label(static_cast<unsigned char>(pattern[j])), {37, 99, 235}, 11);
  for (std::size_t i = 0; i < text.size(); ++i)
    plot.text(
        -0.8,
        -double(i + 1),
        wildcard_byte_label(static_cast<unsigned char>(text[i])),
        {100, 116, 139},
        10);
  plot.x_limits(-1.6, pattern.size() + 0.6)
      .y_limits(-double(text.size()) - 0.6, 1.6)
      .x_label("pattern prefix (bytes)")
      .y_label("text prefix (bytes, downwards)")
      .show_grid(false);
  return plot;
}
dstruct::DataFrame wildcard_summary(
    const dstruct::DataFrame&       data,
    const std::string&              column,
    const std::vector<std::string>& patterns) {
  const auto&         strings = data.get_column<std::string>(column);
  std::vector<double> evaluated, skipped, matches, bytes, steps, comparisons, retries;
  for (const auto& pattern : patterns) {
    std::size_t count = 0, accepted = 0, total_bytes = 0, total_steps = 0, total_comparisons = 0,
                total_retries = 0;
    for (std::size_t i = 0; i < data.nrows(); ++i)
      if (!strings.is_null(i)) {
        const auto& text   = strings.at(i);
        const auto  result = wildcard_match_stats(text, pattern);
        ++count;
        accepted += result.matched;
        total_bytes += text.size();
        total_steps += result.steps;
        total_comparisons += result.literal_tests;
        total_retries += result.retries;
      }
    evaluated.push_back(count);
    skipped.push_back(data.nrows() - count);
    matches.push_back(accepted);
    bytes.push_back(total_bytes);
    steps.push_back(total_steps);
    comparisons.push_back(total_comparisons);
    retries.push_back(total_retries);
  }
  dstruct::DataFrame out;
  out.add_column("pattern", patterns);
  out.add_column("input_rows", std::vector<double>(patterns.size(), data.nrows()));
  out.add_column("evaluated_rows", evaluated);
  out.add_column("skipped_rows", skipped);
  out.add_column("matches", matches);
  out.add_column("text_bytes", bytes);
  out.add_column("steps", steps);
  out.add_column("literal_tests", comparisons);
  out.add_column("retries", retries);
  return out;
}
dstruct::DataFrame wildcard_filter(
    const dstruct::DataFrame& data,
    const std::string&        column,
    const std::string&        pattern) {
  const auto&              strings = data.get_column<std::string>(column);
  std::vector<std::size_t> rows;
  for (std::size_t i = 0; i < data.nrows(); ++i)
    if (!strings.is_null(i) && wildcard_match(strings.at(i), pattern))
      rows.push_back(i);
  return data.take_rows(rows);
}
} // namespace datamunge::algorithms
