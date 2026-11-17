/// @file hirschberg.cpp
/// @brief View-based Hirschberg with released rolling profiles and constant-score-space base ties.
#include <algorithm>
#include <cstdint>
#include <datamunge/algorithms/hirschberg.hpp>
#include <limits>
#include <stdexcept>
#include <string_view>
namespace datamunge::algorithms {
/// @brief Private validated execution context, rolling score sweeps and output-appending recursion.
namespace hirschberg_detail {
/// @brief Checked logical count sum.
/// @param a First count.
/// @param b Second count.
/// @return Exact sum.
/// @throws std::length_error On overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Hirschberg count overflow");
  return a + b;
}
/// @brief Checked logical count product.
/// @param a First factor.
/// @param b Second factor.
/// @return Exact product.
/// @throws std::length_error On overflow.
std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("Hirschberg count product overflow");
  return a * b;
}
/// @brief Borrowed owned-result/history sinks and copied immutable execution policy.
struct Context {
  AlignmentScoring  scoring; ///< Actual symmetric scoring scheme.
  HirschbergOptions options; ///< Actual cap/orientation policy.
  HirschbergRun*    run;     ///< Borrowed producing result/work sink.
  HirschbergTrace*  trace;   ///< Optional borrowed actual split sink.
};
/// @brief Admit a complete sweep/base scan before its allocation or pair work.
/// @param c Producing context.
/// @param cells Upcoming selected work count.
/// @throws std::length_error On count or work policy limit.
void admit_work(Context& c, std::size_t cells) {
  auto total = add(add(c.run->work.rolling_cells, c.run->work.base_pairs), cells);
  if (c.options.max_work_cells && total > c.options.max_work_cells)
    throw std::length_error("Hirschberg online work limit");
}
/// @brief Score one byte pair, including NUL, under validated integer bounds.
/// @param a First byte.
/// @param b Second byte.
/// @param s Copied scoring scheme.
/// @return Exact equal/unequal byte score.
int pair(char a, char b, AlignmentScoring s) {
  return a == b ? s.match : s.mismatch;
}
/// @brief Produce actual last DP row, optionally indexing both input views backwards.
/// @param a Borrowed oriented first subrange.
/// @param b Borrowed oriented row-dimension subrange.
/// @param reverse Reverse both traversals without string allocation.
/// @param held Actual score values retained from the preceding forward row.
/// @param c Producing context, counters and work policy.
/// @return Owned actual last row, moved without copying.
/// @throws std::length_error On work/count/container limits before sweep.
/// @throws std::bad_alloc On row allocation failure.
std::vector<int>
row(std::string_view a, std::string_view b, bool reverse, std::size_t held, Context& c) {
  admit_work(c, multiply(a.size(), b.size()));
  const auto       columns = add(b.size(), 1);
  std::vector<int> previous(columns), current(columns);
  c.run->work.peak_score_slots =
      std::max(c.run->work.peak_score_slots, add(held, add(previous.size(), current.size())));
  for (std::size_t j = 0; j <= b.size(); ++j)
    previous[j] = static_cast<int>(j) * c.scoring.gap;
  for (std::size_t i = 1; i <= a.size(); ++i) {
    current[0] = static_cast<int>(i) * c.scoring.gap;
    for (std::size_t j = 1; j <= b.size(); ++j) {
      const char x = reverse ? a[a.size() - i] : a[i - 1], y = reverse ? b[b.size() - j] : b[j - 1];
      current[j] = std::max(
          {previous[j - 1] + pair(x, y, c.scoring),
           previous[j] + c.scoring.gap,
           current[j - 1] + c.scoring.gap});
      ++c.run->work.rolling_cells;
    }
    previous.swap(current);
  }
  return previous;
}
/// @brief Append a one-character base alignment with original diagonal/up/left NW ties.
/// @param a Nonempty borrowed first range.
/// @param b Nonempty borrowed second range, with at least one range of size1.
/// @param c Producing context and output/work sink.
/// @throws std::length_error On work/count/output limits.
/// @throws std::bad_alloc On output allocation failure.
void base(std::string_view a, std::string_view b, Context& c) {
  const bool single_a = a.size() == 1;
  const auto length   = single_a ? b.size() : a.size();
  admit_work(c, length);
  int         best_pair = std::numeric_limits<int>::min(), last_pair = 0;
  std::size_t at = 0;
  for (std::size_t i = 0; i < length; ++i) {
    const int value = single_a ? pair(a[0], b[i], c.scoring) : pair(a[i], b[0], c.scoring);
    ++c.run->work.base_pairs;
    if (value >= best_pair) {
      best_pair = value;
      at        = i;
    }
    last_pair = value;
  }
  const int gap_only = static_cast<int>(length + 1) * c.scoring.gap,
            diagonal = best_pair + static_cast<int>(length - 1) * c.scoring.gap;
  const int score    = std::max(gap_only, diagonal);
  // At a single-a traceback, up beats an earlier tied diagonal unless the last diagonal ties.
  const bool use_diagonal =
      single_a ? (last_pair + static_cast<int>(length - 1) * c.scoring.gap == score ||
                  diagonal > gap_only)
               : (diagonal >= gap_only);
  auto& out = c.run->alignment;
  if (single_a) {
    if (use_diagonal) {
      out.a_aligned.append(at, '-');
      out.a_aligned.push_back(a[0]);
      out.a_aligned.append(length - at - 1, '-');
      out.b_aligned.append(b);
    } else {
      out.a_aligned.append(length, '-');
      out.a_aligned.push_back(a[0]);
      out.b_aligned.append(b);
      out.b_aligned.push_back('-');
    }
  } else {
    if (use_diagonal) {
      out.a_aligned.append(a);
      out.b_aligned.append(at, '-');
      out.b_aligned.push_back(b[0]);
      out.b_aligned.append(length - at - 1, '-');
    } else {
      out.a_aligned.push_back('-');
      out.a_aligned.append(a);
      out.b_aligned.push_back(b[0]);
      out.b_aligned.append(length, '-');
    }
  }
  out.score += score;
}
/// @brief Append one subproblem alignment, releasing parent score vectors before either child.
/// @param a Borrowed oriented first range.
/// @param b Borrowed oriented second range.
/// @param a0 First range's global offset in oriented input.
/// @param b0 Second range's global offset in oriented input.
/// @param depth Actual recursive depth, root1.
/// @param c Producing context and optional online capture sink.
/// @throws std::length_error On input/work/capture/count/container limits.
/// @throws std::bad_alloc On score/history/output allocation.
void recurse(
    std::string_view a,
    std::string_view b,
    std::size_t      a0,
    std::size_t      b0,
    std::size_t      depth,
    Context&         c) {
  c.run->work.calls     = add(c.run->work.calls, 1);
  c.run->work.max_depth = std::max(c.run->work.max_depth, depth);
  auto& out             = c.run->alignment;
  if (a.empty()) {
    out.a_aligned.append(b.size(), '-');
    out.b_aligned.append(b);
    out.score += static_cast<int>(b.size()) * c.scoring.gap;
    return;
  }
  if (b.empty()) {
    out.a_aligned.append(a);
    out.b_aligned.append(a.size(), '-');
    out.score += static_cast<int>(a.size()) * c.scoring.gap;
    return;
  }
  if (a.size() == 1 || b.size() == 1) {
    base(a, b, c);
    return;
  }
  const auto  mid   = a.size() / 2;
  std::size_t split = 0;
  {
    std::size_t record = 0;
    if (c.trace) {
      const auto units = add(1, multiply(3, add(b.size(), 1)));
      const auto total = add(c.trace->captured_entries, units);
      if ((c.options.max_trace_entries && total > c.options.max_trace_entries) ||
          c.trace->splits.size() == c.trace->splits.max_size())
        throw std::length_error("Hirschberg online split limit");
      c.trace->captured_entries = total;
      record                    = c.trace->splits.size();
      c.trace->splits.emplace_back();
    }
    const auto forward = row(a.substr(0, mid), b, false, 0, c);
    const auto reverse = row(a.substr(mid), b, true, forward.size(), c);
    int        best    = std::numeric_limits<int>::min();
    if (c.trace) {
      auto& r   = c.trace->splits[record];
      r.a_begin = a0;
      r.a_end   = a0 + a.size();
      r.b_begin = b0;
      r.b_end   = b0 + b.size();
      r.depth   = depth;
      r.a_mid   = a0 + mid;
      r.forward = forward;
      r.reverse.resize(reverse.size());
      r.total.resize(reverse.size());
    }
    for (std::size_t j = 0; j <= b.size(); ++j) {
      const int score               = forward[j] + reverse[b.size() - j];
      c.run->work.profile_positions = add(c.run->work.profile_positions, 1);
      if (score > best) {
        best  = score;
        split = j;
      }
      if (c.trace) {
        auto& r      = c.trace->splits[record];
        r.reverse[j] = reverse[b.size() - j];
        r.total[j]   = score;
      }
    }
    if (c.trace)
      c.trace->splits[record].b_mid = b0 + split;
  }
  recurse(a.substr(0, mid), b.substr(0, split), a0, b0, depth + 1, c);
  recurse(a.substr(mid), b.substr(split), a0 + mid, b0 + split, depth + 1, c);
}
/// @brief Compute unsigned magnitude without negating INT_MIN in int arithmetic.
/// @param x Any integer scoring component.
/// @return Exact nonnegative magnitude in uint64_t.
std::uint64_t magnitude(int x) {
  return x < 0 ? -std::int64_t(x) : std::int64_t(x);
}
/// @brief Validate complete byte/scoring domains then run on oriented borrowed views.
/// @param a Borrowed first input.
/// @param b Borrowed second input.
/// @param scoring Integer linear scores.
/// @param options Complete input and online policies.
/// @param trace Optional initially empty actual split sink.
/// @return Owned producing alignment/work/orientation.
/// @throws std::invalid_argument On reserved input '-'.
/// @throws std::length_error On input/work/history/count/container limits.
/// @throws std::overflow_error On conservative score admission failure.
/// @throws std::bad_alloc On allocation failure.
HirschbergRun execute(
    const std::string& a,
    const std::string& b,
    AlignmentScoring   scoring,
    HirschbergOptions  options,
    HirschbergTrace*   trace) {
  if (a.find('-') != std::string::npos || b.find('-') != std::string::npos)
    throw std::invalid_argument("Hirschberg input '-' reserved for inserted gaps");
  const auto length = add(a.size(), b.size());
  if (length > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      length > std::string{}.max_size())
    throw std::length_error("Hirschberg input/output length limit");
  const auto largest =
      std::max({magnitude(scoring.match), magnitude(scoring.mismatch), magnitude(scoring.gap)});
  if (largest && length > std::uint64_t(std::numeric_limits<int>::max()) / largest)
    throw std::overflow_error("Hirschberg conservative int score domain");
  if (options.max_input_bytes && length > options.max_input_bytes)
    throw std::length_error("Hirschberg complete input limit");
  HirschbergRun run;
  run.transposed = options.orient_shorter && a.size() < b.size();
  run.alignment.a_aligned.reserve(length);
  run.alignment.b_aligned.reserve(length);
  Context c{scoring, options, &run, trace};
  recurse(
      run.transposed ? std::string_view(b) : std::string_view(a),
      run.transposed ? std::string_view(a) : std::string_view(b),
      0,
      0,
      1,
      c);
  if (run.transposed)
    run.alignment.a_aligned.swap(run.alignment.b_aligned);
  return run;
}
} // namespace hirschberg_detail
Alignment hirschberg(const std::string& a, const std::string& b, AlignmentScoring scoring) {
  return hirschberg_detail::execute(a, b, scoring, {false, 0, 0, 0}, nullptr).alignment;
}
HirschbergRun hirschberg_run(
    const std::string& a,
    const std::string& b,
    AlignmentScoring   scoring,
    HirschbergOptions  options) {
  return hirschberg_detail::execute(a, b, scoring, options, nullptr);
}
HirschbergTrace hirschberg_trace(
    const std::string& a,
    const std::string& b,
    AlignmentScoring   scoring,
    HirschbergOptions  options) {
  HirschbergTrace t;
  t.run = hirschberg_detail::execute(a, b, scoring, options, &t);
  return t;
}
} // namespace datamunge::algorithms
