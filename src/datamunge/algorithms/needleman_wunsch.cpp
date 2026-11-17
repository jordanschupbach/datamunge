/// @file needleman_wunsch.cpp
/// @brief Checked canonical global alignment with actual fill and reverse-traceback records.
#include <algorithm>
#include <cstdint>
#include <datamunge/algorithms/needleman_wunsch.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete admission and shared global-alignment producer.
namespace {
/// @brief Add complete geometry/capture counts without wrapping.
/// @param a First nonnegative count.
/// @param b Second nonnegative count.
/// @return Checked sum.
/// @throws std::length_error If addition exceeds size_t.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Needleman-Wunsch: count overflow");
  return a + b;
}
/// @brief Multiply complete geometry counts without wrapping.
/// @param a First count.
/// @param b Second count.
/// @return Checked product.
/// @throws std::length_error If product exceeds size_t.
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("Needleman-Wunsch: geometry overflow");
  return a * b;
}
/// @brief Magnitude of a scoring int without negating INT_MIN in int.
/// @param value Arbitrary int score.
/// @return Exact widened nonnegative magnitude.
std::uint64_t magnitude(int value) {
  return value < 0 ? std::uint64_t(-std::int64_t(value)) : std::uint64_t(value);
}
/// @brief Execute complete admitted DP and actual diagonal-priority traceback.
/// @param a Original first bytes.
/// @param b Original second bytes.
/// @param scoring Complete linear score scheme.
/// @param options Complete cell/capture caps, zero disables each.
/// @param out Optional actual trace sink.
/// @return Owned native alignment, score table and work.
/// @throws std::invalid_argument If an input hyphen occurs.
/// @throws std::overflow_error If score bounds exceed int.
/// @throws std::length_error If geometry or capture/cell policies exceed limits.
NeedlemanWunschRun run(
    const std::string&     a,
    const std::string&     b,
    AlignmentScoring       scoring,
    NeedlemanWunschOptions options,
    NeedlemanWunschTrace*  out) {
  if (a.find('-') != std::string::npos || b.find('-') != std::string::npos)
    throw std::invalid_argument("Needleman-Wunsch: input '-' reserved for inserted gaps");
  const auto n = a.size(), m = b.size(), length = add(n, m);
  if (length > std::size_t(std::numeric_limits<int>::max()) || length > std::string{}.max_size())
    throw std::length_error("Needleman-Wunsch: input/output length limit");
  const auto largest =
      std::max({magnitude(scoring.match), magnitude(scoring.mismatch), magnitude(scoring.gap)});
  if (largest && length > std::uint64_t(std::numeric_limits<int>::max()) / largest)
    throw std::overflow_error("Needleman-Wunsch: conservative int score domain");
  const auto rows = add(n, 1), columns = add(m, 1), count = product(rows, columns),
             interior = product(n, m);
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(int))
    throw std::length_error("Needleman-Wunsch: table byte geometry overflow");
  if (options.max_cells && count > options.max_cells)
    throw std::length_error("Needleman-Wunsch: complete cell cap exceeded");
  /// @brief Admit actual owned history units before copy/allocation/append.
  auto admit = [&](std::size_t amount) {
    if (!out)
      return;
    auto next = add(out->captured_entries, amount);
    if (options.max_trace_entries && next > options.max_trace_entries)
      throw std::length_error("Needleman-Wunsch: capture quota exceeded");
    out->captured_entries = next;
  };
  if (out) {
    admit(add(3, add(length, add(rows, add(count, interior)))));
    out->a       = a;
    out->b       = b;
    out->scoring = scoring;
    out->cells.reserve(interior);
  }
  NeedlemanWunschRun r;
  r.cells = count;
  r.scores.assign(rows, std::vector<int>(columns));
  auto& h = r.scores;
  for (std::size_t i = 0; i <= n; ++i) {
    h[i][0] = int(i) * scoring.gap;
    ++r.boundary_assignments;
  }
  for (std::size_t j = 0; j <= m; ++j) {
    h[0][j] = int(j) * scoring.gap;
    ++r.boundary_assignments;
  }
  for (std::size_t i = 1; i <= n; ++i)
    for (std::size_t j = 1; j <= m; ++j) {
      const int diagonal =
                    h[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? scoring.match : scoring.mismatch),
                up = h[i - 1][j] + scoring.gap, left = h[i][j - 1] + scoring.gap;
      int  best   = diagonal;
      auto chosen = NeedlemanWunschMove::Diagonal;
      if (up > best) {
        best   = up;
        chosen = NeedlemanWunschMove::Up;
      }
      if (left > best) {
        best   = left;
        chosen = NeedlemanWunschMove::Left;
      }
      h[i][j] = best;
      ++r.interior_cells;
      if (out)
        out->cells.push_back({i, j, diagonal, up, left, best, chosen});
    }
  r.alignment.score = h[n][m];
  r.alignment.a_aligned.reserve(length);
  r.alignment.b_aligned.reserve(length);
  std::size_t i = n, j = m;
  while (i || j) {
    NeedlemanWunschTraceback step;
    step.i_before = i;
    step.j_before = j;
    if (i && j &&
        h[i][j] == h[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? scoring.match : scoring.mismatch)) {
      step.move = NeedlemanWunschMove::Diagonal;
      step.a    = a[--i];
      step.b    = b[--j];
      ++r.diagonal_steps;
    } else if (i && h[i][j] == h[i - 1][j] + scoring.gap) {
      step.move = NeedlemanWunschMove::Up;
      step.a    = a[--i];
      step.b    = '-';
      ++r.up_steps;
    } else {
      step.move = NeedlemanWunschMove::Left;
      step.a    = '-';
      step.b    = b[--j];
      ++r.left_steps;
    }
    step.i_after = i;
    step.j_after = j;
    if (out) {
      admit(3);
      out->traceback.push_back(step);
    }
    r.alignment.a_aligned.push_back(step.a);
    r.alignment.b_aligned.push_back(step.b);
    ++r.traceback_steps;
  }
  std::reverse(r.alignment.a_aligned.begin(), r.alignment.a_aligned.end());
  std::reverse(r.alignment.b_aligned.begin(), r.alignment.b_aligned.end());
  return r;
}
} // namespace
std::string_view needleman_wunsch_move_name(NeedlemanWunschMove move) noexcept {
  switch (move) {
    case NeedlemanWunschMove::Diagonal:
      return "diagonal";
    case NeedlemanWunschMove::Up:
      return "up";
    case NeedlemanWunschMove::Left:
      return "left";
  }
  return "unknown";
}
Alignment needleman_wunsch(const std::string& a, const std::string& b, AlignmentScoring s) {
  return run(a, b, s, {0, 0}, nullptr).alignment;
}
NeedlemanWunschRun needleman_wunsch_run(
    const std::string&     a,
    const std::string&     b,
    AlignmentScoring       s,
    NeedlemanWunschOptions o) {
  return run(a, b, s, o, nullptr);
}
NeedlemanWunschTrace needleman_wunsch_trace(
    const std::string&     a,
    const std::string&     b,
    AlignmentScoring       s,
    NeedlemanWunschOptions o) {
  NeedlemanWunschTrace t;
  t.run = run(a, b, s, o, &t);
  return t;
}
std::vector<std::pair<std::string, std::string>> needleman_wunsch_demo_pairs() {
  return {{"GATTACA", "GCATGCU"}, {"HELLO", "HELP"}, {"ALIGN", "ALIGNMENT"}};
}
} // namespace datamunge::algorithms
