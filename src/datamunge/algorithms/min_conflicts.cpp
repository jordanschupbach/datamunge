/// @file min_conflicts.cpp
/// @brief Shared seeded n-queens producer with optional actual move capture.
#include <cstddef>
#include <datamunge/algorithms/min_conflicts.hpp>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// @brief Private occupancy and canonical search implementation.
namespace {

/// @brief Owned column and diagonal occupancies for one partially or fully placed board.
struct Board {
  int              n;       ///< Validated board size.
  std::vector<int> col;     ///< Current columns by row.
  std::vector<int> col_cnt; ///< Actual column occupancy.
  std::vector<int> d1_cnt;  ///< Actual r+c diagonal occupancy.
  std::vector<int> d2_cnt;  ///< Actual r-c+n-1 diagonal occupancy.

  /// @brief Allocate zero occupancies for a positive validated board.
  /// @param n_ Positive board size, at most INT_MAX/3.
  explicit Board(int n_)
      : n(n_), col(n_, 0), col_cnt(n_, 0), d1_cnt(2 * n_ - 1, 0), d2_cnt(2 * n_ - 1, 0) {}

  /// @brief Place one queen, incrementing all three actual occupancies.
  /// @param r Valid row.
  /// @param c Valid column.
  void place(int r, int c) {
    col[r] = c;
    ++col_cnt[c];
    ++d1_cnt[r + c];
    ++d2_cnt[r - c + n - 1];
  }
  /// @brief Remove one queen from the three occupancies.
  /// @param r Occupied row.
  /// @param c Its actual current column.
  void remove(int r, int c) {
    --col_cnt[c];
    --d1_cnt[r + c];
    --d2_cnt[r - c + n - 1];
  }
  /// @brief Count attacks on the current queen, excluding itself.
  /// @param r Placed row.
  /// @param c Its current column.
  /// @return Actual other queens sharing a column or diagonal.
  int conflicts_at(int r, int c) const {
    int self_col = (col[r] == c) ? 1 : 0; // whether (r,c) is the current position
    return (col_cnt[c] - self_col) + (d1_cnt[r + c] - self_col) +
           (d2_cnt[r - c + n - 1] - self_col);
  }
  /// @brief Count unordered attacking pairs from actual occupancy multiplicities.
  /// @return Widened exact sum of choose(count,2) over the three constraint families.
  long long total_conflicts() const {
    long long t = 0;
    for (int v : col_cnt)
      t += static_cast<long long>(v) * (v - 1) / 2;
    for (int v : d1_cnt)
      t += static_cast<long long>(v) * (v - 1) / 2;
    for (int v : d2_cnt)
      t += static_cast<long long>(v) * (v - 1) / 2;
    return t;
  }
};

} // namespace

/// @brief Private occupancy and canonical search implementation.
namespace {
/// @brief Execute original greedy initialization and capped actual repairs.
/// @param n Nonnegative board size, validated in body.
/// @param max_steps Nonnegative repair cap.
/// @param seed Actual MT19937-64 seed.
/// @param out Optional actual-history sink.
/// @param cap Capture quota, zero disables.
/// @return Owned actual final board and verified status.
/// @throws std::invalid_argument If size/cap is negative.
/// @throws std::length_error If count geometry or quota is exceeded.
NQueensResult
run_queens(int n, int max_steps, unsigned long long seed, NQueensTrace* out, std::size_t cap) {
  if (n < 0 || max_steps < 0)
    throw std::invalid_argument("min_conflicts: negative size or step cap");
  if (n > std::numeric_limits<int>::max() / 3)
    throw std::length_error("min_conflicts: count geometry exceeded");
  /// @brief Admit actual captured units before appending.
  auto admit = [&](std::size_t amount) {
    if (!out)
      return;
    if (amount > std::numeric_limits<std::size_t>::max() - out->captured_entries ||
        (cap && (out->captured_entries > cap || amount > cap - out->captured_entries)))
      throw std::length_error("min_conflicts: capture quota exceeded");
    out->captured_entries += amount;
  };
  if (out)
    admit(3);
  NQueensResult result;
  if (n <= 0) {
    result.solved = true;
    return result;
  }

  std::mt19937_64 rng(seed);
  Board           b(n);
  // Greedy-random initial placement: each row's queen goes to a min-conflict column.
  for (int r = 0; r < n; ++r) {
    NQueensMove record;
    if (out) {
      admit(1 + std::size_t(n));
      record.initialization   = true;
      record.row              = r;
      record.column_before    = -1;
      record.conflicts_before = b.total_conflicts();
      record.costs.reserve(n);
    }
    int              best = -1;
    std::vector<int> cand;
    for (int c = 0; c < n; ++c) {
      const int cf = b.col_cnt[c] + b.d1_cnt[r + c] + b.d2_cnt[r - c + n - 1];
      if (out)
        record.costs.push_back(cf);
      if (best == -1 || cf < best) {
        best = cf;
        cand.clear();
        cand.push_back(c);
      } else if (cf == best)
        cand.push_back(c);
    }
    const auto draw   = rng();
    const auto column = cand[draw % cand.size()];
    b.place(r, column);
    if (out) {
      admit(cand.size());
      record.candidates      = cand;
      record.column_draw     = draw;
      record.column_after    = column;
      record.conflicts_after = b.total_conflicts();
      out->moves.push_back(std::move(record));
    }
  }

  for (int step = 0; step < max_steps; ++step) {
    // Gather rows whose queen is in conflict.
    std::vector<int> conflicted;
    for (int r = 0; r < n; ++r)
      if (b.conflicts_at(r, b.col[r]) > 0)
        conflicted.push_back(r);
    if (conflicted.empty()) {
      result.solved = true;
      result.steps  = step;
      if (out)
        admit(b.col.size());
      result.queens = b.col;
      return result;
    }

    const auto  row_draw = rng();
    const int   r        = conflicted[row_draw % conflicted.size()];
    NQueensMove record;
    if (out) {
      admit(1 + std::size_t(n));
      admit(conflicted.size());
      record.row              = r;
      record.column_before    = b.col[r];
      record.row_draw         = row_draw;
      record.conflicted       = conflicted;
      record.conflicts_before = b.total_conflicts();
      record.costs.reserve(n);
    }
    // Move that queen to the column minimizing conflicts (temporarily removed).
    b.remove(r, b.col[r]);
    int              best = -1;
    std::vector<int> cand;
    for (int c = 0; c < n; ++c) {
      const int cf = b.col_cnt[c] + b.d1_cnt[r + c] + b.d2_cnt[r - c + n - 1];
      if (out)
        record.costs.push_back(cf);
      if (best == -1 || cf < best) {
        best = cf;
        cand.clear();
        cand.push_back(c);
      } else if (cf == best)
        cand.push_back(c);
    }
    const auto draw   = rng();
    const auto column = cand[draw % cand.size()];
    b.place(r, column);
    if (out) {
      admit(cand.size());
      record.candidates      = cand;
      record.column_draw     = draw;
      record.column_after    = column;
      record.conflicts_after = b.total_conflicts();
      out->moves.push_back(std::move(record));
    }
  }

  if (out)
    admit(b.col.size());
  result.steps  = max_steps;
  result.queens = b.col;
  result.solved = b.total_conflicts() == 0;
  return result;
}

} // namespace
NQueensResult min_conflicts_nqueens(int n, int steps, unsigned long long seed) {
  return run_queens(n, steps, seed, nullptr, 0);
}
NQueensTrace
min_conflicts_nqueens_trace(int n, int steps, unsigned long long seed, std::size_t cap) {
  NQueensTrace t;
  t.n         = n;
  t.max_steps = steps;
  t.seed      = seed;
  t.result    = run_queens(n, steps, seed, &t, cap);
  return t;
}
} // namespace datamunge::algorithms
