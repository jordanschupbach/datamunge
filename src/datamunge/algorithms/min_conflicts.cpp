#include <datamunge/algorithms/min_conflicts.hpp>

#include <cstddef>
#include <random>
#include <vector>

namespace datamunge::algorithms {

namespace {

// Conflict bookkeeping: counts of queens per column and per diagonal.
struct Board {
    int              n;
    std::vector<int> col;     // queens[r] = column of the queen in row r
    std::vector<int> col_cnt; // occupancy of each column
    std::vector<int> d1_cnt;  // "/" diagonals, index r+c
    std::vector<int> d2_cnt;  // "\" diagonals, index r-c+(n-1)

    explicit Board(int n_) : n(n_), col(n_, 0), col_cnt(n_, 0), d1_cnt(2 * n_ - 1, 0), d2_cnt(2 * n_ - 1, 0) {}

    void place(int r, int c) {
        col[r] = c;
        ++col_cnt[c];
        ++d1_cnt[r + c];
        ++d2_cnt[r - c + n - 1];
    }
    void remove(int r, int c) {
        --col_cnt[c];
        --d1_cnt[r + c];
        --d2_cnt[r - c + n - 1];
    }
    // Conflicts a queen at (r,c) has with the others (excluding itself).
    int conflicts_at(int r, int c) const {
        int self_col = (col[r] == c) ? 1 : 0; // whether (r,c) is the current position
        return (col_cnt[c] - self_col) + (d1_cnt[r + c] - self_col) + (d2_cnt[r - c + n - 1] - self_col);
    }
    int total_conflicts() const {
        long long t = 0;
        for (int v : col_cnt) t += static_cast<long long>(v) * (v - 1) / 2;
        for (int v : d1_cnt) t += static_cast<long long>(v) * (v - 1) / 2;
        for (int v : d2_cnt) t += static_cast<long long>(v) * (v - 1) / 2;
        return static_cast<int>(t);
    }
};

} // namespace

NQueensResult min_conflicts_nqueens(int n, int max_steps, unsigned long long seed) {
    NQueensResult result;
    if (n <= 0) {
        result.solved = true;
        return result;
    }

    std::mt19937_64 rng(seed);
    Board           b(n);
    // Greedy-random initial placement: each row's queen goes to a min-conflict column.
    for (int r = 0; r < n; ++r) {
        int              best = -1;
        std::vector<int> cand;
        for (int c = 0; c < n; ++c) {
            const int cf = b.col_cnt[c] + b.d1_cnt[r + c] + b.d2_cnt[r - c + n - 1];
            if (best == -1 || cf < best) { best = cf; cand.clear(); cand.push_back(c); }
            else if (cf == best) cand.push_back(c);
        }
        b.place(r, cand[rng() % cand.size()]);
    }

    for (int step = 0; step < max_steps; ++step) {
        // Gather rows whose queen is in conflict.
        std::vector<int> conflicted;
        for (int r = 0; r < n; ++r)
            if (b.conflicts_at(r, b.col[r]) > 0) conflicted.push_back(r);
        if (conflicted.empty()) {
            result.solved = true;
            result.steps  = step;
            result.queens = b.col;
            return result;
        }

        const int r = conflicted[rng() % conflicted.size()];
        // Move that queen to the column minimizing conflicts (temporarily removed).
        b.remove(r, b.col[r]);
        int              best = -1;
        std::vector<int> cand;
        for (int c = 0; c < n; ++c) {
            const int cf = b.col_cnt[c] + b.d1_cnt[r + c] + b.d2_cnt[r - c + n - 1];
            if (best == -1 || cf < best) { best = cf; cand.clear(); cand.push_back(c); }
            else if (cf == best) cand.push_back(c);
        }
        b.place(r, cand[rng() % cand.size()]);
    }

    result.steps  = max_steps;
    result.queens = b.col;
    result.solved = b.total_conflicts() == 0;
    return result;
}

} // namespace datamunge::algorithms
