/// @file bead_sort.cpp
/// @brief Bead sort through conjugate partitions, and a tick-by-tick gravity simulation.

#include <datamunge/algorithms/sorting_hybrid.hpp>

#include <stdexcept>

namespace datamunge::algorithms {

std::vector<unsigned> conjugate_partition(const std::vector<unsigned>& parts) {
    unsigned mx = 0;
    for (const unsigned p : parts) mx = p > mx ? p : mx;
    // histogram[v] = number of parts equal to v; then lambda'_j = #{parts > j} by a suffix sum.
    std::vector<unsigned> histogram(static_cast<std::size_t>(mx) + 1, 0);
    for (const unsigned p : parts) ++histogram[p];
    std::vector<unsigned> conj(mx, 0);
    unsigned above = 0;
    for (unsigned j = mx; j-- > 0;) {
        above += histogram[j + 1];
        conj[j] = above;
    }
    return conj;
}

std::vector<unsigned> bead_columns(const std::vector<unsigned>& a) { return conjugate_partition(a); }

void bead_sort(std::vector<unsigned>& a) {
    const std::vector<unsigned> rows = conjugate_partition(conjugate_partition(a));  // non-increasing, no zeros
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) a[n - 1 - i] = i < rows.size() ? rows[i] : 0U;
}

BeadFall bead_sort_gravity(const std::vector<unsigned>& a) {
    const std::size_t n = a.size();
    unsigned mx = 0;
    for (const unsigned v : a) mx = v > mx ? v : mx;
    if (static_cast<double>(n) * mx > 1e8) throw std::invalid_argument("bead_sort_gravity: grid larger than 1e8 cells");
    BeadFall out;
    out.before.assign(n, std::vector<unsigned char>(mx, 0));
    for (std::size_t r = 0; r < n; ++r)
        for (unsigned j = 0; j < a[r]; ++j) out.before[r][j] = 1;
    auto grid = out.before;
    for (;;) {
        long long moved = 0;
        // Bottom-up so that each bead moves at most one cell per tick.
        for (std::size_t r = n; r-- > 1;)
            for (unsigned j = 0; j < mx; ++j)
                if (grid[r][j] == 0 && grid[r - 1][j] == 1) {
                    grid[r][j] = 2;  // landed this tick; cannot move again until the next one
                    grid[r - 1][j] = 0;
                    ++moved;
                }
        for (auto& row : grid)
            for (auto& c : row)
                if (c == 2) c = 1;
        if (moved == 0) break;
        ++out.ticks;
        out.bead_moves += moved;
        out.moves_per_tick.push_back(moved);
    }
    out.after = grid;
    return out;
}

} // namespace datamunge::algorithms
