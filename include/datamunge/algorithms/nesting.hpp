#pragma once

// Nesting / bin packing: pack a set of one-dimensional items into as few
// fixed-capacity bins as possible -- the abstraction behind cutting stock,
// container loading, and memory allocation. The problem is NP-hard, but the
// First-Fit-Decreasing (FFD) heuristic is fast and provably good: sort the items
// largest-first and drop each into the first bin it fits, opening a new bin only
// when none has room. FFD never uses more than 11/9 * OPT + 6/9 bins, and on many
// practical instances hits the optimum outright.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace datamunge::algorithms {

struct NestingResult {
    int                           bins{0};
    std::vector<std::vector<int>> assignment; // item indices per bin
    std::vector<double>           loads;       // total size in each bin
};

// Pack `sizes` into bins of capacity `capacity` by First-Fit-Decreasing.
// Item indices in the result refer to the original `sizes` order.
inline NestingResult nest_first_fit_decreasing(const std::vector<double>& sizes, double capacity) {
    NestingResult          out;
    std::vector<int>       order(sizes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return sizes[a] > sizes[b]; });

    for (int item : order) {
        const double s = sizes[item];
        int          placed = -1;
        for (std::size_t b = 0; b < out.loads.size(); ++b)
            if (out.loads[b] + s <= capacity + 1e-12) { placed = static_cast<int>(b); break; }
        if (placed < 0) {
            placed = static_cast<int>(out.loads.size());
            out.loads.push_back(0.0);
            out.assignment.emplace_back();
        }
        out.loads[placed] += s;
        out.assignment[placed].push_back(item);
    }
    out.bins = static_cast<int>(out.loads.size());
    return out;
}

// A trivial lower bound on the number of bins: total size divided by capacity,
// rounded up. The optimum is at least this many bins.
inline int nesting_lower_bound(const std::vector<double>& sizes, double capacity) {
    double total = 0;
    for (double s : sizes) total += s;
    return static_cast<int>(std::ceil(total / capacity - 1e-12));
}

} // namespace datamunge::algorithms
