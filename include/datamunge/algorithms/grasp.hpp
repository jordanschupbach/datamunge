#pragma once

// GRASP -- Greedy Randomized Adaptive Search Procedure (Feo & Resende, 1995): a
// multi-start metaheuristic for hard combinatorial optimization. Each iteration
// has two phases:
//
//   * Construction -- build a solution greedily, but at each step choose randomly
//     among the near-best candidates (a Restricted Candidate List controlled by a
//     parameter alpha), so different runs explore different solutions;
//   * Local search -- improve the constructed solution to a local optimum.
//
// Repeating this from many randomized starts and keeping the best found combines
// greedy quality with the diversity of randomization. Here GRASP is applied to
// the MAX-CUT problem: partition the vertices of a weighted graph into two sets
// to maximize the total weight of edges crossing the cut.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

struct GraspMaxCutResult {
    double           value{0};  // total weight of the cut
    std::vector<int> side;      // side (0/1) of each vertex
};

namespace detail {
struct GraspRng {
    std::uint64_t s;
    explicit GraspRng(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    std::uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double        uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    std::size_t   below(std::size_t n) { return static_cast<std::size_t>(next() % n); }
};
} // namespace detail

// Maximize the cut of an n-vertex weighted graph (adjacency list `adj[v]` of
// (neighbor, weight) pairs, symmetric) using GRASP with `iterations` restarts and
// candidate-list parameter `alpha` in [0,1] (0 = greedy, 1 = random).
inline GraspMaxCutResult grasp_max_cut(int n, const std::vector<std::vector<std::pair<int, double>>>& adj,
                                       int iterations = 50, double alpha = 0.3, std::uint64_t seed = 1) {
    detail::GraspRng rng(seed);

    auto cut_value = [&](const std::vector<int>& side) {
        double v = 0;
        for (int u = 0; u < n; ++u)
            for (const auto& [w, wt] : adj[u])
                if (u < w && side[u] != side[w]) v += wt;
        return v;
    };

    GraspMaxCutResult best;
    best.value = -1e300;

    for (int iter = 0; iter < iterations; ++iter) {
        // --- Greedy randomized construction ---
        std::vector<int> side(n, -1);
        std::vector<char> assigned(n, 0);
        int              placed = 0;
        while (placed < n) {
            // For each unassigned vertex, the gain of its better side w.r.t. assigned neighbors.
            double best_gain = -1e300, worst_gain = 1e300;
            std::vector<std::pair<int, int>> cand; // (vertex, side)
            std::vector<double>              gains;
            for (int v = 0; v < n; ++v) {
                if (assigned[v]) continue;
                double g0 = 0, g1 = 0; // gain if v -> side 0 / side 1
                for (const auto& [u, wt] : adj[v])
                    if (assigned[u]) { if (side[u] == 1) g0 += wt; else g1 += wt; }
                const int    s = g0 >= g1 ? 0 : 1;
                const double g = g0 >= g1 ? g0 : g1;
                cand.emplace_back(v, s);
                gains.push_back(g);
                best_gain = std::max(best_gain, g);
                worst_gain = std::min(worst_gain, g);
            }
            const double thr = best_gain - alpha * (best_gain - worst_gain);
            std::vector<std::size_t> rcl;
            for (std::size_t i = 0; i < cand.size(); ++i)
                if (gains[i] >= thr - 1e-12) rcl.push_back(i);
            const std::size_t pick = rcl[rng.below(rcl.size())];
            side[cand[pick].first] = cand[pick].second;
            assigned[cand[pick].first] = 1;
            ++placed;
        }

        // --- Local search: flip any vertex that improves the cut ---
        bool improved = true;
        while (improved) {
            improved = false;
            for (int v = 0; v < n; ++v) {
                double delta = 0; // change in cut if v flips
                for (const auto& [u, wt] : adj[v]) delta += (side[u] != side[v]) ? -wt : wt;
                if (delta > 1e-12) { side[v] ^= 1; improved = true; }
            }
        }

        const double val = cut_value(side);
        if (val > best.value) { best.value = val; best.side = side; }
    }
    return best;
}

} // namespace datamunge::algorithms
