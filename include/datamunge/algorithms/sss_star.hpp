#pragma once

// SSS* (Stockman, 1979): a best-first game-tree search that returns the same
// minimax value as alpha-beta while never expanding a node alpha-beta would
// prune -- it dominates alpha-beta in nodes examined. Rather than the original
// OPEN-list formulation, this uses Plaat's equivalent and far simpler
// reformulation: SSS* = MTD(+INF), a sequence of *null-window* alpha-beta
// searches driven downward from +infinity, each reusing bounds stored in a
// transposition table (alpha-beta with memory). Successive searches tighten the
// bound until it converges to the game value.
//
// Leaf values are assumed to differ by at least `resolution` (1 by default, i.e.
// integer-valued leaves), which sets the null-window width.

#include <datamunge/algorithms/game_search.hpp> // GameNode, GameResult

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace datamunge::algorithms {

namespace detail {

struct SssBounds { double lower; double upper; };

struct SssState {
    std::unordered_map<const GameNode*, SssBounds> table;
    long long                                      leaves{0};
};

// Alpha-beta with a transposition table (Plaat's AlphaBetaWithMemory).
inline double sss_abwm(const GameNode& n, double alpha, double beta, bool maximizing, SssState& st) {
    auto it = st.table.find(&n);
    if (it != st.table.end()) {
        if (it->second.lower >= beta) return it->second.lower;
        if (it->second.upper <= alpha) return it->second.upper;
        alpha = std::max(alpha, it->second.lower);
        beta  = std::min(beta, it->second.upper);
    }

    double g;
    if (n.leaf) {
        ++st.leaves;
        g = n.value;
    } else if (maximizing) {
        g          = -std::numeric_limits<double>::infinity();
        double a   = alpha;
        for (const auto& c : n.children) {
            if (g >= beta) break;
            g = std::max(g, sss_abwm(c, a, beta, false, st));
            a = std::max(a, g);
        }
    } else {
        g          = std::numeric_limits<double>::infinity();
        double b   = beta;
        for (const auto& c : n.children) {
            if (g <= alpha) break;
            g = std::min(g, sss_abwm(c, alpha, b, true, st));
            b = std::min(b, g);
        }
    }

    // Store the derived bound.
    SssBounds bounds{-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    if (g <= alpha) bounds.upper = g;                 // failed low
    else if (g >= beta) bounds.lower = g;             // failed high
    else { bounds.lower = g; bounds.upper = g; }      // exact
    // Merge with any existing entry (keep the tightest known bounds).
    if (it != st.table.end()) {
        bounds.lower = std::max(bounds.lower, it->second.lower);
        bounds.upper = std::min(bounds.upper, it->second.upper);
        it->second   = bounds;
    } else {
        st.table.emplace(&n, bounds);
    }
    return g;
}

// Largest leaf value in the tree -- a finite starting upper bound for the driver.
inline double sss_max_leaf(const GameNode& n) {
    if (n.leaf) return n.value;
    double m = -std::numeric_limits<double>::infinity();
    for (const auto& c : n.children) m = std::max(m, sss_max_leaf(c));
    return m;
}

} // namespace detail

// SSS* search of the game tree rooted at `root`. Returns the minimax value and
// the number of leaf evaluations performed.
inline GameResult sss_star(const GameNode& root, bool maximizing, double resolution = 1.0) {
    detail::SssState st;
    // Seed the driver from a finite upper bound just above the true value
    // (MTD(f) = SSS* when started from above and descended).
    double           g          = detail::sss_max_leaf(root) + resolution;
    double           lowerbound = -std::numeric_limits<double>::infinity();
    double           upperbound = std::numeric_limits<double>::infinity();
    int              guard      = 0;
    do {
        const double beta = (g == lowerbound) ? g + resolution : g;
        g = detail::sss_abwm(root, beta - resolution, beta, maximizing, st);
        if (g < beta) upperbound = g;
        else lowerbound = g;
    } while (lowerbound < upperbound && ++guard < 100000);

    GameResult r;
    r.value            = g;
    r.leaves_evaluated = st.leaves;
    return r;
}

} // namespace datamunge::algorithms
