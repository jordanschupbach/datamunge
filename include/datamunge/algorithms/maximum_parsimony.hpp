#pragma once

/// \file maximum_parsimony.hpp
/// \brief Maximum parsimony via Fitch's algorithm (small-parsimony scoring).
///
/// *Maximum parsimony* explains observed character data (e.g. DNA columns across species) by
/// the phylogenetic tree requiring the *fewest* evolutionary changes. The *small-parsimony*
/// subproblem -- scoring a *given* tree -- is solved exactly by Fitch's algorithm (1971) in one
/// post-order pass: each internal node's candidate state set is the *intersection* of its
/// children's sets, or their *union* (counting one change) if the intersection is empty. The
/// number of unions is the minimum number of changes for that character. Summed over all
/// characters, it is the tree's parsimony score. This module implements Fitch scoring with
/// bitset state sets.

#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A rooted binary tree: children[node] = {left, right}, or {-1,-1} for a leaf.
struct ParsimonyTree {
    int                              root;
    std::vector<std::pair<int, int>> children;   ///< Indexed by node id.
};

namespace detail {
/// Fitch post-order: returns the node's state set (bitmask); increments \c score on a union.
inline std::uint32_t fitch_recurse(const ParsimonyTree& t, int node,
                                   const std::vector<int>& leaf_state, int& score) {
    auto [l, r] = t.children[node];
    if (l < 0) return std::uint32_t{1} << leaf_state[node];   // leaf: singleton set
    std::uint32_t a     = fitch_recurse(t, l, leaf_state, score);
    std::uint32_t b     = fitch_recurse(t, r, leaf_state, score);
    std::uint32_t inter = a & b;
    if (inter) return inter;                                  // agreement: no change
    ++score;                                                  // conflict -> one change
    return a | b;
}
}  // namespace detail

/// \brief Minimum number of changes for one character on the given tree (Fitch's algorithm).
///
/// \param leaf_state  state (0..num_states-1) of each leaf node; ignored for internal nodes.
inline int fitch_small_parsimony(const ParsimonyTree& t, const std::vector<int>& leaf_state) {
    int score = 0;
    detail::fitch_recurse(t, t.root, leaf_state, score);
    return score;
}

/// \brief Parsimony score of a tree over a character matrix (sum of Fitch scores per column).
///
/// \param leaf_states  leaf_states[c] gives every node's state for character c (leaves used).
inline int parsimony_score(const ParsimonyTree& t, const std::vector<std::vector<int>>& leaf_states) {
    int total = 0;
    for (const auto& col : leaf_states) total += fitch_small_parsimony(t, col);
    return total;
}

}  // namespace datamunge::algorithms
