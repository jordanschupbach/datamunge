#pragma once

/// \file signed_reversals.hpp
/// \brief Sorting a signed permutation by reversals (genome rearrangement distance).
///
/// Genomes of related species often differ by *reversals*: a chromosomal segment is excised,
/// flipped, and reinserted, which reverses the gene order *and* the strand (sign) of each gene.
/// The *reversal distance* between two genomes -- the minimum number of such operations to
/// transform one into the other -- measures their evolutionary divergence. Modeling genes as a
/// *signed permutation*, this module computes the exact reversal distance (and an optimal
/// sequence of reversals) by breadth-first search over permutations, and counts *breakpoints*,
/// the classic lower bound.

#include <algorithm>
#include <cstddef>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// Result of sorting a signed permutation by reversals.
struct SignedReversalResult {
    int                                 distance;   ///< Minimum number of reversals.
    std::vector<std::pair<int, int>>    reversals;  ///< An optimal sequence of [i,j] reversals.
};

namespace detail {
inline std::string encode_perm(const std::vector<int>& p) {
    std::string s;
    for (int x : p) { s += (x < 0 ? '-' : '+'); s += static_cast<char>('a' + (x < 0 ? -x : x)); }
    return s;
}
/// Reverse p[i..j] inclusive and negate each element (a signed reversal).
inline std::vector<int> apply_reversal(std::vector<int> p, int i, int j) {
    while (i < j) { std::swap(p[i], p[j]); p[i] = -p[i]; p[j] = -p[j]; ++i; --j; }
    if (i == j) p[i] = -p[i];
    return p;
}
}  // namespace detail

/// \brief Number of *breakpoints* in a signed permutation (a lower bound on reversal distance).
///
/// Frame the permutation with 0 at the front and n+1 at the back; a *breakpoint* is an adjacent
/// pair \((a,b)\) that is not "consecutive" (i.e. \(b\neq a+1\)).
inline int signed_breakpoints(const std::vector<int>& perm) {
    std::vector<int> framed;
    framed.push_back(0);
    for (int x : perm) framed.push_back(x);
    framed.push_back(static_cast<int>(perm.size()) + 1);
    int bp = 0;
    for (std::size_t i = 0; i + 1 < framed.size(); ++i)
        if (framed[i + 1] != framed[i] + 1) ++bp;
    return bp;
}

/// \brief Exact reversal distance and an optimal reversal sequence, by BFS (for small n).
///
/// \param start  a signed permutation of 1..n (values +/-1..+/-n); the target is the identity
///               (+1,+2,...,+n).
inline SignedReversalResult sort_by_reversals(const std::vector<int>& start) {
    const int        n = static_cast<int>(start.size());
    std::vector<int> identity(n);
    for (int i = 0; i < n; ++i) identity[i] = i + 1;

    if (start == identity) return {0, {}};

    // BFS storing, per state, its predecessor and the reversal that produced it.
    std::unordered_map<std::string, std::pair<std::string, std::pair<int, int>>> parent;
    std::queue<std::vector<int>>                                                 q;
    q.push(start);
    parent[detail::encode_perm(start)] = {"", {-1, -1}};

    while (!q.empty()) {
        std::vector<int> cur = q.front();
        q.pop();
        std::string curkey = detail::encode_perm(cur);
        for (int i = 0; i < n; ++i)
            for (int j = i; j < n; ++j) {
                std::vector<int> nxt    = detail::apply_reversal(cur, i, j);
                std::string      nxtkey = detail::encode_perm(nxt);
                if (parent.count(nxtkey)) continue;
                parent[nxtkey] = {curkey, {i, j}};
                if (nxt == identity) {
                    // Reconstruct the reversal sequence.
                    SignedReversalResult r;
                    std::string          k = nxtkey;
                    while (parent[k].first != "") {
                        r.reversals.push_back(parent[k].second);
                        k = parent[k].first;
                    }
                    std::reverse(r.reversals.begin(), r.reversals.end());
                    r.distance = static_cast<int>(r.reversals.size());
                    return r;
                }
                q.push(std::move(nxt));
            }
    }
    return {-1, {}};   // unreachable for a valid permutation
}

}  // namespace datamunge::algorithms
