#include <datamunge/algorithms/prim.hpp>

#include <functional>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

MinimumSpanningTree prim(std::size_t n,
                         const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                         std::size_t start) {
    if (start >= n)
        throw std::invalid_argument("prim: start vertex out of range");

    // Build an undirected adjacency list: every edge {u, v, w} is stored on both endpoints.
    std::vector<std::vector<std::pair<std::size_t, double>>> adjacency(n);
    for (const auto& [u, v, w] : edges) {
        if (u >= n || v >= n)
            throw std::invalid_argument("prim: edge endpoint out of range");
        adjacency[u].emplace_back(v, w);
        adjacency[v].emplace_back(u, w);
    }

    MinimumSpanningTree result;
    std::vector<char> in_tree(n, 0);

    // Lazy min-heap of candidate crossing edges: (weight, in-tree endpoint, outside endpoint).
    // std::greater orders the tuples lexicographically, smallest weight first -- turning the
    // default max-heap into a min-heap keyed by weight.
    using Candidate = std::tuple<double, std::size_t, std::size_t>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> frontier;

    // Seed the frontier with the start vertex's edges.
    in_tree[start] = 1;
    std::size_t reached = 1;
    for (const auto& [v, w] : adjacency[start])
        frontier.emplace(w, start, v);

    while (!frontier.empty()) {
        const auto [w, u, v] = frontier.top();
        frontier.pop();

        // Lazy deletion: v may already have joined the tree via a cheaper edge; skip stale entries.
        if (in_tree[v]) continue;

        // (u, v) is the cheapest edge crossing the cut between the tree and the rest -- safe to add.
        in_tree[v] = 1;
        ++reached;
        result.edges.emplace_back(u, v, w);
        result.total_weight += w;

        // v is now in the tree: offer its edges to still-outside vertices as new crossing candidates.
        for (const auto& [x, wx] : adjacency[v])
            if (!in_tree[x])
                frontier.emplace(wx, v, x);
    }

    result.is_connected = (reached == n);
    return result;
}

} // namespace datamunge::algorithms
