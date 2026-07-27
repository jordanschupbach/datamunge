#include <datamunge/algorithms/prufer.hpp>

#include <set>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

std::vector<std::size_t> tree_to_prufer(std::size_t n,
                                        const std::vector<std::pair<std::size_t, std::size_t>>& tree_edges) {
    if (n < 2)
        throw std::invalid_argument("tree_to_prufer: n must be at least 2");
    if (tree_edges.size() != n - 1)
        throw std::invalid_argument("tree_to_prufer: a tree on n vertices must have exactly n-1 edges");

    std::vector<std::vector<std::size_t>> adj(n);
    std::vector<std::size_t> degree(n, 0);
    for (const auto& [u, v] : tree_edges) {
        if (u >= n || v >= n)
            throw std::invalid_argument("tree_to_prufer: edge endpoints must lie in [0, n)");
        adj[u].push_back(v);
        adj[v].push_back(u);
        ++degree[u];
        ++degree[v];
    }

    // The sequence has length n-2; for n == 2 it is empty.
    std::vector<std::size_t> sequence;
    if (n == 2) return sequence;
    sequence.reserve(n - 2);

    // Leaves held in an ordered set so begin() is always the smallest-labeled leaf.
    std::set<std::size_t> leaves;
    for (std::size_t v = 0; v < n; ++v)
        if (degree[v] == 1) leaves.insert(v);

    std::vector<char> removed(n, 0);
    for (std::size_t step = 0; step < n - 2; ++step) {
        const std::size_t leaf = *leaves.begin();
        leaves.erase(leaves.begin());
        removed[leaf] = 1;

        // A leaf has exactly one still-present neighbor; record it.
        std::size_t neighbor = leaf;
        for (const std::size_t w : adj[leaf])
            if (!removed[w]) { neighbor = w; break; }

        sequence.push_back(neighbor);
        if (--degree[neighbor] == 1) leaves.insert(neighbor);
    }
    return sequence;
}

std::vector<std::pair<std::size_t, std::size_t>> prufer_to_tree(const std::vector<std::size_t>& sequence) {
    const std::size_t n = sequence.size() + 2;

    // Each vertex starts with degree 1, plus one for every time it appears in the sequence.
    std::vector<std::size_t> degree(n, 1);
    for (const std::size_t s : sequence) {
        if (s >= n)
            throw std::invalid_argument("prufer_to_tree: sequence entries must lie in [0, n) with n = length + 2");
        ++degree[s];
    }

    std::set<std::size_t> leaves;
    for (std::size_t v = 0; v < n; ++v)
        if (degree[v] == 1) leaves.insert(v);

    std::vector<std::pair<std::size_t, std::size_t>> edges;
    edges.reserve(n - 1);
    for (const std::size_t s : sequence) {
        const std::size_t leaf = *leaves.begin(); // smallest available leaf
        leaves.erase(leaves.begin());
        edges.emplace_back(leaf, s);
        --degree[leaf];
        if (--degree[s] == 1) leaves.insert(s);
    }

    // Exactly two vertices of degree one remain; they form the last edge.
    const std::size_t u = *leaves.begin();
    leaves.erase(leaves.begin());
    const std::size_t v = *leaves.begin();
    edges.emplace_back(u, v);
    return edges;
}

} // namespace datamunge::algorithms
