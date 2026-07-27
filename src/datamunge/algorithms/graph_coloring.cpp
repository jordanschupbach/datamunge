#include <datamunge/algorithms/graph_coloring.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace datamunge::algorithms {

namespace {

// Sentinel for an as-yet-uncolored vertex during the greedy sweep.
constexpr std::size_t kUncolored = static_cast<std::size_t>(-1);

// Builds the undirected adjacency list, validating endpoints and dropping self-loops.
std::vector<std::vector<std::size_t>> build_adjacency(
    std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
    std::vector<std::vector<std::size_t>> adj(n);
    for (const auto& [u, v] : edges) {
        if (u >= n || v >= n)
            throw std::invalid_argument("greedy_coloring: edge endpoint out of range (must be < n)");
        if (u == v) continue; // ignore self-loops: a vertex never conflicts with itself
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    return adj;
}

} // namespace

ColoringResult greedy_coloring(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
    const auto adj = build_adjacency(n, edges);

    // Degree of each vertex (parallel edges counted; harmless for the bound and the sweep).
    std::vector<std::size_t> degree(n);
    for (std::size_t v = 0; v < n; ++v) degree[v] = adj[v].size();

    // Welsh-Powell order: descending degree, ties broken by ascending index.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (degree[a] != degree[b]) return degree[a] > degree[b];
        return a < b;
    });

    ColoringResult result;
    result.color.assign(n, kUncolored);

    for (const std::size_t v : order) {
        // The smallest available color is at most degree[v], so degree[v] + 1 slots always suffice.
        std::vector<bool> used(degree[v] + 1, false);
        for (const std::size_t w : adj[v]) {
            const std::size_t cw = result.color[w];
            if (cw != kUncolored && cw < used.size()) used[cw] = true;
        }
        std::size_t c = 0;
        while (c < used.size() && used[c]) ++c;
        result.color[v] = c;
    }

    // Every vertex is colored, and the greedy uses colors 0..max contiguously (color k appears only
    // when colors 0..k-1 all block some vertex), so the distinct count is max(color) + 1.
    if (n == 0) {
        result.num_colors = 0;
    } else {
        result.num_colors = *std::max_element(result.color.begin(), result.color.end()) + 1;
    }
    return result;
}

bool is_proper_coloring(std::size_t n,
                        const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                        const std::vector<std::size_t>& color) {
    if (color.size() != n) return false;
    for (const auto& [u, v] : edges) {
        if (u >= n || v >= n) return false;
        if (u == v) continue; // self-loops are ignored, matching greedy_coloring
        if (color[u] == color[v]) return false;
    }
    return true;
}

} // namespace datamunge::algorithms
