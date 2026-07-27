#include <datamunge/algorithms/topological_sort.hpp>

#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {

namespace {

// Validates that every edge endpoint names a vertex in [0, n); throws otherwise.
void validate_edges(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
    for (const auto& [u, v] : edges)
        if (u >= n || v >= n)
            throw std::invalid_argument("topological_sort: edge endpoint out of range [0, n)");
}

} // namespace

TopologicalSortResult topological_sort(std::size_t n,
                                       const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
    validate_edges(n, edges);

    // Build the successor adjacency list and count each vertex's incoming edges.
    std::vector<std::vector<std::size_t>> successors(n);
    std::vector<std::size_t> in_degree(n, 0);
    for (const auto& [u, v] : edges) {
        successors[u].push_back(v);
        ++in_degree[v];
    }

    // Seed the queue with every source (in-degree 0), in ascending index order for determinism.
    std::queue<std::size_t> ready;
    for (std::size_t u = 0; u < n; ++u)
        if (in_degree[u] == 0) ready.push(u);

    TopologicalSortResult r;
    r.order.reserve(n);
    while (!ready.empty()) {
        const std::size_t u = ready.front();
        ready.pop();
        r.order.push_back(u);
        // Remove u: each successor loses an incoming edge; those hitting zero become new sources.
        for (const std::size_t v : successors[u])
            if (--in_degree[v] == 0) ready.push(v);
    }

    // Emitting fewer than n vertices means the leftovers form (or feed into) a cycle: not a DAG.
    if (r.order.size() != n) {
        r.is_dag = false;
        r.order.clear();
    }
    return r;
}

bool is_topological_order(std::size_t n,
                          const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                          const std::vector<std::size_t>& order) {
    if (order.size() != n) return false;

    // position[v] = index of v within `order`; the sentinel n marks "not yet seen".
    std::vector<std::size_t> position(n, n);
    for (std::size_t p = 0; p < order.size(); ++p) {
        const std::size_t v = order[p];
        if (v >= n || position[v] != n) return false; // out of range, or not a permutation
        position[v] = p;
    }

    // Every edge u -> v must place u strictly before v.
    for (const auto& [u, v] : edges) {
        if (u >= n || v >= n) return false;
        if (position[u] >= position[v]) return false;
    }
    return true;
}

} // namespace datamunge::algorithms
