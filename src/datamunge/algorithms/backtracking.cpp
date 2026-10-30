/// @file backtracking.cpp
/// @brief Simple-path counting, Hamiltonian-path search and graph generators.

#include <datamunge/algorithms/backtracking.hpp>

#include <algorithm>
#include <stdexcept>

namespace datamunge::algorithms {

/// @brief Adjacency-list construction.
namespace {

/// @brief Sorted adjacency lists of the graph.
std::vector<std::vector<std::size_t>> adjacency(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                                bool directed) {
    std::vector<std::vector<std::size_t>> a(n);
    for (auto [u, v] : edges) {
        if (u >= n || v >= n) throw std::invalid_argument("backtracking: edge endpoint out of range");
        a[u].push_back(v);
        if (!directed) a[v].push_back(u);
    }
    for (auto& l : a) {
        std::sort(l.begin(), l.end());
        l.erase(std::unique(l.begin(), l.end()), l.end());
    }
    return a;
}

}  // namespace

std::vector<std::pair<std::size_t, std::size_t>> grid_graph_edges(std::size_t rows, std::size_t cols) {
    std::vector<std::pair<std::size_t, std::size_t>> e;
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) {
            if (c + 1 < cols) e.emplace_back(r * cols + c, r * cols + c + 1);
            if (r + 1 < rows) e.emplace_back(r * cols + c, (r + 1) * cols + c);
        }
    return e;
}

std::vector<std::pair<std::size_t, std::size_t>> knight_graph_edges(std::size_t rows, std::size_t cols) {
    std::vector<std::pair<std::size_t, std::size_t>> e;
    const int dr[4] = {1, 1, 2, 2}, dc[4] = {-2, 2, -1, 1};
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            for (int k = 0; k < 4; ++k) {
                const long rr = static_cast<long>(r) + dr[k], cc = static_cast<long>(c) + dc[k];
                if (rr < static_cast<long>(rows) && cc >= 0 && cc < static_cast<long>(cols))
                    e.emplace_back(r * cols + c, static_cast<std::size_t>(rr) * cols + static_cast<std::size_t>(cc));
            }
    return e;
}

std::vector<std::pair<std::size_t, std::size_t>> complete_graph_edges(std::size_t n) {
    std::vector<std::pair<std::size_t, std::size_t>> e;
    for (std::size_t u = 0; u < n; ++u)
        for (std::size_t v = u + 1; v < n; ++v) e.emplace_back(u, v);
    return e;
}

SimplePathCount count_simple_paths(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                   std::size_t start, std::size_t goal, bool directed, std::uint64_t node_limit) {
    if (start >= n || goal >= n) throw std::invalid_argument("count_simple_paths: vertex out of range");
    const auto        a = adjacency(n, edges, directed);
    std::vector<char> on(n, 0);
    SimplePathCount   out;
    auto dfs = [&](auto&& self, std::size_t u, std::size_t depth) -> void {
        if (node_limit && out.nodes >= node_limit) return;
        ++out.nodes;
        out.max_depth = std::max(out.max_depth, depth);
        if (u == goal) {
            ++out.paths;
            return;
        }
        on[u]         = 1;
        bool extended = false;
        for (std::size_t v : a[u])
            if (!on[v]) {
                extended = true;
                self(self, v, depth + 1);
            }
        if (!extended) ++out.dead_ends;
        on[u] = 0;
    };
    dfs(dfs, start, 0);
    return out;
}

HamiltonianSearch hamiltonian_path(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                   std::size_t start, BranchOrder order, bool prune, std::uint64_t node_limit) {
    if (start >= n) throw std::invalid_argument("hamiltonian_path: start out of range");
    const auto        a = adjacency(n, edges, false);
    std::vector<char> on(n, 0);
    std::vector<std::size_t> free_degree(n);
    for (std::size_t v = 0; v < n; ++v) free_degree[v] = a[v].size();
    HamiltonianSearch out;
    std::vector<char> seen(n);
    std::vector<std::size_t> stack;

    // True when every unvisited vertex is reachable from u through unvisited vertices.
    auto completable = [&](std::size_t u, std::size_t remaining) {
        std::fill(seen.begin(), seen.end(), 0);
        stack.assign(1, u);
        seen[u] = 1;
        std::size_t reached = 0;
        while (!stack.empty()) {
            const std::size_t x = stack.back();
            stack.pop_back();
            for (std::size_t y : a[x])
                if (!on[y] && !seen[y]) {
                    seen[y] = 1;
                    ++reached;
                    stack.push_back(y);
                }
        }
        return reached == remaining;
    };

    auto visit = [&](std::size_t v) {
        on[v] = 1;
        for (std::size_t w : a[v]) --free_degree[w];
        out.path.push_back(v);
    };
    auto unvisit = [&](std::size_t v) {
        on[v] = 0;
        for (std::size_t w : a[v]) ++free_degree[w];
        out.path.pop_back();
        ++out.backtracks;
    };

    auto dfs = [&](auto&& self, std::size_t u) -> bool {
        if (node_limit && out.nodes >= node_limit) {
            out.hit_limit = true;
            return false;
        }
        ++out.nodes;
        if (out.path.size() == n) return true;
        if (prune && !completable(u, n - out.path.size())) {
            ++out.pruned;
            return false;
        }
        std::vector<std::size_t> next;
        for (std::size_t v : a[u])
            if (!on[v]) next.push_back(v);
        if (order == BranchOrder::FewestOnward)
            std::stable_sort(next.begin(), next.end(),
                             [&](std::size_t x, std::size_t y) { return free_degree[x] < free_degree[y]; });
        for (std::size_t v : next) {
            visit(v);
            if (self(self, v)) return true;
            unvisit(v);
            if (out.hit_limit) return false;
        }
        return false;
    };
    visit(start);
    out.found = dfs(dfs, start);
    if (!out.found) out.path.clear();
    return out;
}

}  // namespace datamunge::algorithms
