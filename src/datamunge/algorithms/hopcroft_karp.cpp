#include <datamunge/algorithms/hopcroft_karp.hpp>

#include <limits>
#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {

namespace {

// Sentinel "infinite" BFS distance, distinct from the unmatched sentinel.
constexpr std::size_t kInf = std::numeric_limits<std::size_t>::max();

} // namespace

BipartiteMatchingResult hopcroft_karp(std::size_t n_left, std::size_t n_right,
                                      const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
    // Build left-side adjacency, validating endpoints as we go.
    std::vector<std::vector<std::size_t>> adj(n_left);
    for (const auto& [l, r] : edges) {
        if (l >= n_left || r >= n_right)
            throw std::invalid_argument("hopcroft_karp: edge endpoint out of range");
        adj[l].push_back(r);
    }

    BipartiteMatchingResult result;
    result.match_left.assign(n_left, kBipartiteUnmatched);
    result.match_right.assign(n_right, kBipartiteUnmatched);

    std::vector<std::size_t>& match_left = result.match_left;
    std::vector<std::size_t>& match_right = result.match_right;
    std::vector<std::size_t> dist(n_left); // BFS layer of each left vertex within a phase

    // One phase's BFS: layer the left vertices by the length of the shortest alternating path from a
    // free left vertex, and report whether any augmenting path (reaching a free right vertex) exists.
    auto bfs = [&]() {
        std::queue<std::size_t> q;
        for (std::size_t u = 0; u < n_left; ++u) {
            if (match_left[u] == kBipartiteUnmatched) {
                dist[u] = 0;
                q.push(u);
            } else {
                dist[u] = kInf;
            }
        }
        bool found_augmenting = false;
        while (!q.empty()) {
            const std::size_t u = q.front();
            q.pop();
            for (const std::size_t v : adj[u]) {
                const std::size_t w = match_right[v]; // left vertex currently holding right vertex v
                if (w == kBipartiteUnmatched) {
                    found_augmenting = true; // v is free: an augmenting path ends here
                } else if (dist[w] == kInf) {
                    dist[w] = dist[u] + 1;
                    q.push(w);
                }
            }
        }
        return found_augmenting;
    };

    // One phase's DFS from free left vertex u: try to extend an augmenting path that strictly
    // follows the BFS layering. On success it flips the matching along the path and returns true.
    auto dfs = [&](auto&& self, std::size_t u) -> bool {
        for (const std::size_t v : adj[u]) {
            const std::size_t w = match_right[v];
            if (w == kBipartiteUnmatched || (dist[w] == dist[u] + 1 && self(self, w))) {
                match_right[v] = u;
                match_left[u] = v;
                return true;
            }
        }
        dist[u] = kInf; // dead end: never revisit u this phase
        return false;
    };

    while (bfs()) {
        for (std::size_t u = 0; u < n_left; ++u) {
            if (match_left[u] == kBipartiteUnmatched && dfs(dfs, u))
                ++result.size;
        }
    }
    return result;
}

} // namespace datamunge::algorithms
