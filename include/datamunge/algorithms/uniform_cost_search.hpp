#pragma once

// Uniform-cost search: find the least-cost path from a start to a goal in a graph
// with non-negative edge weights by always expanding the frontier node of least
// accumulated cost. It is Dijkstra's algorithm run as a goal-directed search --
// it stops as soon as the goal is dequeued, so it explores only the cost sphere
// up to the goal rather than the whole graph.

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

struct UniformCostResult {
    double           cost{0};
    std::vector<int> path;
    bool             found{false};
    int              expanded{0}; // number of nodes dequeued/expanded
};

// `adj[u]` lists (neighbour, edge weight >= 0). Finds start -> goal.
inline UniformCostResult uniform_cost_search(int n, const std::vector<std::vector<std::pair<int, double>>>& adj,
                                             int start, int goal) {
    const double            inf = 1e300;
    std::vector<double>     dist(n, inf);
    std::vector<int>        parent(n, -1);
    std::vector<char>       done(n, 0);
    using Item = std::pair<double, int>; // (cost, node)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;

    dist[start] = 0.0;
    pq.emplace(0.0, start);
    UniformCostResult result;
    while (!pq.empty()) {
        const auto [c, u] = pq.top();
        pq.pop();
        if (done[u]) continue;
        done[u] = 1;
        ++result.expanded;
        if (u == goal) {
            result.found = true;
            result.cost  = c;
            for (int v = goal; v != -1; v = parent[v]) result.path.push_back(v);
            std::reverse(result.path.begin(), result.path.end());
            return result;
        }
        for (const auto& [v, w] : adj[u]) {
            if (!done[v] && c + w < dist[v]) {
                dist[v]   = c + w;
                parent[v] = u;
                pq.emplace(dist[v], v);
            }
        }
    }
    return result; // not found
}

} // namespace datamunge::algorithms
