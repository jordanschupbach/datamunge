#include <datamunge/algorithms/dijkstra.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {

ShortestPathResult dijkstra(std::size_t n,
                            const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                            std::size_t source) {
    if (source >= n)
        throw std::invalid_argument("dijkstra: source vertex out of range");

    // Build a forward adjacency list, validating endpoints and rejecting negative weights.
    std::vector<std::vector<std::pair<std::size_t, double>>> adjacency(n);
    for (const auto& [from, to, weight] : edges) {
        if (from >= n || to >= n)
            throw std::invalid_argument("dijkstra: edge endpoint out of range");
        if (weight < 0.0)
            throw std::invalid_argument("dijkstra: negative edge weight (Dijkstra requires non-negative weights)");
        adjacency[from].emplace_back(to, weight);
    }

    ShortestPathResult r;
    r.distance.assign(n, std::numeric_limits<double>::infinity());
    r.predecessor.assign(n, kNoPredecessor);
    r.distance[source] = 0.0;

    // Min-heap of (tentative distance, vertex), ordered by smallest distance first. std::greater
    // turns the default max-heap into a min-heap.
    using Entry = std::pair<double, std::size_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    heap.emplace(0.0, source);

    while (!heap.empty()) {
        const auto [d, u] = heap.top();
        heap.pop();

        // Lazy deletion: skip stale heap entries left over from a since-improved distance.
        if (d > r.distance[u]) continue;

        // Relax every outgoing edge of the just-settled vertex u.
        for (const auto& [v, w] : adjacency[u]) {
            const double candidate = d + w;
            if (candidate < r.distance[v]) {
                r.distance[v] = candidate;
                r.predecessor[v] = u;
                heap.emplace(candidate, v);
            }
        }
    }
    return r;
}

std::vector<std::size_t> reconstruct_path(const ShortestPathResult& result, std::size_t target) {
    if (target >= result.distance.size() || result.distance[target] == std::numeric_limits<double>::infinity())
        return {}; // out of range or unreachable

    std::vector<std::size_t> path;
    for (std::size_t v = target; v != kNoPredecessor; v = result.predecessor[v])
        path.push_back(v);
    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace datamunge::algorithms
