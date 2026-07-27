#include <datamunge/algorithms/bellman_ford.hpp>

#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {

BellmanFordResult bellman_ford(std::size_t n,
                               const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                               std::size_t source) {
    if (source >= n)
        throw std::invalid_argument("bellman_ford: source vertex out of range");
    for (const auto& [u, v, w] : edges) {
        (void)w;
        if (u >= n || v >= n)
            throw std::invalid_argument("bellman_ford: edge endpoint out of range");
    }

    constexpr double kInfinity = std::numeric_limits<double>::infinity();

    BellmanFordResult r;
    r.distance.assign(n, kInfinity);
    r.predecessor.assign(n, kNoPredecessor);
    r.has_negative_cycle = false;
    r.distance[source] = 0.0;

    // Relax every edge n-1 times. A shortest path free of negative cycles has at most n-1 edges,
    // so after n-1 rounds every distance has converged. The early exit stops as soon as a full
    // pass changes nothing (which can only happen when there is no reachable negative cycle).
    for (std::size_t iter = 1; iter < n; ++iter) {
        bool relaxed = false;
        for (const auto& [u, v, w] : edges) {
            if (r.distance[u] != kInfinity && r.distance[u] + w < r.distance[v]) {
                r.distance[v] = r.distance[u] + w;
                r.predecessor[v] = u;
                relaxed = true;
            }
        }
        if (!relaxed) break;
    }

    // One extra pass: any edge that can still be relaxed lies on (or downstream of) a
    // negative-weight cycle reachable from the source. The distance[u] != inf guard confines the
    // test to cycles the source can actually reach.
    for (const auto& [u, v, w] : edges) {
        if (r.distance[u] != kInfinity && r.distance[u] + w < r.distance[v]) {
            r.has_negative_cycle = true;
            break;
        }
    }

    return r;
}

} // namespace datamunge::algorithms
