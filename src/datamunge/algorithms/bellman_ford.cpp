/// @file bellman_ford.cpp
/// @brief Bellman-Ford in passes and in queue form, with negative-cycle extraction.

#include <datamunge/algorithms/bellman_ford.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace datamunge::algorithms {

/// @brief Argument checks and negative-cycle extraction.
namespace {

/// @brief Throws if the source or an edge endpoint is out of range.
void validate(std::size_t n, const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
              std::size_t source, const char* name) {
    if (source >= n) throw std::invalid_argument(std::string(name) + ": source vertex out of range");
    for (const auto& [u, v, w] : edges) {
        (void)w;
        if (u >= n || v >= n) throw std::invalid_argument(std::string(name) + ": edge endpoint out of range");
    }
}

/// @brief Follows predecessors back from @p v, a vertex downstream of a negative cycle, and
///        returns the cycle in edge order (empty if the pointers do not close a cycle).
std::vector<std::size_t> extract_cycle(const std::vector<std::size_t>& pred, std::size_t v) {
    const std::size_t n = pred.size();
    // n steps back from a vertex downstream of the cycle always land on the cycle.
    for (std::size_t k = 0; k < n && pred[v] != kNoPredecessor; ++k) v = pred[v];
    std::vector<std::size_t> cycle{v};
    std::size_t              u = pred[v];
    for (; u != v && u != kNoPredecessor && cycle.size() <= n; u = pred[u]) cycle.push_back(u);
    if (u != v) return {};  // the pointers do not close a cycle
    std::reverse(cycle.begin(), cycle.end());
    return cycle;
}

}  // namespace

BellmanFordResult bellman_ford(std::size_t n,
                               const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                               std::size_t source) {
    validate(n, edges, source, "bellman_ford");

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
        std::size_t relaxed = 0;
        for (const auto& [u, v, w] : edges) {
            ++r.edge_checks;
            if (r.distance[u] != kInfinity && r.distance[u] + w < r.distance[v]) {
                r.distance[v] = r.distance[u] + w;
                r.predecessor[v] = u;
                ++relaxed;
            }
        }
        ++r.passes;
        r.relaxations_per_pass.push_back(relaxed);
        if (relaxed == 0) break;
    }

    // One extra pass: any edge that can still be relaxed lies on (or downstream of) a
    // negative-weight cycle reachable from the source. The distance[u] != inf guard confines the
    // test to cycles the source can actually reach.
    for (const auto& [u, v, w] : edges) {
        if (r.distance[u] != kInfinity && r.distance[u] + w < r.distance[v]) {
            r.has_negative_cycle = true;
            r.predecessor[v] = u;
            r.negative_cycle = extract_cycle(r.predecessor, v);
            break;
        }
    }

    return r;
}

BellmanFordResult bellman_ford_queue(std::size_t n,
                                     const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                                     std::size_t source) {
    validate(n, edges, source, "bellman_ford_queue");
    constexpr double kInfinity = std::numeric_limits<double>::infinity();
    std::vector<std::vector<std::pair<std::size_t, double>>> out(n);
    for (const auto& [u, v, w] : edges) out[u].push_back({v, w});

    BellmanFordResult r;
    r.distance.assign(n, kInfinity);
    r.predecessor.assign(n, kNoPredecessor);
    r.distance[source] = 0.0;
    std::vector<std::size_t> depth(n, 0), current{source}, next;
    std::vector<char>        queued(n, 0);
    while (!current.empty()) {
        next.clear();
        std::size_t relaxed = 0;
        for (const std::size_t u : current) queued[u] = 0;
        for (const std::size_t u : current) {
            for (const auto& [v, w] : out[u]) {
                ++r.edge_checks;
                if (r.distance[u] + w < r.distance[v]) {
                    r.distance[v]    = r.distance[u] + w;
                    r.predecessor[v] = u;
                    depth[v]         = depth[u] + 1;
                    ++relaxed;
                    if (depth[v] >= n) {  // a path with n edges repeats a vertex: a negative cycle
                        r.has_negative_cycle = true;
                        r.negative_cycle     = extract_cycle(r.predecessor, v);
                        if (r.negative_cycle.empty()) r.negative_cycle = bellman_ford(n, edges, source).negative_cycle;
                        ++r.passes;
                        r.relaxations_per_pass.push_back(relaxed);
                        return r;
                    }
                    if (!queued[v]) {
                        queued[v] = 1;
                        next.push_back(v);
                    }
                }
            }
        }
        ++r.passes;
        r.relaxations_per_pass.push_back(relaxed);
        current.swap(next);
    }
    return r;
}

std::vector<std::size_t> shortest_path_to(const BellmanFordResult& result, std::size_t target) {
    const std::size_t n = result.distance.size();
    if (target >= n) throw std::invalid_argument("shortest_path_to: target out of range");
    std::vector<std::size_t> path;
    if (result.distance[target] == std::numeric_limits<double>::infinity()) return path;
    for (std::size_t v = target; v != kNoPredecessor; v = result.predecessor[v]) {
        path.push_back(v);
        if (path.size() > n) return {};  // the tree contains a cycle
    }
    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace datamunge::algorithms
