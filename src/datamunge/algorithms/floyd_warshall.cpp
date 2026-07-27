#include <datamunge/algorithms/floyd_warshall.hpp>

#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {

AllPairsShortestPaths floyd_warshall(std::size_t n,
                                     const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges) {
    constexpr double inf = std::numeric_limits<double>::infinity();

    AllPairsShortestPaths r;
    r.distance.assign(n, std::vector<double>(n, inf));
    r.next.assign(n, std::vector<std::size_t>(n, kNoVertex));
    for (std::size_t i = 0; i < n; ++i) r.distance[i][i] = 0.0; // a vertex reaches itself at cost 0

    // Seed the matrices from the edge list, validating endpoints and keeping the lightest parallel edge.
    for (const auto& [u, v, w] : edges) {
        if (u >= n || v >= n)
            throw std::invalid_argument("floyd_warshall: edge endpoint out of range");
        if (w < r.distance[u][v]) {
            r.distance[u][v] = w;
            r.next[u][v] = v;
        }
    }

    // dist^{(k)}[i][j] = min(dist^{(k-1)}[i][j], dist^{(k-1)}[i][k] + dist^{(k-1)}[k][j]), in place.
    for (std::size_t k = 0; k < n; ++k)
        for (std::size_t i = 0; i < n; ++i) {
            if (r.distance[i][k] == inf) continue; // guard against +infinity arithmetic
            for (std::size_t j = 0; j < n; ++j) {
                if (r.distance[k][j] == inf) continue;
                const double candidate = r.distance[i][k] + r.distance[k][j];
                if (candidate < r.distance[i][j]) {
                    r.distance[i][j] = candidate;
                    r.next[i][j] = r.next[i][k];
                }
            }
        }

    // A negative-weight cycle exists iff some vertex can return to itself at negative cost.
    for (std::size_t i = 0; i < n; ++i)
        if (r.distance[i][i] < 0.0) {
            r.has_negative_cycle = true;
            break;
        }

    return r;
}

std::vector<std::size_t> reconstruct_path(const AllPairsShortestPaths& apsp, std::size_t i, std::size_t j) {
    const std::size_t n = apsp.distance.size();
    if (i >= n || j >= n)
        throw std::invalid_argument("reconstruct_path: endpoint out of range");
    if (i == j) return {i};
    if (apsp.next[i][j] == kNoVertex) return {}; // j is unreachable from i

    std::vector<std::size_t> path{i};
    std::size_t current = i;
    while (current != j) {
        current = apsp.next[current][j];
        if (current == kNoVertex) return {}; // defensive: a broken next-chain means no path
        path.push_back(current);
    }
    return path;
}

} // namespace datamunge::algorithms
