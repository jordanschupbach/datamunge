/// @file high_water_mark.cpp
/// @brief Minimax-path high-water mark and the flood regions around it.

#include <datamunge/algorithms/high_water_mark.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {

HighWaterMark high_water_mark(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                              std::size_t start, std::size_t goal, const std::vector<double>& heuristic,
                              bool directed) {
    if (start >= n || goal >= n) throw std::invalid_argument("high_water_mark: vertex out of range");
    if (heuristic.size() != n) throw std::invalid_argument("high_water_mark: one heuristic value per vertex");
    std::vector<std::vector<std::size_t>> adjacency(n);
    for (const auto& [u, v] : edges) {
        if (u >= n || v >= n) throw std::invalid_argument("high_water_mark: edge vertex out of range");
        adjacency[u].push_back(v);
        if (!directed) adjacency[v].push_back(u);
    }
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> level(n, inf);
    using Entry = std::pair<double, std::size_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
    level[start] = heuristic[start];
    open.emplace(level[start], start);
    while (!open.empty()) {
        const auto [l, u] = open.top();
        open.pop();
        if (l > level[u]) continue;
        if (u == goal) break;
        for (const std::size_t v : adjacency[u]) {
            const double cand = std::max(l, heuristic[v]);
            if (cand < level[v]) {
                level[v] = cand;
                open.emplace(cand, v);
            }
        }
    }
    HighWaterMark out;
    out.level = level[goal];
    if (out.level == inf) return out;
    const auto flood = [&](bool strict) {
        const auto inside = [&](std::size_t v) { return strict ? heuristic[v] < out.level : heuristic[v] <= out.level; };
        std::vector<char> seen(n, 0);
        std::vector<std::size_t> stack;
        std::size_t count = 0;
        if (inside(start)) {
            seen[start] = 1;
            stack.push_back(start);
        }
        while (!stack.empty()) {
            const std::size_t u = stack.back();
            stack.pop_back();
            ++count;
            for (const std::size_t v : adjacency[u])
                if (!seen[v] && inside(v)) {
                    seen[v] = 1;
                    stack.push_back(v);
                }
        }
        return count;
    };
    out.below = flood(true);
    out.at_most = flood(false);
    return out;
}

}  // namespace datamunge::algorithms
