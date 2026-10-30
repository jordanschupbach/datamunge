/// @file bidirectional_search.cpp
/// @brief Bidirectional BFS and Dijkstra with one-directional baselines.

#include <datamunge/algorithms/bidirectional_search.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {

/// @brief Helpers for bidirectional_search.cpp.
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();  ///< Distance of an unreached vertex.
constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();  ///< Missing parent or depth.

using Adjacency = std::vector<std::vector<std::pair<std::size_t, double>>>;  ///< (neighbour, weight) lists.

/// Throws unless both terminals are vertices.
void check_terminals(std::size_t n, std::size_t s, std::size_t t) {
    if (s >= n || t >= n) throw std::invalid_argument("pair search: start or goal out of range");
}

/// Forward and reverse adjacency lists of an unweighted edge list (unit weights).
std::pair<Adjacency, Adjacency> unweighted(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                           bool directed) {
    Adjacency f(n), b(n);
    for (const auto& [u, v] : edges) {
        if (u >= n || v >= n) throw std::invalid_argument("pair search: endpoint out of range");
        f[u].emplace_back(v, 1.0);
        b[v].emplace_back(u, 1.0);
        if (!directed) {
            f[v].emplace_back(u, 1.0);
            b[u].emplace_back(v, 1.0);
        }
    }
    return {std::move(f), std::move(b)};
}

/// Forward and reverse adjacency lists of a weighted directed edge list.
std::pair<Adjacency, Adjacency> weighted(std::size_t n, const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges) {
    Adjacency f(n), b(n);
    for (const auto& [u, v, w] : edges) {
        if (u >= n || v >= n) throw std::invalid_argument("pair search: endpoint out of range");
        if (!(w >= 0)) throw std::invalid_argument("pair search: weights must be non-negative");
        f[u].emplace_back(v, w);
        b[v].emplace_back(u, w);
    }
    return {std::move(f), std::move(b)};
}

/// Path start -> meet from forward parents, then meet -> goal from backward parents.
std::vector<std::size_t> join(std::size_t meet, const std::vector<std::size_t>& forward_parent,
                              const std::vector<std::size_t>& backward_parent) {
    std::vector<std::size_t> path;
    for (std::size_t v = meet; v != kNone; v = forward_parent[v]) path.push_back(v);
    std::reverse(path.begin(), path.end());
    for (std::size_t v = backward_parent[meet]; v != kNone; v = backward_parent[v]) path.push_back(v);
    return path;
}

/// Fills a result for an unreachable goal.
PairSearchResult not_found(PairSearchResult r) {
    r.found = false;
    r.distance = kInf;
    r.path.clear();
    return r;
}

/// Dijkstra (or BFS with unit weights) from s that stops when t is settled.
PairSearchResult one_directional(const Adjacency& adj, std::size_t s, std::size_t t) {
    const std::size_t n = adj.size();
    std::vector<double> dist(n, kInf);
    std::vector<std::size_t> parent(n, kNone);
    std::vector<bool> done(n, false);
    using Item = std::pair<double, std::size_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<>> heap;
    dist[s] = 0;
    heap.emplace(0.0, s);
    PairSearchResult r;
    while (!heap.empty()) {
        const auto [d, u] = heap.top();
        heap.pop();
        if (done[u]) continue;
        done[u] = true;
        ++r.forward_expanded;
        if (u == t) break;
        for (const auto& [v, w] : adj[u]) {
            ++r.edges_scanned;
            if (d + w < dist[v]) dist[v] = d + w, parent[v] = u, heap.emplace(dist[v], v);
        }
    }
    if (!done[t]) return not_found(r);
    r.found = true;
    r.distance = dist[t];
    r.meeting_vertex = t;
    for (std::size_t v = t; v != kNone; v = parent[v]) r.path.push_back(v);
    std::reverse(r.path.begin(), r.path.end());
    return r;
}

} // namespace

/// Plain FIFO breadth-first search that stops when the goal is dequeued.
PairSearchResult bfs_pair_search(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                 std::size_t start, std::size_t goal, bool directed) {
    check_terminals(n, start, goal);
    const auto adj = unweighted(n, edges, directed).first;
    // A plain FIFO queue: with unit weights it settles vertices in distance order.
    std::vector<std::size_t> dist(n, kNone), parent(n, kNone);
    std::queue<std::size_t> queue;
    dist[start] = 0;
    queue.push(start);
    PairSearchResult r;
    while (!queue.empty()) {
        const std::size_t u = queue.front();
        queue.pop();
        ++r.forward_expanded;
        if (u == goal) break;
        for (const auto& [v, w] : adj[u]) {
            ++r.edges_scanned;
            if (dist[v] == kNone) dist[v] = dist[u] + 1, parent[v] = u, queue.push(v);
        }
    }
    if (dist[goal] == kNone) return not_found(r);
    r.found = true;
    r.distance = static_cast<double>(dist[goal]);
    r.meeting_vertex = goal;
    for (std::size_t v = goal; v != kNone; v = parent[v]) r.path.push_back(v);
    std::reverse(r.path.begin(), r.path.end());
    return r;
}

/// Layer-at-a-time expansion of the smaller frontier, stopping after the first layer that meets.
PairSearchResult bidirectional_bfs(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                   std::size_t start, std::size_t goal, bool directed) {
    check_terminals(n, start, goal);
    PairSearchResult r;
    if (start == goal) {
        r.found = true;
        r.path = {start};
        r.meeting_vertex = start;
        return r;
    }
    const auto [fwd, bwd] = unweighted(n, edges, directed);
    std::vector<std::size_t> dist_f(n, kNone), dist_b(n, kNone), parent_f(n, kNone), parent_b(n, kNone);
    dist_f[start] = 0;
    dist_b[goal] = 0;
    std::vector<std::size_t> frontier_f{start}, frontier_b{goal};
    std::size_t best = kNone, meet = kNone;
    while (!frontier_f.empty() && !frontier_b.empty() && best == kNone) {
        const bool forward = frontier_f.size() <= frontier_b.size();
        auto& frontier = forward ? frontier_f : frontier_b;
        auto& dist = forward ? dist_f : dist_b;
        auto& other = forward ? dist_b : dist_f;
        auto& parent = forward ? parent_f : parent_b;
        const auto& adj = forward ? fwd : bwd;
        std::vector<std::size_t> next;
        for (const std::size_t u : frontier) {
            (forward ? r.forward_expanded : r.backward_expanded) += 1;
            for (const auto& [v, w] : adj[u]) {
                ++r.edges_scanned;
                if (dist[v] == kNone) dist[v] = dist[u] + 1, parent[v] = u, next.push_back(v);
                if (other[v] != kNone && dist[u] + 1 + other[v] < best) {
                    best = dist[u] + 1 + other[v];
                    // The joining edge is u -> v (forward) or v -> u (backward); meet at v with
                    // u as its parent on this side.
                    meet = v;
                    parent[v] = u;
                }
            }
        }
        frontier.swap(next);
    }
    if (best == kNone) return not_found(r);
    r.found = true;
    r.distance = static_cast<double>(best);
    r.meeting_vertex = meet;
    r.path = join(meet, parent_f, parent_b);
    return r;
}

PairSearchResult dijkstra_pair_search(std::size_t n, const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                                      std::size_t start, std::size_t goal) {
    check_terminals(n, start, goal);
    return one_directional(weighted(n, edges).first, start, goal);
}

/// Alternates the side with the smaller heap top and stops when the top keys sum to at least mu.
PairSearchResult bidirectional_dijkstra(std::size_t n, const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                                        std::size_t start, std::size_t goal) {
    check_terminals(n, start, goal);
    const auto [fwd, bwd] = weighted(n, edges);
    std::vector<double> dist_f(n, kInf), dist_b(n, kInf);
    std::vector<std::size_t> parent_f(n, kNone), parent_b(n, kNone);
    std::vector<bool> done_f(n, false), done_b(n, false);
    using Item = std::pair<double, std::size_t>;
    using Heap = std::priority_queue<Item, std::vector<Item>, std::greater<>>;
    Heap heap_f, heap_b;
    dist_f[start] = 0;
    dist_b[goal] = 0;
    heap_f.emplace(0.0, start);
    heap_b.emplace(0.0, goal);
    double mu = start == goal ? 0.0 : kInf;
    std::size_t meet = start == goal ? start : kNone;
    PairSearchResult r;
    const auto top = [](Heap& heap, const std::vector<bool>& done) {
        while (!heap.empty() && done[heap.top().second]) heap.pop();
        return heap.empty() ? kInf : heap.top().first;
    };
    while (true) {
        const double kf = top(heap_f, done_f), kb = top(heap_b, done_b);
        if (kf == kInf || kb == kInf || kf + kb >= mu) break;
        const bool forward = kf <= kb;
        auto& heap = forward ? heap_f : heap_b;
        auto& dist = forward ? dist_f : dist_b;
        auto& done = forward ? done_f : done_b;
        auto& parent = forward ? parent_f : parent_b;
        const auto& other = forward ? dist_b : dist_f;
        const auto& adj = forward ? fwd : bwd;
        const auto [d, u] = heap.top();
        heap.pop();
        done[u] = true;
        (forward ? r.forward_expanded : r.backward_expanded) += 1;
        for (const auto& [v, w] : adj[u]) {
            ++r.edges_scanned;
            if (d + w < dist[v]) dist[v] = d + w, parent[v] = u, heap.emplace(dist[v], v);
            if (other[v] < kInf && d + w + other[v] < mu) {
                mu = d + w + other[v];
                meet = v;
            }
        }
    }
    if (meet == kNone) return not_found(r);
    // Parents along the best path: dist[meet] on each side may come from a different edge than
    // the one that set mu only if it is at least as short, so following parents stays optimal.
    r.found = true;
    r.distance = dist_f[meet] + dist_b[meet];
    r.meeting_vertex = meet;
    r.path = join(meet, parent_f, parent_b);
    return r;
}

std::vector<std::size_t> ball_sizes(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                    std::size_t source, std::size_t max_radius, bool directed) {
    check_terminals(n, source, source);
    const auto adj = unweighted(n, edges, directed).first;
    std::vector<std::size_t> dist(n, kNone), counts(max_radius + 1, 0);
    std::queue<std::size_t> queue;
    dist[source] = 0;
    queue.push(source);
    while (!queue.empty()) {
        const std::size_t u = queue.front();
        queue.pop();
        if (dist[u] > max_radius) break;
        ++counts[dist[u]];
        for (const auto& [v, w] : adj[u])
            if (dist[v] == kNone) dist[v] = dist[u] + 1, queue.push(v);
    }
    for (std::size_t r = 1; r <= max_radius; ++r) counts[r] += counts[r - 1];
    return counts;
}

} // namespace datamunge::algorithms
