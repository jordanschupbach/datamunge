#include <datamunge/algorithms/edmonds_karp.hpp>

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {

namespace {

// A single directed residual arc. `residual` is the remaining pushable capacity; `rev` is the
// index (in the shared arc pool) of the paired opposite arc, whose capacity absorbs cancellations.
struct Arc {
    std::size_t to;
    double residual;
    std::size_t rev;
};

constexpr std::size_t kNoParent = static_cast<std::size_t>(-1);

} // namespace

MaxFlowResult edmonds_karp(std::size_t n,
                           const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                           std::size_t source, std::size_t sink) {
    if (source == sink)
        throw std::invalid_argument("edmonds_karp: source and sink must be distinct");
    if (source >= n || sink >= n)
        throw std::invalid_argument("edmonds_karp: source and sink must be valid vertices (< n)");

    // Build the residual graph. Every input edge contributes a forward arc carrying its capacity
    // and a reverse arc initialized to zero capacity; keeping one arc pair per input edge (rather
    // than merging endpoints) makes parallel and antiparallel edges just work and lets us read the
    // flow of each input edge back off its own forward arc.
    std::vector<Arc> arcs;
    arcs.reserve(2 * edges.size());
    std::vector<std::vector<std::size_t>> head(n); // head[v] = indices of arcs leaving v
    std::vector<std::size_t> forward_arc(edges.size()); // forward_arc[i] = arc index of edge i
    std::vector<double> capacity(edges.size());

    for (std::size_t i = 0; i < edges.size(); ++i) {
        const auto [u, v, c] = edges[i];
        if (u >= n || v >= n)
            throw std::invalid_argument("edmonds_karp: edge endpoints must be valid vertices (< n)");
        if (c < 0.0)
            throw std::invalid_argument("edmonds_karp: edge capacities must be non-negative");
        capacity[i] = c;
        const std::size_t fwd = arcs.size();
        const std::size_t bwd = fwd + 1;
        arcs.push_back(Arc{v, c, bwd});
        arcs.push_back(Arc{u, 0.0, fwd});
        head[u].push_back(fwd);
        head[v].push_back(bwd);
        forward_arc[i] = fwd;
    }

    MaxFlowResult result;

    // Repeatedly BFS for a shortest augmenting path and push its bottleneck until none remains.
    std::vector<std::size_t> parent_arc(n); // arc used to first reach each vertex, per BFS
    while (true) {
        std::fill(parent_arc.begin(), parent_arc.end(), kNoParent);
        parent_arc[source] = kNoParent;
        std::vector<char> visited(n, 0);
        visited[source] = 1;
        std::queue<std::size_t> bfs;
        bfs.push(source);
        while (!bfs.empty() && !visited[sink]) {
            const std::size_t u = bfs.front();
            bfs.pop();
            for (const std::size_t ai : head[u]) {
                const Arc& a = arcs[ai];
                if (a.residual > 0.0 && !visited[a.to]) {
                    visited[a.to] = 1;
                    parent_arc[a.to] = ai;
                    bfs.push(a.to);
                }
            }
        }
        if (!visited[sink]) break; // no augmenting path: current flow is maximum

        // Bottleneck = minimum residual capacity along the discovered path.
        double bottleneck = std::numeric_limits<double>::infinity();
        for (std::size_t v = sink; v != source; v = arcs[arcs[parent_arc[v]].rev].to)
            bottleneck = std::min(bottleneck, arcs[parent_arc[v]].residual);
        // Push the bottleneck: consume forward residuals, restore the reverse residuals.
        for (std::size_t v = sink; v != source; v = arcs[arcs[parent_arc[v]].rev].to) {
            Arc& a = arcs[parent_arc[v]];
            a.residual -= bottleneck;
            arcs[a.rev].residual += bottleneck;
        }
        result.max_flow += bottleneck;
    }

    // Per-input-edge flow = capacity consumed on that edge's forward arc.
    result.edge_flow.resize(edges.size());
    for (std::size_t i = 0; i < edges.size(); ++i)
        result.edge_flow[i] = capacity[i] - arcs[forward_arc[i]].residual;

    // Source side of a minimum cut = vertices reachable from the source in the final residual graph.
    std::vector<char> reachable(n, 0);
    reachable[source] = 1;
    std::queue<std::size_t> bfs;
    bfs.push(source);
    while (!bfs.empty()) {
        const std::size_t u = bfs.front();
        bfs.pop();
        for (const std::size_t ai : head[u]) {
            const Arc& a = arcs[ai];
            if (a.residual > 0.0 && !reachable[a.to]) {
                reachable[a.to] = 1;
                bfs.push(a.to);
            }
        }
    }
    for (std::size_t v = 0; v < n; ++v)
        if (reachable[v]) result.min_cut_source_side.push_back(v);

    return result;
}

} // namespace datamunge::algorithms
