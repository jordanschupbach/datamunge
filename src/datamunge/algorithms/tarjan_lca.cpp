#include <datamunge/algorithms/tarjan_lca.hpp>

#include <stdexcept>

namespace datamunge::algorithms {

namespace {

// Disjoint-set forest with path compression and union by rank. Every node starts in its own
// singleton set; Tarjan's method resets the per-set representative `ancestor` after each union,
// so any union heuristic is correct -- rank just keeps the trees shallow for near-linear cost.
struct DisjointSet {
    std::vector<std::size_t> parent;
    std::vector<std::size_t> rank;

    explicit DisjointSet(std::size_t n) : parent(n), rank(n, 0) {
        for (std::size_t i = 0; i < n; ++i) parent[i] = i;
    }

    std::size_t find(std::size_t x) {
        std::size_t root = x;
        while (parent[root] != root) root = parent[root];
        while (parent[x] != root) { // path compression: point everything straight at the root
            const std::size_t next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    }

    void unite(std::size_t a, std::size_t b) {
        std::size_t ra = find(a);
        std::size_t rb = find(b);
        if (ra == rb) return;
        if (rank[ra] < rank[rb]) { const std::size_t t = ra; ra = rb; rb = t; }
        parent[rb] = ra;
        if (rank[ra] == rank[rb]) ++rank[ra];
    }
};

} // namespace

std::vector<std::size_t>
tarjan_offline_lca(std::size_t n, std::size_t root,
                   const std::vector<std::pair<std::size_t, std::size_t>>& tree_edges,
                   const std::vector<std::pair<std::size_t, std::size_t>>& queries) {
    // --- validation (checking root < n first also rejects the n == 0 case before n - 1) ---
    if (root >= n)
        throw std::invalid_argument("tarjan_offline_lca: root must be < n");
    if (tree_edges.size() != n - 1)
        throw std::invalid_argument("tarjan_offline_lca: a tree on n nodes must have exactly n-1 edges");
    for (const auto& [u, v] : tree_edges)
        if (u >= n || v >= n)
            throw std::invalid_argument("tarjan_offline_lca: edge endpoints must be < n");
    for (const auto& [a, b] : queries)
        if (a >= n || b >= n)
            throw std::invalid_argument("tarjan_offline_lca: query endpoints must be < n");

    // --- undirected adjacency list of the tree ---
    std::vector<std::vector<std::size_t>> adj(n);
    for (const auto& [u, v] : tree_edges) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // --- queries indexed by node: for query k = (a, b), record (other-endpoint, k) at both ends ---
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> query_at(n);
    for (std::size_t k = 0; k < queries.size(); ++k) {
        const auto [a, b] = queries[k];
        query_at[a].push_back({b, k});
        query_at[b].push_back({a, k});
    }

    std::vector<std::size_t> lca(queries.size(), 0);
    std::vector<char>        black(n, 0);   // node's subtree DFS is complete
    std::vector<std::size_t> ancestor(n);   // representative ancestor of each set (valid at its root)
    DisjointSet              dsu(n);
    for (std::size_t i = 0; i < n; ++i) ancestor[i] = i;

    // --- explicit-stack DFS (avoids recursion depth limits on deep/degenerate trees) ---
    struct Frame {
        std::size_t node;
        std::size_t parent;    // DFS parent, or n for the root (skipped during traversal)
        std::size_t next_child; // index into adj[node] of the next neighbour to visit
    };
    std::vector<Frame> stack;
    stack.push_back({root, n, 0}); // ancestor[root] already == root

    while (!stack.empty()) {
        Frame& f = stack.back();
        if (f.next_child < adj[f.node].size()) {
            const std::size_t v = adj[f.node][f.next_child++];
            if (v == f.parent) continue; // don't walk back up the tree edge we came in on
            ancestor[v] = v;             // make_set(v) already done by DisjointSet ctor
            stack.push_back({v, f.node, 0});
        } else {
            // All of f.node's children are fully processed -> finish this node.
            const std::size_t u      = f.node;
            const std::size_t parent = f.parent;
            black[u] = 1;
            // Answer every query touching u whose partner has already been finished.
            for (const auto& [w, k] : query_at[u])
                if (black[w]) lca[k] = ancestor[dsu.find(w)];
            stack.pop_back();
            // Merge u's whole (now-finished) subtree into its parent's set; the set's ancestor
            // becomes the parent -- so any later find() from within reports the parent.
            if (parent != n) {
                dsu.unite(parent, u);
                ancestor[dsu.find(parent)] = parent;
            }
        }
    }

    return lca;
}

} // namespace datamunge::algorithms
