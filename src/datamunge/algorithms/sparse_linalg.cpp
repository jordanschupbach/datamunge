#include <datamunge/algorithms/sparse_linalg.hpp>

#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>

namespace datamunge::algorithms {

std::vector<int> minimum_degree_ordering(const std::vector<std::vector<int>>& adjacency) {
    const int                 n = static_cast<int>(adjacency.size());
    std::vector<std::set<int>> adj(n);
    for (int i = 0; i < n; ++i)
        for (int v : adjacency[i]) if (v != i) adj[i].insert(v);

    std::vector<char> gone(n, 0);
    std::vector<int>  order;
    order.reserve(n);

    for (int step = 0; step < n; ++step) {
        // Pick the uneliminated vertex of minimum current degree.
        int v = -1;
        for (int i = 0; i < n; ++i)
            if (!gone[i] && (v == -1 || adj[i].size() < adj[v].size())) v = i;

        order.push_back(v);
        gone[v] = 1;

        std::vector<int> nbrs(adj[v].begin(), adj[v].end());
        // Make the neighbours a clique (fill), then detach v.
        for (std::size_t a = 0; a < nbrs.size(); ++a)
            for (std::size_t b = a + 1; b < nbrs.size(); ++b) {
                adj[nbrs[a]].insert(nbrs[b]);
                adj[nbrs[b]].insert(nbrs[a]);
            }
        for (int u : nbrs) adj[u].erase(v);
        adj[v].clear();
    }
    return order;
}

std::vector<std::vector<int>> symbolic_cholesky(const std::vector<std::vector<int>>& adjacency) {
    const int                  n = static_cast<int>(adjacency.size());
    std::vector<std::set<int>> adj(n);
    for (int i = 0; i < n; ++i)
        for (int v : adjacency[i]) if (v != i) adj[i].insert(v);

    std::vector<std::set<int>> Lset(n); // Lset[i] = columns j<i with L(i,j) nonzero
    for (int k = 0; k < n; ++k) {
        // Below-diagonal nonzeros of column k: current neighbours with index > k.
        std::vector<int> higher;
        for (int u : adj[k]) if (u > k) higher.push_back(u);
        for (int u : higher) Lset[u].insert(k);
        // Eliminating k couples all of them (fill).
        for (std::size_t a = 0; a < higher.size(); ++a)
            for (std::size_t b = a + 1; b < higher.size(); ++b) {
                adj[higher[a]].insert(higher[b]);
                adj[higher[b]].insert(higher[a]);
            }
    }

    std::vector<std::vector<int>> L(n);
    for (int i = 0; i < n; ++i) L[i].assign(Lset[i].begin(), Lset[i].end());
    return L;
}

} // namespace datamunge::algorithms
