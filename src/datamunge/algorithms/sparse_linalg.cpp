#include <datamunge/algorithms/sparse_linalg.hpp>

#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>

namespace datamunge::algorithms {

std::vector<int> cuthill_mckee(const std::vector<std::vector<int>>& adjacency, bool reverse) {
    const int        n = static_cast<int>(adjacency.size());
    std::vector<int> degree(n);
    for (int i = 0; i < n; ++i) degree[i] = static_cast<int>(adjacency[i].size());

    std::vector<char> visited(n, 0);
    std::vector<int>  order;
    order.reserve(n);

    while (static_cast<int>(order.size()) < n) {
        // Start a new component at the lowest-degree unvisited vertex.
        int start = -1;
        for (int i = 0; i < n; ++i)
            if (!visited[i] && (start == -1 || degree[i] < degree[start])) start = i;

        std::vector<int> queue{start};
        visited[start] = 1;
        std::size_t head = 0;
        while (head < queue.size()) {
            const int u = queue[head++];
            order.push_back(u);
            // Enqueue unvisited neighbours in order of increasing degree.
            std::vector<int> nbrs;
            for (int v : adjacency[u])
                if (!visited[v]) nbrs.push_back(v);
            std::sort(nbrs.begin(), nbrs.end(), [&](int a, int b) { return degree[a] < degree[b]; });
            for (int v : nbrs) {
                visited[v] = 1;
                queue.push_back(v);
            }
        }
    }

    if (reverse) std::reverse(order.begin(), order.end());
    return order;
}

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

std::vector<double> cannon_matmul(const std::vector<double>& A, const std::vector<double>& B, int n) {
    auto idx = [n](int i, int j) { return i * n + j; };

    // Initial skew: A row i shifts left by i, B column j shifts up by j.
    std::vector<double> a(n * n), b(n * n), c(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            a[idx(i, j)] = A[idx(i, (j + i) % n)];
            b[idx(i, j)] = B[idx((i + j) % n, j)];
        }

    for (int s = 0; s < n; ++s) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) c[idx(i, j)] += a[idx(i, j)] * b[idx(i, j)];
        // Shift a left by one column, b up by one row.
        std::vector<double> a2(n * n), b2(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                a2[idx(i, j)] = a[idx(i, (j + 1) % n)];
                b2[idx(i, j)] = b[idx((i + 1) % n, j)];
            }
        a.swap(a2);
        b.swap(b2);
    }
    return c;
}

} // namespace datamunge::algorithms
