#include <datamunge/algorithms/kruskal.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {

namespace {

// Disjoint-set union (union-find) with path halving and union by rank. find/unite run in
// O(alpha(n)) amortized -- effectively constant -- which is what makes the cycle test cheap.
class DisjointSet {
  public:
    explicit DisjointSet(std::size_t n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), std::size_t{0}); // every vertex its own root
    }

    std::size_t find(std::size_t x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]]; // path halving: point x at its grandparent
            x = parent_[x];
        }
        return x;
    }

    // Merge the sets of x and y; returns true iff they were previously distinct (so the edge is safe).
    bool unite(std::size_t x, std::size_t y) {
        std::size_t rx = find(x);
        std::size_t ry = find(y);
        if (rx == ry) return false; // same component already: adding (x, y) would close a cycle
        if (rank_[rx] < rank_[ry]) std::swap(rx, ry); // attach the shorter tree under the taller
        parent_[ry] = rx;
        if (rank_[rx] == rank_[ry]) ++rank_[rx];
        return true;
    }

  private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> rank_;
};

} // namespace

MinimumSpanningTree kruskal(std::size_t n,
                            const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges) {
    for (const auto& [u, v, w] : edges) {
        (void)w;
        if (u >= n || v >= n)
            throw std::invalid_argument("kruskal: edge endpoint out of range [0, n)");
    }

    // Sort a copy of the edges ascending by weight. A stable sort keeps equal-weight edges in input
    // order, so the accepted set is deterministic when weights tie.
    std::vector<std::tuple<std::size_t, std::size_t, double>> sorted = edges;
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });

    MinimumSpanningTree result;
    DisjointSet dsu(n);
    for (const auto& e : sorted) {
        const auto& [u, v, w] = e;
        if (dsu.unite(u, v)) { // edge joins two different components -> safe to add (no cycle)
            result.edges.push_back(e);
            result.total_weight += w;
            if (n != 0 && result.edges.size() == n - 1) break; // a tree on n vertices has n-1 edges
        }
    }
    // A spanning tree exists iff exactly n-1 edges were accepted; otherwise this is a forest.
    result.is_connected = (n == 0) || (result.edges.size() == n - 1);
    return result;
}

} // namespace datamunge::algorithms
