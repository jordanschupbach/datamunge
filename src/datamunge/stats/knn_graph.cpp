/// @file knn_graph.cpp
/// @brief Brute-force symmetric k-nearest-neighbour graph and its random-walk operator.

#include <datamunge/stats/knn_graph.hpp>

#include <datamunge/nn/linalg.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

KnnGraph::KnnGraph(const std::vector<std::vector<double>>& points, std::size_t k) {
    const std::size_t n = points.size();
    if (n < 2 || k == 0 || k >= n) throw std::invalid_argument("KnnGraph: need 1 <= k < number of points");
    const std::size_t d = points[0].size();
    for (const auto& p : points)
        if (p.size() != d) throw std::invalid_argument("KnnGraph: all points must have the same dimension");

    std::vector<std::vector<std::size_t>> adj(n);
    std::vector<std::pair<double, std::size_t>> dist(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t c = 0; c < d; ++c) {
                const double t = points[i][c] - points[j][c];
                s += t * t;
            }
            dist[j] = {j == i ? INFINITY : s, j};
        }
        std::partial_sort(dist.begin(), dist.begin() + static_cast<std::ptrdiff_t>(k), dist.end());
        for (std::size_t t = 0; t < k; ++t) {
            adj[i].push_back(dist[t].second);
            adj[dist[t].second].push_back(i);
        }
    }
    row_start_.assign(n + 1, 0);
    degree_.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
        row_start_[i + 1] = row_start_[i] + adj[i].size();
        degree_[i]        = static_cast<double>(adj[i].size());
        neighbors_.insert(neighbors_.end(), adj[i].begin(), adj[i].end());
    }
}

std::vector<std::size_t> KnnGraph::neighbors(std::size_t i) const {
    return {neighbors_.begin() + static_cast<std::ptrdiff_t>(row_start_.at(i)),
            neighbors_.begin() + static_cast<std::ptrdiff_t>(row_start_.at(i + 1))};
}

std::size_t KnnGraph::components() const {
    const std::size_t        n = size();
    std::vector<std::size_t> label(n, n), stack;
    std::size_t              count = 0;
    for (std::size_t s = 0; s < n; ++s) {
        if (label[s] != n) continue;
        label[s] = count;
        stack.push_back(s);
        while (!stack.empty()) {
            const std::size_t v = stack.back();
            stack.pop_back();
            for (std::size_t e = row_start_[v]; e < row_start_[v + 1]; ++e)
                if (label[neighbors_[e]] == n) {
                    label[neighbors_[e]] = count;
                    stack.push_back(neighbors_[e]);
                }
        }
        ++count;
    }
    return count;
}

std::vector<double> KnnGraph::adjacency_apply(const std::vector<double>& x) const {
    if (x.size() != size()) throw std::invalid_argument("KnnGraph: vector length must equal the number of vertices");
    std::vector<double> y(size(), 0.0);
    for (std::size_t i = 0; i < size(); ++i)
        for (std::size_t e = row_start_[i]; e < row_start_[i + 1]; ++e) y[i] += x[neighbors_[e]];
    return y;
}

std::vector<double> KnnGraph::random_walk_apply(const std::vector<double>& x) const {
    auto y = adjacency_apply(x);
    for (std::size_t i = 0; i < y.size(); ++i) y[i] /= degree_[i];
    return y;
}

std::vector<double> KnnGraph::random_walk_spectrum() const {
    const std::size_t   n = size();
    std::vector<double> s(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t e = row_start_[i]; e < row_start_[i + 1]; ++e) {
            const std::size_t j = neighbors_[e];
            s[i * n + j]        = 1.0 / std::sqrt(degree_[i] * degree_[j]);
        }
    const auto eig = nn::eigh(nn::from_vector<double>(s, {static_cast<std::int64_t>(n), static_cast<std::int64_t>(n)}),
                              false);
    auto values = eig.e.to_vector<double>();
    std::reverse(values.begin(), values.end());
    return values;
}

}  // namespace datamunge::stats
