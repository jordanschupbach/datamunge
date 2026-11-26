/// @file gauss_hermite.cpp
/// @brief Golub-Welsch construction of normal Gauss-Hermite rules.
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/stats/gauss_hermite.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

GaussHermiteRule normal_gauss_hermite(std::size_t points) {
    if (points == 0)
        throw std::invalid_argument("normal_gauss_hermite: need at least one point");
    GaussHermiteRule rule;
    if (points == 1) {
        rule.nodes = {0.0};
        rule.weights = {1.0};
        return rule;
    }
    linalg::DenseMatrix<double> jacobi(points, points, 0.0);
    for (std::size_t j = 1; j < points; ++j)
        jacobi(j, j - 1) = jacobi(j - 1, j) = std::sqrt(static_cast<double>(j));
    const auto eigen = linalg::symmetric_eigen(jacobi);
    std::vector<std::size_t> order(points);
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](std::size_t a, std::size_t b) { return eigen.eigenvalues[a] < eigen.eigenvalues[b]; });
    double total = 0;
    for (const std::size_t j : order) {
        const double v = eigen.eigenvectors(0, j);
        rule.nodes.push_back(eigen.eigenvalues[j]);
        rule.weights.push_back(v * v);
        total += v * v;
    }
    for (double& w : rule.weights)
        w /= total;
    // Symmetrise to remove rounding asymmetry.
    for (std::size_t j = 0; j < points / 2; ++j) {
        const std::size_t k = points - 1 - j;
        const double x = 0.5 * (rule.nodes[k] - rule.nodes[j]), w = 0.5 * (rule.weights[j] + rule.weights[k]);
        rule.nodes[j] = -x;
        rule.nodes[k] = x;
        rule.weights[j] = rule.weights[k] = w;
    }
    if (points % 2 == 1)
        rule.nodes[points / 2] = 0;
    return rule;
}

} // namespace datamunge::stats
