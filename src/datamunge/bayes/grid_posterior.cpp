/// @file grid_posterior.cpp
/// @brief Midpoint-rule tensor-grid posterior quadrature.

#include <datamunge/bayes/grid_posterior.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::bayes {

GridPosterior grid_posterior(optim::ArbitraryFunction& log_density, const std::vector<double>& lower,
                             const std::vector<double>& upper, const std::size_t points) {
    const std::size_t d = lower.size();
    if (d == 0 || upper.size() != d) throw std::invalid_argument("grid_posterior: lower and upper must match and be non-empty");
    if (points < 3) throw std::invalid_argument("grid_posterior: need at least 3 points per coordinate");
    double cells = 1.0;
    for (std::size_t j = 0; j < d; ++j) {
        if (!(upper[j] > lower[j])) throw std::invalid_argument("grid_posterior: upper must exceed lower");
        cells *= static_cast<double>(points);
    }
    if (cells > 5e7) throw std::invalid_argument("grid_posterior: grid has more than 5e7 cells");
    const auto total = static_cast<std::size_t>(cells);

    GridPosterior out;
    std::vector<double> h(d);
    double volume = 1.0;
    for (std::size_t j = 0; j < d; ++j) {
        h[j] = (upper[j] - lower[j]) / static_cast<double>(points);
        volume *= h[j];
        std::vector<double> g(points);
        for (std::size_t i = 0; i < points; ++i) g[i] = lower[j] + (static_cast<double>(i) + 0.5) * h[j];
        out.marginal_grid.push_back(g);
    }

    // Pass 1: log density at every cell centre (index digits = per-coordinate cell numbers).
    std::vector<double> values(total);
    std::vector<std::size_t> idx(d, 0);
    std::vector<double> theta(d);
    double top = -std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < total; ++c) {
        for (std::size_t j = 0; j < d; ++j) theta[j] = out.marginal_grid[j][idx[j]];
        values[c] = log_density.evaluate(theta);
        if (values[c] > top) top = values[c];
        for (std::size_t j = 0; j < d && ++idx[j] == points; ++j) idx[j] = 0;
    }
    if (!std::isfinite(top)) throw std::invalid_argument("grid_posterior: log density is not finite anywhere on the grid");

    // Pass 2: weighted moments and marginals.
    double z = 0.0, edge = 0.0;
    std::vector<double> m1(d, 0.0);
    std::vector<std::vector<double>> m2(d, std::vector<double>(d, 0.0));
    out.marginal_density.assign(d, std::vector<double>(points, 0.0));
    std::fill(idx.begin(), idx.end(), 0);
    for (std::size_t c = 0; c < total; ++c) {
        const double w = std::exp(values[c] - top);
        z += w;
        bool on_edge = false;
        for (std::size_t j = 0; j < d; ++j) {
            theta[j] = out.marginal_grid[j][idx[j]];
            on_edge = on_edge || idx[j] == 0 || idx[j] + 1 == points;
            out.marginal_density[j][idx[j]] += w;
        }
        if (on_edge) edge += w;
        for (std::size_t i = 0; i < d; ++i) {
            m1[i] += w * theta[i];
            for (std::size_t j = 0; j < d; ++j) m2[i][j] += w * theta[i] * theta[j];
        }
        for (std::size_t j = 0; j < d && ++idx[j] == points; ++j) idx[j] = 0;
    }
    out.log_evidence = top + std::log(z * volume);
    out.edge_mass = edge / z;
    for (std::size_t i = 0; i < d; ++i) out.mean.push_back(m1[i] / z);
    out.covariance.assign(d, std::vector<double>(d, 0.0));
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j) out.covariance[i][j] = m2[i][j] / z - out.mean[i] * out.mean[j];
    for (std::size_t i = 0; i < d; ++i) out.sd.push_back(std::sqrt(std::max(0.0, out.covariance[i][i])));
    for (std::size_t j = 0; j < d; ++j)
        for (double& v : out.marginal_density[j]) v /= z * h[j];
    return out;
}

} // namespace datamunge::bayes
