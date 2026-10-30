/// @file frank_wolfe.cpp
/// @brief Gaussian-location NPMLE of mixing weights by Frank-Wolfe, and the EM comparison.

#include <datamunge/algorithms/frank_wolfe.hpp>

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::algorithms {

/// @brief Helpers private to this file.
namespace {

/// @brief Matrix of component densities \f$\varphi_\sigma(x_i-\mu_j)\f$, row i, column j.
std::vector<std::vector<double>> density_matrix(const std::vector<double>& data, const std::vector<double>& grid,
                                                double sigma) {
    if (data.empty() || grid.empty()) throw std::invalid_argument("gaussian mixing weights: empty data or grid");
    if (!(sigma > 0)) throw std::invalid_argument("gaussian mixing weights: sigma must be positive");
    const double                     c = 1.0 / (sigma * std::sqrt(2 * std::numbers::pi));
    std::vector<std::vector<double>> phi(data.size(), std::vector<double>(grid.size()));
    for (std::size_t i = 0; i < data.size(); ++i)
        for (std::size_t j = 0; j < grid.size(); ++j) {
            const double z = (data[i] - grid[j]) / sigma;
            phi[i][j]      = c * std::exp(-0.5 * z * z);
        }
    return phi;
}

/// @brief Mixture density \f$m_i=\sum_j\pi_j\varphi_{ij}\f$ at every observation.
std::vector<double> mixture(const std::vector<std::vector<double>>& phi, const std::vector<double>& w) {
    std::vector<double> m(phi.size(), 0.0);
    for (std::size_t i = 0; i < phi.size(); ++i)
        for (std::size_t j = 0; j < w.size(); ++j) m[i] += w[j] * phi[i][j];
    return m;
}

/// @brief Mean negative log-likelihood \f$-\frac1n\sum_i\log m_i\f$.
double mean_nll(const std::vector<double>& m) {
    double s = 0;
    for (const double v : m) s -= std::log(v);
    return s / static_cast<double>(m.size());
}

/// @brief Lindsay's gradient \f$D_j=\frac1n\sum_i\varphi_{ij}/m_i\f$.
std::vector<double> lindsay_gradient(const std::vector<std::vector<double>>& phi, const std::vector<double>& m) {
    std::vector<double> d(phi.front().size(), 0.0);
    for (std::size_t i = 0; i < phi.size(); ++i)
        for (std::size_t j = 0; j < d.size(); ++j) d[j] += phi[i][j] / m[i];
    for (double& v : d) v /= static_cast<double>(phi.size());
    return d;
}

/// @brief Number of weights above \f$10^{-8}\f$.
int support_size(const std::vector<double>& w) {
    int k = 0;
    for (const double v : w) k += v > 1e-8;
    return k;
}

} // namespace

MixingWeightsFit gaussian_location_npmle(const std::vector<double>& data, const std::vector<double>& grid,
                                         double sigma, const FrankWolfeOptions& options) {
    const auto phi = density_matrix(data, grid, sigma);
    const auto f   = [&](const std::vector<double>& w) {
        const auto m = mixture(phi, w);
        for (const double v : m)
            if (!(v > 0)) return std::numeric_limits<double>::infinity();
        return mean_nll(m);
    };
    const auto grad = [&](const std::vector<double>& w) {
        auto d = lindsay_gradient(phi, mixture(phi, w));
        for (double& v : d) v = -v;
        return d;
    };
    // Equal weight on every grid point, each its own atom, so that away and pairwise steps can
    // drop grid points one at a time.
    const std::size_t                J = grid.size();
    std::vector<std::vector<double>> atoms(J, std::vector<double>(J, 0.0));
    for (std::size_t j = 0; j < J; ++j) atoms[j][j] = 1.0;
    const FrankWolfeResult r = frank_wolfe(f, grad, simplex_lmo, std::move(atoms),
                                           std::vector<double>(J, 1.0 / static_cast<double>(J)), options);

    MixingWeightsFit fit;
    fit.grid           = grid;
    fit.weights        = r.x;
    fit.objective      = r.objective;
    fit.gap            = r.gap;
    fit.iterations     = r.iterations;
    fit.converged      = r.converged;
    fit.log_likelihood = -r.objective.back() * static_cast<double>(data.size());
    fit.support.push_back(static_cast<int>(J));
    for (const int a : r.active_atoms) fit.support.push_back(a);
    return fit;
}

MixingWeightsFit gaussian_location_em(const std::vector<double>& data, const std::vector<double>& grid,
                                      double sigma, int max_iter, double tolerance) {
    const auto          phi = density_matrix(data, grid, sigma);
    std::vector<double> w(grid.size(), 1.0 / static_cast<double>(grid.size()));
    MixingWeightsFit    fit;
    fit.grid = grid;
    for (int k = 0;; ++k) {
        const auto m = mixture(phi, w);
        const auto d = lindsay_gradient(phi, m);
        fit.objective.push_back(mean_nll(m));
        fit.gap.push_back(*std::max_element(d.begin(), d.end()) - 1.0);
        fit.support.push_back(support_size(w));
        if (fit.gap.back() <= tolerance) {
            fit.converged = true;
            break;
        }
        if (k >= max_iter) break;
        for (std::size_t j = 0; j < w.size(); ++j) w[j] *= d[j];
        ++fit.iterations;
    }
    fit.weights        = w;
    fit.log_likelihood = -fit.objective.back() * static_cast<double>(data.size());
    return fit;
}

} // namespace datamunge::algorithms
