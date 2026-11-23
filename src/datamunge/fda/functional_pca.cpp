/// @file functional_pca.cpp
/// @brief Quadrature-weighted covariance eigenfunctions and curve scores.

#include <datamunge/fda/functional_pca.hpp>
#include <datamunge/linalg/svd.hpp>

#include "functional_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::fda {
/// @brief Numerical checks for the common-grid analysis.
namespace {

/// @brief Converts extended precision arithmetic to a finite reported value.
double report_value(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("functional PCA: result is outside the finite reporting range");
    return static_cast<double>(value);
}

} // namespace

FunctionalPCAFit functional_pca(const std::vector<double>& grid,
    const linalg::DenseMatrix<double>& curves, std::size_t components) {
    const auto n = curves.rows(), m = curves.cols();
    if (n < 2 || m < 2 || grid.size() != m || components == 0 || components > std::min(n-1,m))
        throw std::invalid_argument("functional PCA: invalid grid, curve dimensions or component count");
    const auto weights = detail::trapezoidal_weights(grid);
    if (n > std::vector<long double>{}.max_size()/m)
        throw std::length_error("functional PCA: weighted data exceed dense storage capacity");

    long double amplitude = 0;
    for (const double x : curves.data()) {
        if (!std::isfinite(x))
            throw std::invalid_argument("functional PCA: curves must be finite");
        amplitude = std::max(amplitude,static_cast<long double>(std::abs(x)));
    }
    if (amplitude == 0)
        throw std::domain_error("functional PCA: centered curves have zero variance");
    const long double max_weight = *std::max_element(weights.begin(),weights.end());
    const long double root_weight = std::sqrt(max_weight);
    linalg::DenseMatrix<long double> weighted(n,m);
    std::vector<double> mean(m);
    for (std::size_t j = 0; j < m; ++j) {
        // Center offsets from a fixed row so a constant column stays exactly zero.
        const auto origin = static_cast<long double>(curves(0,j))/amplitude;
        long double shift = 0;
        for (std::size_t i = 0; i < n; ++i)
            shift += static_cast<long double>(curves(i,j))/amplitude-origin;
        shift /= n;
        mean[j] = report_value((origin+shift)*amplitude);
        const auto weight = std::sqrt(static_cast<long double>(weights[j]))/root_weight;
        for (std::size_t i = 0; i < n; ++i)
            weighted(i,j) = (static_cast<long double>(curves(i,j))/amplitude-origin-shift)*weight;
    }
    const auto decomposition = linalg::svd(weighted);
    if (!decomposition.converged)
        throw std::runtime_error("functional PCA: weighted SVD did not converge");
    const auto rank = std::min(n-1,decomposition.rank(std::max(n,m)*std::numeric_limits<double>::epsilon()));
    if (rank < components)
        throw std::domain_error("functional PCA: too few positive-variance directions");
    const auto total = std::inner_product(decomposition.s.begin(),decomposition.s.end(),
        decomposition.s.begin(),0.0L);
    FunctionalPCAFit fit{std::move(mean),weights,linalg::DenseMatrix<double>(m,components),
        linalg::DenseMatrix<double>(n,components),std::vector<double>(components),
        std::vector<double>(components),rank};
    for (std::size_t k = 0; k < components; ++k) {
        const auto singular_value = decomposition.s[k];
        const auto sample_deviation = singular_value/std::sqrt(static_cast<long double>(n-1))
            * root_weight * amplitude;
        fit.eigenvalues[k] = report_value(sample_deviation*sample_deviation);
        if (fit.eigenvalues[k] == 0)
            throw std::overflow_error("functional PCA: a positive eigenvalue rounds to zero");
        fit.explained_fraction[k] = report_value(singular_value*singular_value/total);
        long double sign_value = 0;
        for (std::size_t j = 0; j < m; ++j) {
            const auto value = decomposition.V(j,k)/std::sqrt(static_cast<long double>(weights[j]));
            fit.eigenfunctions(j,k) = report_value(value);
            if (std::abs(value) > std::abs(sign_value))
                sign_value = value;
        }
        const long double sign = sign_value < 0 ? -1 : 1;
        for (std::size_t j = 0; j < m; ++j)
            fit.eigenfunctions(j,k) *= static_cast<double>(sign);
        for (std::size_t i = 0; i < n; ++i)
            fit.scores(i,k) = report_value(sign*decomposition.U(i,k)*singular_value*root_weight*amplitude);
    }
    return fit;
}

} // namespace datamunge::fda
