/// @file covariance_shrinkage.cpp
/// @brief Ledoit–Wolf and Gaussian oracle-approximating spherical shrinkage.

#include <datamunge/stats/covariance_shrinkage.hpp>

#include "empirical_covariance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Shared scaled scatter and estimator-specific shrinkage ratios.
namespace {

/// @brief Working data and empirical covariance matrix.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Scaled observations and second-order quantities for the shrinkage estimator.
struct Moments {
    Matrix data;                 ///< Data after optional centering and common scaling.
    Matrix covariance;           ///< Empirical covariance in working units.
    std::vector<double> location; ///< Location in the original feature units.
    long double scale{1};       ///< Common data scale.
    long double trace{0};       ///< Trace of the empirical covariance.
    long double squared{0};     ///< Squared Frobenius norm of the empirical covariance.
    long double spread{0};      ///< Squared Frobenius distance to the spherical target.
};

/// @brief Calculates the trace and squared norms for the spherical shrinkage target.
Moments empirical(const linalg::DenseMatrix<double>& observations, bool assume_centered) {
    auto sample = detail::empirical_covariance(observations,assume_centered);
    const auto p = sample.covariance.cols();
    Moments result{std::move(sample.data),std::move(sample.covariance),std::move(sample.location),sample.scale};
    for (std::size_t j = 0; j < p; ++j)
        result.trace += result.covariance(j,j);
    const auto target = result.trace/p;
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k <= j; ++k) {
            const auto value = result.covariance(j,k);
            const auto difference = j == k ? value-target : value;
            const auto multiplicity = j == k ? 1 : 2;
            result.squared += multiplicity*value*value;
            result.spread += multiplicity*difference*difference;
        }
    return result;
}

/// @brief Estimates variation of the empirical covariance from centered outer products.
long double ledoit_wolf_variability(const Moments& moments) {
    const auto n = moments.data.rows(), p = moments.data.cols();
    long double fourth = 0;
    for (std::size_t i = 0; i < n; ++i) {
        long double length = 0;
        for (std::size_t j = 0; j < p; ++j)
            length += moments.data(i,j)*moments.data(i,j);
        fourth += length*length;
    }
    auto variation = fourth-n*moments.squared;
    const auto rounding = 64*std::numeric_limits<long double>::epsilon()*std::max(n,p)*fourth;
    if (variation <= rounding) {
        // Recompute the squared residuals directly when the moment subtraction loses precision.
        variation = 0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < p; ++j)
                for (std::size_t k = 0; k <= j; ++k) {
                    const auto residual = moments.data(i,j)*moments.data(i,k)-moments.covariance(j,k);
                    variation += (j == k ? 1 : 2)*residual*residual;
                }
    }
    return (variation/n)/n;
}

/// @brief Reports a covariance entry in original units and checks the finite range.
double restore(long double value, long double scale, bool diagonal) {
    const auto restored = (value*scale)*scale;
    if (!std::isfinite(restored) || std::abs(restored) > std::numeric_limits<double>::max())
        throw std::overflow_error("covariance shrinkage: fitted covariance exceeds double precision range");
    const auto output = static_cast<double>(restored);
    if (diagonal && value > 0 && output == 0)
        throw std::overflow_error("covariance shrinkage: a positive variance underflows double precision");
    return output;
}

/// @brief Mixes the empirical covariance with its spherical target and restores original units.
CovarianceShrinkageResult finish(Moments moments, long double shrinkage) {
    const auto p = moments.covariance.rows();
    const auto target = moments.trace/p;
    CovarianceShrinkageResult result{linalg::DenseMatrix<double>(p,p),std::move(moments.location),
        static_cast<double>(shrinkage),restore(target,moments.scale,true)};
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k <= j; ++k) {
            const auto value = std::lerp(moments.covariance(j,k),j == k ? target : 0,shrinkage);
            result.covariance(j,k) = result.covariance(k,j) = restore(value,moments.scale,j == k);
        }
    return result;
}

} // namespace

CovarianceShrinkageResult ledoit_wolf_covariance(const linalg::DenseMatrix<double>& observations, bool assume_centered) {
    auto moments = empirical(observations,assume_centered);
    const auto shrinkage = moments.spread == 0 ? 1 : std::clamp(ledoit_wolf_variability(moments)/moments.spread,0.0L,1.0L);
    return finish(std::move(moments),shrinkage);
}

CovarianceShrinkageResult oracle_approximating_covariance(const linalg::DenseMatrix<double>& observations,
    bool assume_centered) {
    auto moments = empirical(observations,assume_centered);
    const auto n = static_cast<long double>(observations.rows());
    const auto ratio = 2.0L/observations.cols();
    const auto shrinkage = moments.spread == 0 ? 1 :
        std::clamp(((1-ratio)*moments.squared+moments.trace*moments.trace) / ((n+1-ratio)*moments.spread),0.0L,1.0L);
    return finish(std::move(moments),shrinkage);
}

} // namespace datamunge::stats
