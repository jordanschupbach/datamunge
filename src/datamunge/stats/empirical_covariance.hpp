#pragma once

/// @file empirical_covariance.hpp
/// @brief Shared wide-precision empirical covariance with common data scaling.

#include <datamunge/linalg/dense_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::stats::detail {

/// @brief Wide-precision observations and covariance matrices.
using CovarianceMatrix = linalg::DenseMatrix<long double>;

/// @brief Centered observations and maximum-likelihood covariance on a common working scale.
struct EmpiricalCovariance {
    CovarianceMatrix data;       ///< Working observations after optional centering and common scaling.
    CovarianceMatrix covariance; ///< Working covariance, divided by the number of original observations.
    std::vector<double> location; ///< Column means in original units, or zero when supplied.
    long double scale;            ///< Common factor for restoring original data units.
};

/// @brief Computes empirical covariance using working precision and a common scale.
inline EmpiricalCovariance empirical_covariance(const linalg::DenseMatrix<double>& observations, bool assume_centered) {
    const auto n = observations.rows(), p = observations.cols();
    if (n == 0 || (!assume_centered && n == 1) || p == 0 ||
        !std::ranges::all_of(observations.data(), [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument("empirical covariance: invalid or insufficient data");
    const auto capacity = std::vector<long double>{}.max_size();
    if (n > capacity / p || p > capacity / p)
        throw std::length_error("empirical covariance: dense workspace exceeds storage capacity");

    EmpiricalCovariance result{CovarianceMatrix(n,p),CovarianceMatrix(p,p),std::vector<double>(p),1};
    long double peak = 0;
    for (std::size_t j = 0; j < p; ++j) {
        const auto origin = assume_centered ? 0.0L : static_cast<long double>(observations(0,j));
        long double offset = 0;
        for (std::size_t i = 0; i < n; ++i) {
            result.data(i,j) = observations(i,j)-origin;
            if (!assume_centered)
                offset += result.data(i,j);
        }
        offset /= n;
        result.location[j] = static_cast<double>(origin+offset);
        for (std::size_t i = 0; i < n; ++i) {
            result.data(i,j) -= offset;
            peak = std::max(peak,std::abs(result.data(i,j)));
        }
    }
    if (peak > 0) {
        result.scale = peak;
        for (auto& value : result.data.data())
            value /= peak;
    }
    // Accumulate by row so each observation and the covariance triangle stream contiguously.
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k <= j; ++k)
                result.covariance(j,k) += result.data(i,j)*result.data(i,k);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t k = 0; k <= j; ++k)
            result.covariance(j,k) = result.covariance(k,j) = result.covariance(j,k)/n;
    }
    return result;
}

} // namespace datamunge::stats::detail
