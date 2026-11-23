/// @file robust_regression.cpp
/// @brief Iteratively reweighted least squares with Huber or bisquare weights.

#include <datamunge/stats/robust_regression.hpp>
#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Scale, weight and arithmetic helpers for robust regression.
namespace {

/// @brief Safely converts a result from working precision.
double finite_result(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("robust_regression: result exceeds finite double range");
    return static_cast<double>(value);
}

/// @brief Computes the normal-consistent median absolute residual about zero.
long double residual_scale(const std::vector<long double>& residuals) {
    std::vector<long double> absolute;
    absolute.reserve(residuals.size());
    for (const auto residual : residuals)
        absolute.push_back(std::abs(residual));
    const auto middle = absolute.begin() + static_cast<std::ptrdiff_t>(absolute.size() / 2);
    std::nth_element(absolute.begin(), middle, absolute.end());
    const long double median = absolute.size() % 2 ? *middle
        : std::midpoint(*std::max_element(absolute.begin(), middle), *middle);
    return median / 0.6744897501960817432L;
}

/// @brief Evaluates a robust weight at a nonnegative standardized residual.
long double robust_weight(long double absolute_residual, long double cutoff, RobustRegressionLoss loss) {
    if (absolute_residual == 0)
        return 1;
    if (loss == RobustRegressionLoss::Huber)
        return std::min(1.0L, cutoff / absolute_residual);
    if (absolute_residual >= cutoff)
        return 0;
    const long double ratio = absolute_residual / cutoff;
    const long double distance = (1 - ratio) * (1 + ratio);
    return distance * distance;
}

/// @brief Computes residuals in scaled coordinates, rounding changes below the dot-product resolution to zero.
std::vector<long double> model_residuals(const linalg::DenseMatrix<long double>& design,
                                         const std::vector<long double>& response,
                                         const std::vector<long double>& coefficients) {
    std::vector<long double> residuals(response);
    for (std::size_t i = 0; i < design.rows(); ++i) {
        long double magnitude = std::abs(response[i]);
        for (std::size_t j = 0; j < design.cols(); ++j) {
            residuals[i] = std::fma(-design(i, j), coefficients[j], residuals[i]);
            magnitude += std::abs(design(i, j) * coefficients[j]);
        }
        if (!std::isfinite(residuals[i]) || !std::isfinite(magnitude))
            throw std::overflow_error("robust_regression: scaled prediction exceeds working range");
        if (std::abs(residuals[i]) <= 8 * std::numeric_limits<long double>::epsilon() * magnitude)
            residuals[i] = 0;
    }
    return residuals;
}

} // namespace

RobustRegressionResult robust_regression(const linalg::DenseMatrix<double>& design,
                                         const std::vector<double>& response,
                                         const RobustRegressionOptions& options) {
    const auto n = design.rows();
    const auto p = design.cols();
    const auto finite = [](double value) { return std::isfinite(value); };
    const auto positive = [&](double value) { return finite(value) && value > 0; };
    if (p == 0 || n <= p || response.size() != n || !std::ranges::all_of(response, finite) ||
        !std::ranges::all_of(design.data(), finite))
        throw std::invalid_argument("robust_regression: invalid design or response");
    if (options.loss != RobustRegressionLoss::Huber && options.loss != RobustRegressionLoss::TukeyBisquare)
        throw std::invalid_argument("robust_regression: unrecognized loss");
    const auto tuning = options.tuning_constant.value_or(options.loss == RobustRegressionLoss::Huber ? 1.345 : 4.685);
    if (!positive(tuning) || options.max_iterations == 0 || !positive(options.tolerance) ||
        (options.fixed_scale && !positive(*options.fixed_scale)) ||
        (!options.initial_coefficients.empty() && options.initial_coefficients.size() != p) ||
        !std::ranges::all_of(options.initial_coefficients, finite))
        throw std::invalid_argument("robust_regression: invalid fitting controls");

    std::vector<long double> column_scale(p);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            column_scale[j] = std::max(column_scale[j], std::abs(static_cast<long double>(design(i, j))));
    if (std::ranges::find(column_scale, 0.0L) != column_scale.end())
        throw std::invalid_argument("robust_regression: design is rank deficient");
    long double response_scale = 0;
    for (const auto value : response)
        response_scale = std::max(response_scale, std::abs(static_cast<long double>(value)));
    if (response_scale == 0)
        response_scale = 1;

    linalg::DenseMatrix<long double> scaled(n, p);
    std::vector<long double> scaled_response(n);
    for (std::size_t i = 0; i < n; ++i) {
        scaled_response[i] = response[i] / response_scale;
        for (std::size_t j = 0; j < p; ++j)
            scaled(i, j) = design(i, j) / column_scale[j];
    }
    const auto initial = linalg::svd(scaled);
    if (!initial.converged)
        throw std::runtime_error("robust_regression: initial design's SVD reached its sweep limit");
    if (initial.rank() != p)
        throw std::invalid_argument("robust_regression: design is rank deficient");
    auto coefficients = initial.solve(scaled_response);
    if (!options.initial_coefficients.empty())
        for (std::size_t j = 0; j < p; ++j)
            coefficients[j] = (options.initial_coefficients[j] / response_scale) * column_scale[j];
    auto residuals = model_residuals(scaled, scaled_response, coefficients);
    const auto scale_for = [&](const auto& residual) {
        return options.fixed_scale ? *options.fixed_scale / response_scale : residual_scale(residual);
    };

    RobustRegressionResult result{};
    result.status = RobustRegressionStatus::IterationLimit;
    long double relative_change = 0;
    linalg::DenseMatrix<long double> weighted(n, p);
    std::vector<long double> weighted_response(n);
    while (result.iterations < options.max_iterations) {
        const long double scale = scale_for(residuals);
        if (scale == 0)
            break;
        const long double cutoff = scale * tuning;
        for (std::size_t i = 0; i < n; ++i) {
            const auto root_weight = std::sqrt(robust_weight(std::abs(residuals[i]), cutoff, options.loss));
            weighted_response[i] = root_weight * scaled_response[i];
            for (std::size_t j = 0; j < p; ++j)
                weighted(i, j) = root_weight * scaled(i, j);
        }
        const auto fit = linalg::svd(weighted);
        if (!fit.converged) {
            result.status = RobustRegressionStatus::LinearSolveLimit;
            break;
        }
        if (fit.rank() != p) {
            result.status = RobustRegressionStatus::RankDeficient;
            break;
        }
        coefficients = fit.solve(weighted_response);
        auto next = model_residuals(scaled, scaled_response, coefficients);
        long double change = 0;
        for (std::size_t i = 0; i < n; ++i)
            change = std::max(change, std::abs(next[i] - residuals[i]));
        relative_change = change / scale;
        residuals = std::move(next);
        ++result.iterations;
        if (relative_change <= options.tolerance) {
            result.status = RobustRegressionStatus::Converged;
            break;
        }
    }

    const long double final_scale = scale_for(residuals);
    if (final_scale == 0)
        result.status = RobustRegressionStatus::ZeroScale;
    result.scale = finite_result(final_scale * response_scale);
    result.relative_change = finite_result(relative_change);
    result.coefficients.reserve(p);
    for (std::size_t j = 0; j < p; ++j)
        result.coefficients.push_back(finite_result(coefficients[j] * response_scale / column_scale[j]));
    result.residuals.reserve(n);
    result.weights.reserve(n);
    for (const auto residual : residuals) {
        result.residuals.push_back(finite_result(residual * response_scale));
        result.weights.push_back(static_cast<double>(robust_weight(std::abs(residual), final_scale * tuning, options.loss)));
    }
    return result;
}

} // namespace datamunge::stats
