/// @file lm_marginal_means.cpp
/// @brief Marginal means and contrasts for a fitted linear model.

#include <datamunge/stats/lm.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Reference-grid averaging and interval arithmetic.
namespace {

/// @brief Dense matrix for linear-function and covariance calculations.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Narrows a finite estimate, covariance entry or limit.
double marginal_output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("LM::estimated_marginal_means: result exceeds finite double range");
    return static_cast<double>(value);
}

/// @brief Evaluates a family of linear functions and propagates the coefficient covariance.
MarginalInference summarize_margins(const Matrix& functions, const std::vector<double>& beta,
                                    const linalg::DenseMatrix<double>& covariance,
                                    std::size_t df, const MarginalMeansOptions& options) {
    const auto count = functions.rows();
    const auto p = functions.cols();
    MarginalInference result{{}, linalg::DenseMatrix<double>(count, count)};
    if (count == 0)
        return result;
    const auto family_size = options.simultaneous ? count : 1u;
    const double tail = (1 - options.confidence_level) / (2.0 * static_cast<double>(family_size));
    const auto critical = -random::student_t_quantile(tail, static_cast<double>(df));
    Matrix projected(count, p);
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k < p; ++k)
                projected(i, j) += functions(i, k) * covariance(k, j);

    for (std::size_t i = 0; i < count; ++i) {
        long double estimate = 0, variance = 0;
        for (std::size_t j = 0; j < p; ++j)
            estimate += functions(i, j) * beta[j];
        for (std::size_t j = 0; j <= i; ++j) {
            long double entry = 0;
            for (std::size_t k = 0; k < p; ++k)
                entry += projected(i, k) * functions(j, k);
            if (i == j) {
                // Roundoff in a covariance quadratic form can put a zero just below zero.
                entry = std::max(entry, 0.0L);
                variance = entry;
            }
            result.covariance(i, j) = result.covariance(j, i) = marginal_output(entry);
        }
        const auto se = std::sqrt(variance);
        const auto half_width = critical * se;
        result.estimates.push_back({marginal_output(estimate), marginal_output(se),
                                    marginal_output(estimate - half_width), marginal_output(estimate + half_width)});
    }
    return result;
}

} // namespace

MarginalMeansResult LM::estimated_marginal_means(const dstruct::DataFrame& grid,
                                                const MarginalMeansOptions& options) const {
    const auto n = grid.nrows();
    if (n == 0 || (!options.groups.empty() && options.groups.size() != n) ||
        (!options.weights.empty() && options.weights.size() != n) ||
        !(options.confidence_level > 0 && options.confidence_level < 1))
        throw std::invalid_argument("LM::estimated_marginal_means: invalid grid or interval options");
    if (!std::ranges::all_of(options.weights, [](double x) { return x >= 0 && std::isfinite(x); }) ||
        !std::ranges::all_of(options.contrasts.data(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("LM::estimated_marginal_means: invalid weight or contrast coefficient");

    design_.validate_categorical_levels(grid, "LM::estimated_marginal_means");
    const auto model = design_.build_matrix(grid, /*require_response=*/false);
    if (model.X.rows() != n || !std::ranges::all_of(model.X.data(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("LM::estimated_marginal_means: missing or nonfinite grid predictor");
    MarginalMeansResult result;
    std::map<std::string, std::size_t> group_indices;
    std::vector<std::size_t> groups(n);
    std::vector<double> max_weights;
    const auto weight = [&](std::size_t i) { return options.weights.empty() ? 1.0 : options.weights[i]; };
    for (std::size_t i = 0; i < n; ++i) {
        const std::string label = options.groups.empty() ? std::string{} : options.groups[i];
        const auto [position, inserted] = group_indices.try_emplace(label, result.groups.size());
        const auto index = position->second;
        if (inserted) {
            result.groups.push_back(label);
            max_weights.push_back(0);
        }
        groups[i] = index;
        max_weights[index] = std::max(max_weights[index], weight(i));
    }
    if (std::ranges::find(max_weights, 0.0) != max_weights.end())
        throw std::invalid_argument("LM::estimated_marginal_means: every group needs positive total weight");
    const auto group_count = result.groups.size();
    const auto& contrasts = options.contrasts;
    if (contrasts.cols() != group_count && (contrasts.rows() != 0 || contrasts.cols() != 0))
        throw std::invalid_argument("LM::estimated_marginal_means: contrast columns must match groups");
    const auto p = model.X.cols();
    Matrix means(group_count, p);
    std::vector<long double> total_weights(group_count);
    for (std::size_t i = 0; i < n; ++i) {
        const auto group = groups[i];
        const auto value = static_cast<long double>(weight(i)) / max_weights[group];
        total_weights[group] += value;
        for (std::size_t j = 0; j < p; ++j)
            means(group, j) += value * model.X(i, j);
    }
    for (std::size_t i = 0; i < group_count; ++i)
        for (std::size_t j = 0; j < p; ++j)
            means(i, j) /= total_weights[i];
    Matrix comparison(contrasts.rows(), p);
    for (std::size_t i = 0; i < contrasts.rows(); ++i)
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k < group_count; ++k)
                comparison(i, j) += contrasts(i, k) * means(k, j);
    result.means = summarize_margins(means, coefficients_, covariance_, degrees_of_freedom_, options);
    result.contrasts = summarize_margins(comparison, coefficients_, covariance_, degrees_of_freedom_, options);
    return result;
}

} // namespace datamunge::stats
