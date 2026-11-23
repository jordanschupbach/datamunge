/// @file group_lasso.cpp
/// @brief Accelerated proximal optimization and KKT diagnostics for Gaussian group paths.

#include <datamunge/stats/group_lasso.hpp>

#include "proximal_least_squares.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Working design, separable proximal operators, and path construction.
namespace {
/// @brief Working-precision vector.
using Vector = detail::ProximalVector;
using detail::proximal_output;
using detail::soft_threshold;

/// @brief A group of nonoverlapping predictor columns.
struct Group {
    std::size_t label;                ///< Label used in the option and result order.
    std::vector<std::size_t> columns; ///< Columns in input order.
    long double weight;               ///< Multiplier of the group Euclidean norm.
};

/// @brief Group penalties and the associated weighted quadratic objective.
struct Problem {
    detail::PenalizedLeastSquares quadratic; ///< Weighted least-squares data in working coordinates.
    std::vector<Group> groups;                ///< Groups in ascending label order.
    long double alpha;                        ///< Within-group proportion of the penalty.
};

/// @brief Combined L1 and group L2 proximal operation, in this order.
void proximal(const Problem& problem, Vector& candidate, long double lambda, long double lipschitz) {
    const auto individual = (lambda / lipschitz) * problem.alpha;
    const auto grouped = (lambda / lipschitz) * (1 - problem.alpha);
    for (const auto& group : problem.groups) {
        long double norm = 0;
        for (const auto j : group.columns) {
            candidate[j] = soft_threshold(candidate[j], individual);
            norm = std::hypot(norm, candidate[j]);
        }
        const auto cutoff = grouped * group.weight;
        const auto multiplier = norm > cutoff ? 1 - cutoff / norm : 0;
        for (const auto j : group.columns)
            candidate[j] *= multiplier;
    }
}

/// @brief Checks stationarity for active slopes and for whole groups at zero.
long double kkt(const Problem& problem, const Vector& beta, const Vector& gradient, long double lambda) {
    long double residual = 0;
    const auto individual = lambda * problem.alpha, grouped = lambda * (1 - problem.alpha);
    for (const auto& group : problem.groups) {
        long double norm = 0;
        for (const auto j : group.columns)
            norm = std::hypot(norm, beta[j]);
        const auto threshold = grouped * group.weight;
        if (norm == 0) {
            long double excess = 0;
            for (const auto j : group.columns)
                excess = std::hypot(excess, soft_threshold(gradient[j], individual));
            residual = std::max(residual, excess - threshold);
        } else {
            for (const auto j : group.columns) {
                const auto violation = beta[j] == 0 ? soft_threshold(gradient[j], individual) :
                    gradient[j] + threshold * (beta[j] / norm) + std::copysign(individual, beta[j]);
                residual = std::max(residual, std::abs(violation));
            }
        }
    }
    return residual;
}

/// @brief Computes the largest automatic-path penalty needed to make all groups zero.
long double lambda_maximum(const Problem& problem, const Vector& gradient, long double peak_score) {
    if (peak_score == 0 || problem.alpha == 1)
        return peak_score;
    long double upper = 0;
    for (const auto& group : problem.groups) {
        long double norm = 0;
        for (const auto j : group.columns)
            norm = std::hypot(norm, gradient[j]);
        upper = std::max(upper, norm / group.weight / (1 - problem.alpha));
    }
    if (problem.alpha == 0)
        return upper;
    upper = std::min(upper, peak_score / problem.alpha);
    long double lower = 0;
    const Vector zero(gradient.size());
    for (int i = 0; i < 80; ++i) {
        const auto middle = std::midpoint(lower, upper);
        if (kkt(problem, zero, gradient, middle) > 0)
            lower = middle;
        else
            upper = middle;
    }
    return upper;
}

/// @brief Solves one penalty and restores coefficients to their original units.
GroupLassoFit fit_lambda(const Problem& problem, Vector& beta, long double lambda, long double& lipschitz,
                       long double kkt_tolerance, std::size_t limit) {
    /// @brief Applies the two group shrinkage operations at this penalty.
    const auto prox = [&](Vector& values, long double curvature) { proximal(problem, values, lambda, curvature); };
    /// @brief Checks the active and inactive group stationarity conditions.
    const auto stationary = [&](const Vector& values, const Vector& gradient) {
        return kkt(problem, values, gradient, lambda) <= kkt_tolerance;
    };
    const auto solution = detail::solve_proximal(problem.quadratic, beta, lipschitz, limit, prox, stationary);
    const auto residual = kkt(problem, beta, solution.state.gradient, lambda);
    GroupLassoFit fit{proximal_output(lambda), 0, {}, {}, 0, proximal_output(residual),
                      solution.iterations, solution.converged};
    long double intercept = problem.quadratic.intercept, l1 = 0, grouped = 0;
    fit.coefficients.reserve(beta.size());
    for (std::size_t j = 0; j < beta.size(); ++j) {
        const auto raw = beta[j] / problem.quadratic.scales[j];
        fit.coefficients.push_back(proximal_output(raw));
        intercept = std::fma(-raw, problem.quadratic.centers[j], intercept);
        l1 += std::abs(beta[j]);
    }
    for (const auto& group : problem.groups) {
        long double norm = 0;
        for (const auto j : group.columns)
            norm = std::hypot(norm, beta[j]);
        grouped += group.weight * norm;
        if (std::ranges::any_of(group.columns, [&](std::size_t j) { return fit.coefficients[j] != 0; }))
            fit.active_groups.push_back(group.label);
    }
    fit.intercept = proximal_output(intercept);
    fit.objective = proximal_output(solution.state.loss + lambda * (problem.alpha * l1 + (1 - problem.alpha) * grouped));
    return fit;
}

} // namespace

std::vector<double> GroupLassoFit::predict(const linalg::DenseMatrix<double>& design) const {
    return detail::predict_penalized_least_squares(design, coefficients, intercept);
}

std::vector<GroupLassoFit> group_lasso_path(const linalg::DenseMatrix<double>& design,
    const std::vector<double>& response, const std::vector<std::size_t>& groups, const GroupLassoOptions& options) {
    const auto n = design.rows(), d = design.cols();
    const auto finite = [](double value) { return std::isfinite(value); };
    const auto positive = [&](double value) { return value > 0 && finite(value); };
    if (n == 0 || d == 0 || response.size() != n || groups.size() != d ||
        (!options.weights.empty() && options.weights.size() != n) ||
        !(options.alpha >= 0 && options.alpha <= 1) || options.max_iterations == 0 ||
        !positive(options.tolerance) ||
        (options.lambdas.empty() && (options.path_length == 0 ||
          !(options.lambda_min_ratio > 0 && options.lambda_min_ratio < 1))) ||
        !std::ranges::all_of(options.lambdas, [&](double v) { return v >= 0 && finite(v); }) ||
        std::ranges::adjacent_find(options.lambdas, std::less_equal<>{}) != options.lambdas.end() ||
        !std::ranges::all_of(options.group_weights, positive) || !std::ranges::all_of(options.weights, positive) ||
        !std::ranges::all_of(design.data(), finite) || !std::ranges::all_of(response, finite))
        throw std::invalid_argument("group lasso: invalid data or path controls");

    std::map<std::size_t, std::vector<std::size_t>> partition;
    for (std::size_t j = 0; j < d; ++j)
        partition[groups[j]].push_back(j);
    if (!options.group_weights.empty() && options.group_weights.size() != partition.size())
        throw std::invalid_argument("group lasso: expected one penalty weight per distinct group");
    Problem problem{detail::prepare_penalized_least_squares(design, response, options.weights, options.standardize),
                    {}, options.alpha};
    for (auto& [label, columns] : partition) {
        const auto weight = options.group_weights.empty() ? std::sqrt(static_cast<long double>(columns.size())) :
                            static_cast<long double>(options.group_weights[problem.groups.size()]);
        problem.groups.push_back({label, std::move(columns), weight});
    }

    auto lipschitz = problem.quadratic.curvature;
    Vector beta(d);
    const auto centered = detail::evaluate_quadratic(problem.quadratic, beta);
    long double peak_score = 0;
    for (auto value : centered.gradient)
        peak_score = std::max(peak_score, std::abs(value));
    Vector lambdas(options.lambdas.begin(), options.lambdas.end());
    if (lambdas.empty()) {
        const auto maximum = lambda_maximum(problem, centered.gradient, peak_score);
        lambdas.push_back(maximum);
        if (maximum > 0)
            for (std::size_t j = 1; j < options.path_length; ++j) {
                const auto fraction = static_cast<long double>(j) / (options.path_length - 1);
                const auto lambda = maximum * std::pow(static_cast<long double>(options.lambda_min_ratio), fraction);
                if (proximal_output(lambda) < proximal_output(lambdas.back()))
                    lambdas.push_back(lambda);
            }
    }
    std::vector<GroupLassoFit> result;
    result.reserve(lambdas.size());
    for (const auto lambda : lambdas)
        result.push_back(fit_lambda(problem, beta, lambda, lipschitz,
            options.tolerance * peak_score, options.max_iterations));
    return result;
}

} // namespace datamunge::stats
