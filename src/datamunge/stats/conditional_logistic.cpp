/// @file conditional_logistic.cpp
/// @brief Log-space subset recursion and maximum conditional likelihood.

#include <datamunge/stats/conditional_logistic.hpp>

#include "likelihood_newton.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Scaled matched-set arithmetic.
namespace {

/// @brief Working-precision vector.
using Vector = std::vector<long double>;

/// @brief Working-precision dense matrix.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Internal representation of one informative matched set.
struct Stratum {
    Matrix x;                      ///< Centered, scaled covariates, negated for control-subset recursion.
    std::vector<std::size_t> rows; ///< Original row positions.
    std::vector<bool> selected;    ///< Observed membership in the smaller outcome class.
    std::size_t count;             ///< Cardinality of the selected subset.
    bool complement;               ///< True when the recursion selects controls.
};

/// @brief Conditional moments for subsets of a single cardinality.
struct Moments {
    long double log_mass; ///< Log sum of the subset weights.
    Vector mean;          ///< Expected covariate sum over the subset distribution.
    Matrix covariance;    ///< Lower triangle of its covariance.
};

/// @brief Conditional likelihood and derivatives at a coefficient vector.
struct Evaluation {
    long double value; ///< Conditional log likelihood.
    Vector score;      ///< Observed totals minus conditional expectations.
    Matrix info;       ///< Conditional information, stored as a lower triangle.
};

/// @brief Logarithm of the weight of an impossible subset.
constexpr auto zero_mass = -std::numeric_limits<long double>::infinity();

/// @brief Adds nonnegative weights represented by their logarithms.
long double log_add(long double a, long double b) {
    const auto hi = std::max(a, b);
    return hi == zero_mass ? zero_mass : hi + std::log1p(std::exp(std::min(a, b) - hi));
}

/// @brief Narrows a finite statistic or coefficient.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("conditional_logistic: result exceeds finite double range");
    return static_cast<double>(value);
}

/// @brief Merges the exclude and include states using their stable relative weights.
void add_observation(Moments& exclude, const Moments& prev, const Matrix& x, std::size_t row, long double eta,
                     Vector& delta) {
    const auto p = x.cols();
    const auto included = prev.log_mass + eta;
    if (exclude.log_mass == zero_mass) {
        exclude = prev;
        exclude.log_mass = included;
        for (std::size_t j = 0; j < p; ++j)
            exclude.mean[j] += x(row, j);
        return;
    }
    // Centering the mixture covariance prevents cancellation in E[XX'] - E[X]E[X'].
    const auto relative = std::exp(-std::abs(exclude.log_mass - included));
    const auto smaller = relative / (1 + relative);
    const auto take = included > exclude.log_mass ? 1 - smaller : smaller;
    const auto omit = included > exclude.log_mass ? smaller : 1 - smaller;
    exclude.log_mass = std::max(exclude.log_mass, included) + std::log1p(relative);
    for (std::size_t j = 0; j < p; ++j)
        delta[j] = prev.mean[j] + x(row, j) - exclude.mean[j];
    for (std::size_t j = 0; j < p; ++j) {
        const auto with_row = prev.mean[j] + x(row, j);
        exclude.mean[j] = take > omit ? with_row - omit * delta[j] : exclude.mean[j] + take * delta[j];
        for (std::size_t k = 0; k <= j; ++k)
            exclude.covariance(j, k) = omit * exclude.covariance(j, k) + take * prev.covariance(j, k)
                                     + take * omit * delta[j] * delta[k];
    }
}

/// @brief Computes one stratum's logits, removing a common shift.
Vector logits(const Stratum& group, const Vector& beta) {
    auto result = group.x * beta;
    const auto maximum = std::ranges::max(result);
    for (auto& value : result)
        value -= maximum;
    return result;
}

/// @brief Accumulates the exact likelihood and its first two derivatives.
Evaluation evaluate(const std::vector<Stratum>& groups, const Vector& beta) {
    const auto p = beta.size();
    Evaluation result{0, Vector(p), Matrix(p, p)};
    Vector delta(p);
    for (const auto& group : groups) {
        const auto eta = logits(group, beta);
        std::vector<Moments> sums(group.count + 1, Moments{zero_mass, Vector(p), Matrix(p, p)});
        sums[0].log_mass = 0;
        Vector observed(p);
        long double selected_eta = 0;
        for (std::size_t i = 0; i < eta.size(); ++i) {
            if (group.selected[i]) {
                selected_eta += eta[i];
                for (std::size_t j = 0; j < p; ++j)
                    observed[j] += group.x(i, j);
            }
            for (std::size_t k = std::min(group.count, i + 1); k > 0; --k)
                add_observation(sums[k], sums[k - 1], group.x, i, eta[i], delta);
        }
        const auto& total = sums.back();
        result.value += selected_eta - total.log_mass;
        for (std::size_t j = 0; j < p; ++j) {
            result.score[j] += observed[j] - total.mean[j];
            for (std::size_t k = 0; k <= j; ++k)
                result.info(j, k) += total.covariance(j, k);
        }
    }
    return result;
}

/// @brief Checks the ordering of cases and controls along a candidate Newton step.
detail::LikelihoodStep step_effect(const std::vector<Stratum>& groups, const Vector& step) {
    long double maximum = 0, min_margin = std::numeric_limits<long double>::infinity(), max_margin = 0;
    for (const auto& group : groups) {
        const auto shift = group.x * step;
        long double case_min = -zero_mass, case_max = zero_mass, control_min = -zero_mass, control_max = zero_mass;
        for (std::size_t i = 0; i < shift.size(); ++i) {
            maximum = std::max(maximum, std::abs(shift[i]));
            if (group.selected[i]) {
                case_min = std::min(case_min, shift[i]);
                case_max = std::max(case_max, shift[i]);
            } else {
                control_min = std::min(control_min, shift[i]);
                control_max = std::max(control_max, shift[i]);
            }
        }
        min_margin = std::min(min_margin, case_min - control_max);
        max_margin = std::max(max_margin, case_max - control_min);
    }
    const auto rounding = 64 * std::numeric_limits<long double>::epsilon() * maximum;
    return {maximum, min_margin >= -rounding && max_margin > rounding};
}

/// @brief Forward-backward log recursion for inclusion probabilities at a fixed case total.
void assign_probabilities(const Stratum& group, const Vector& beta, std::vector<double>& result) {
    const auto eta = logits(group, beta);
    const auto n = eta.size(), m = group.count;
    if (n == std::numeric_limits<std::size_t>::max() ||
        n + 1 > std::numeric_limits<std::size_t>::max() / (m + 1))
        throw std::length_error("conditional_logistic: probability table exceeds capacity");
    Matrix forward(n + 1, m + 1, zero_mass);
    forward(0, 0) = 0;
    for (std::size_t i = 0; i < n; ++i) {
        forward(i + 1, 0) = 0;
        for (std::size_t k = 1; k <= std::min(i + 1, m); ++k)
            forward(i + 1, k) = log_add(forward(i, k), forward(i, k - 1) + eta[i]);
    }
    Vector suffix(m, zero_mass);
    suffix[0] = 0;
    for (std::size_t i = n; i-- > 0;) {
        auto numerator = zero_mass;
        for (std::size_t k = 0; k < m; ++k)
            numerator = log_add(numerator, forward(i, k) + suffix[m - 1 - k]);
        const auto log_probability = std::min(0.0L, eta[i] + numerator - forward(n, m));
        result[group.rows[i]] = static_cast<double>(group.complement ? -std::expm1(log_probability)
                                                                     : std::exp(log_probability));
        for (std::size_t k = std::min(n - i, m - 1); k > 0; --k)
            suffix[k] = log_add(suffix[k], suffix[k - 1] + eta[i]);
    }
}

} // namespace

ConditionalLogisticResult conditional_logistic(
    const linalg::DenseMatrix<double>& predictors, const std::vector<double>& cases,
    const std::vector<std::int64_t>& strata, const ConditionalLogisticOptions& options) {
    using Status = ConditionalLogisticStatus;
    const auto n = predictors.rows(), p = predictors.cols();
    if (p == 0 || n == 0 || cases.size() != n || strata.size() != n ||
        options.max_iterations == 0 || !(options.tolerance > 0 && options.tolerance < 1) ||
        !std::ranges::all_of(predictors.data(), [](double x) { return std::isfinite(x); }) ||
        !std::ranges::all_of(cases, [](double y) { return y == 0 || y == 1; }))
        throw std::invalid_argument("conditional_logistic: invalid design, cases, strata, or controls");
    std::map<std::int64_t, std::vector<std::size_t>> grouped_rows;
    for (std::size_t i = 0; i < n; ++i)
        grouped_rows[strata[i]].push_back(i);

    std::vector<Stratum> groups;
    Vector scales(p);
    for (auto& entry : grouped_rows) {
        auto& rows = entry.second;
        const auto size = rows.size();
        const auto count = static_cast<std::size_t>(std::ranges::count_if(rows, [&](auto row) { return cases[row] == 1; }));
        if (count == 0 || count == size)
            continue;
        const bool complement = size - count < count;
        Stratum group{Matrix(size, p), std::move(rows), std::vector<bool>(size),
                      std::min(count, size - count), complement};
        for (std::size_t i = 0; i < size; ++i) {
            group.selected[i] = (cases[group.rows[i]] == 1) != complement;
            for (std::size_t j = 0; j < p; ++j) {
                const auto difference = static_cast<long double>(predictors(group.rows[i], j)) -
                                        predictors(group.rows.front(), j);
                group.x(i, j) = complement ? -difference : difference;
                scales[j] = std::max(scales[j], std::abs(difference));
            }
        }
        groups.push_back(std::move(group));
    }
    if (groups.empty() || !std::ranges::all_of(scales, [](long double x) { return x > 0 && std::isfinite(x); }))
        throw std::invalid_argument("conditional_logistic: no mixed strata or a constant within-stratum predictor");
    for (auto& group : groups)
        for (std::size_t i = 0; i < group.x.rows(); ++i)
            for (std::size_t j = 0; j < p; ++j)
                group.x(i, j) /= scales[j];

    const auto fit = detail::maximize_likelihood<Status>(p, options.max_iterations, options.tolerance,
        [&](const Vector& beta) { return evaluate(groups, beta); },
        [&](const Vector& step) { return step_effect(groups, step); });
    if (!fit)
        throw std::invalid_argument("conditional_logistic: predictor design is aliased within strata");

    ConditionalLogisticResult result{{}, {}, cases, output(fit->value), output(fit->null_value), groups.size(),
                                      fit->iterations, fit->status};
    for (std::size_t j = 0; j < p; ++j)
        result.coefficients.push_back(output(fit->beta[j] / scales[j]));
    if (fit->status == Status::Converged) {
        const auto inverse = fit->factor.solve(Matrix::identity(p));
        result.covariance = linalg::DenseMatrix<double>(p, p);
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k <= j; ++k)
                result.covariance(j, k) = result.covariance(k, j) = output(inverse(j, k) / scales[j] / scales[k]);
    }
    for (const auto& group : groups)
        assign_probabilities(group, fit->beta, result.fitted_probabilities);
    return result;
}

} // namespace datamunge::stats
