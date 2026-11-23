/// @file negative_binomial_regression.cpp
/// @brief Profile likelihood for NB2 regression including its Poisson limit.

#include <datamunge/stats/negative_binomial_regression.hpp>

#include "likelihood_newton.hpp"
#include "../random/count_likelihood.hpp"

#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Working-scale count likelihood and profile optimization.
namespace {
/// @brief Working-precision vector.
using Vector = std::vector<long double>;
/// @brief Working-precision matrix.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Fit status for both coefficient and profile optimization.
using Status = NegativeBinomialRegressionStatus;
/// @brief Coefficient fit at fixed alpha.
using Fit = detail::LikelihoodFit<Status>;

/// @brief A scaled NB2 design and fixed per-row inputs.
struct Problem {
    Matrix x;                              ///< Column-scaled design.
    const std::vector<std::size_t>& counts; ///< Observed event counts.
    Vector offsets;                        ///< Additive log offsets.
    Vector weights;                        ///< Case weights.
};

/// @brief Conditional likelihood and derivatives at fixed dispersion.
struct Evaluation {
    long double value; ///< Weighted log likelihood.
    Vector score;      ///< Derivative with respect to working coefficients.
    Matrix info;       ///< Lower triangle of observed coefficient information.
};

/// @brief Converts a finite working quantity to public precision.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("negative_binomial_regression: result exceeds finite double range");
    return static_cast<double>(value);
}

/// @brief Selects the shared count log mass at positive alpha or the exact limiting mass.
long double log_mass(std::size_t count, long double eta, long double mean, long double alpha) {
    if (alpha == 0)
        return random::detail::poisson_log_mass(count, eta, mean);
    return random::detail::negative_binomial_log_mass(count, mean, 1 / alpha, alpha * mean);
}

/// @brief Computes the likelihood, score, and information for the conditional coefficient fit.
Evaluation evaluate(const Problem& problem, const Vector& beta, long double alpha) {
    const auto p = beta.size();
    Evaluation result{0, Vector(p), Matrix(p, p)};
    auto linear = problem.x * beta;
    for (std::size_t i = 0; i < linear.size(); ++i) {
        const auto eta = linear[i] + problem.offsets[i];
        const auto mu = std::exp(eta);
        if (!(mu > 0) || !std::isfinite(mu) || !std::isfinite(alpha * mu))
            return {-std::numeric_limits<long double>::infinity(), {}, {}};
        const auto count = problem.counts[i];
        const auto weight = problem.weights[i];
        result.value += weight * log_mass(count, eta, mu, alpha);
        const auto reciprocal = 1 / (1 + alpha * mu);
        const auto score = weight * (count - mu) * reciprocal;
        const auto info = weight * (mu * reciprocal) * ((1 + alpha * count) * reciprocal);
        for (std::size_t j = 0; j < p; ++j) {
            result.score[j] += score * problem.x(i, j);
            for (std::size_t k = 0; k <= j; ++k)
                result.info(j, k) += info * problem.x(i, j) * problem.x(i, k);
        }
    }
    return result;
}

/// @brief Checks coefficient changes in log-mean units and tests for a recession direction.
detail::LikelihoodStep step_effect(const Problem& problem, const Vector& beta) {
    const auto direction = problem.x * beta;
    long double maximum = 0, violation = 0, improvement = 0;
    for (std::size_t i = 0; i < direction.size(); ++i) {
        const auto value = direction[i];
        if (!std::isfinite(value))
            return {std::numeric_limits<long double>::infinity(), false};
        maximum = std::max(maximum, std::abs(value));
        if (problem.counts[i] == 0) {
            violation = std::max(violation, value);
            improvement = std::max(improvement, -value);
        } else {
            violation = std::max(violation, std::abs(value));
        }
    }
    const auto rounding = 64 * std::numeric_limits<long double>::epsilon() * maximum;
    return {maximum, violation <= rounding && improvement > rounding};
}

/// @brief Conditional coefficient optimization for one dispersion value.
std::optional<Fit> fit_coefficients(const Problem& problem, const Vector& initial, long double alpha,
                                   const NegativeBinomialRegressionOptions& options) {
    return detail::maximize_likelihood<Status>(initial, options.max_iterations, options.tolerance,
        [&](const Vector& beta) { return evaluate(problem, beta, alpha); },
        [&](const Vector& step) { return step_effect(problem, step); });
}

/// @brief Optimum found on the profile, including the exact zero-dispersion candidate.
struct ProfileFit {
    Fit fit;            ///< Coefficient fit at the selected dispersion.
    long double alpha; ///< Selected dispersion.
    Status status;     ///< Outcome of the profile search.
};

/// @brief Brackets and refines positive dispersion modes, comparing them with Poisson.
ProfileFit profile(const Problem& problem, Fit poisson, const NegativeBinomialRegressionOptions& options) {
    ProfileFit best{poisson, 0, poisson.status};
    if (poisson.status != Status::Converged)
        return best;
    const auto threshold = 128 * std::numeric_limits<long double>::epsilon() * std::max(1.0L, std::abs(poisson.value));
    best.status = Status::PoissonBoundary;
    bool evaluated = false;
    /// @brief Records the best converged point while evaluating the profile in log-alpha units.
    const auto value = [&](long double log_alpha) {
        const auto alpha = std::exp(log_alpha);
        auto fit = fit_coefficients(problem, poisson.beta, alpha, options);
        if (!fit || fit->status != Status::Converged)
            return -std::numeric_limits<long double>::infinity();
        evaluated = true;
        const auto likelihood = fit->value;
        if (likelihood > best.fit.value && likelihood > poisson.value + threshold)
            best = {std::move(*fit), alpha, Status::Converged};
        return likelihood;
    };
    const auto scale = std::max<std::size_t>(1, std::ranges::max(problem.counts));
    const auto lower = -36 - std::log(static_cast<long double>(scale));
    const long double cap = std::log(static_cast<long double>(std::numeric_limits<double>::max()));
    Vector grid, scores;
    for (auto x = lower; x <= 20; x += 2) {
        grid.push_back(x);
        scores.push_back(value(x));
    }
    while (scores.back() > scores[scores.size() - 2] && scores.back() > poisson.value + threshold &&
           grid.back() + 2 < cap) {
        grid.push_back(grid.back() + 2);
        scores.push_back(value(grid.back()));
    }
    const auto golden = (std::sqrt(5.0L) - 1) / 2;
    for (std::size_t i = 1; i + 1 < grid.size(); ++i) {
        if (scores[i] <= poisson.value + threshold || scores[i] < scores[i - 1] || scores[i] < scores[i + 1])
            continue;
        auto a = grid[i - 1], b = grid[i + 1];
        auto x = std::lerp(a, b, 1 - golden), y = std::lerp(a, b, golden);
        auto fx = value(x), fy = value(y);
        for (std::size_t j = 0; j < options.max_iterations && b - a > options.tolerance; ++j) {
            if (fx < fy) {
                a = x;
                x = y;
                fx = fy;
                y = std::lerp(a, b, golden);
                fy = value(y);
            } else {
                b = y;
                y = x;
                fy = fx;
                x = std::lerp(a, b, 1 - golden);
                fx = value(x);
            }
        }
        if (b - a > options.tolerance && best.alpha > 0 &&
            std::log(best.alpha) >= grid[i - 1] && std::log(best.alpha) <= grid[i + 1])
            best.status = Status::IterationLimit;
    }
    if (!evaluated || (best.alpha > 0 && best.alpha >= std::exp(grid.back())))
        best.status = Status::NumericalFailure;
    return best;
}

/// @brief Observed information for the coefficients and log dispersion at an interior fit.
Matrix joint_information(const Problem& problem, const ProfileFit& fit) {
    const auto p = problem.x.cols();
    Matrix result(p + 1, p + 1);
    auto beta_info = evaluate(problem, fit.fit.beta, fit.alpha).info;
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k <= j; ++k)
            result(j, k) = beta_info(j, k);
    const auto linear = problem.x * fit.fit.beta;
    // The fourth-order stencil differentiates the stable log mass in log-alpha units.
    constexpr long double h = .001L;
    const std::array alpha{fit.alpha * std::exp(-2 * h), fit.alpha * std::exp(-h),
                           fit.alpha * std::exp(h), fit.alpha * std::exp(2 * h)};
    for (std::size_t i = 0; i < linear.size(); ++i) {
        const auto eta = linear[i] + problem.offsets[i];
        const auto mu = std::exp(eta);
        const auto count = problem.counts[i];
        const auto weight = problem.weights[i];
        const auto reciprocal = 1 / (1 + fit.alpha * mu);
        const auto cross = weight * (count - mu) * (fit.alpha * mu * reciprocal) * reciprocal;
        for (std::size_t j = 0; j < p; ++j)
            result(p, j) += cross * problem.x(i, j);
        const auto base = log_mass(count, eta, mu, fit.alpha);
        const auto inner = (log_mass(count, eta, mu, alpha[1]) - base) +
                           (log_mass(count, eta, mu, alpha[2]) - base);
        const auto outer = (log_mass(count, eta, mu, alpha[0]) - base) +
                           (log_mass(count, eta, mu, alpha[3]) - base);
        result(p, p) += weight * (outer - 16 * inner) / (12 * h * h);
    }
    return result;
}
} // namespace

std::vector<double> NegativeBinomialRegressionResult::predict(const linalg::DenseMatrix<double>& design,
                                                               const std::vector<double>& offset) const {
    const auto finite = [](double x) { return std::isfinite(x); };
    if (design.cols() != coefficients.size() || (!offset.empty() && offset.size() != design.rows()) ||
        !std::ranges::all_of(design.data(), finite) || !std::ranges::all_of(offset, finite) ||
        !std::ranges::all_of(coefficients, finite))
        throw std::invalid_argument("negative_binomial_regression: invalid prediction design or offset");
    std::vector<double> result(design.rows());
    for (std::size_t i = 0; i < design.rows(); ++i) {
        long double eta = offset.empty() ? 0 : offset[i];
        for (std::size_t j = 0; j < design.cols(); ++j)
            eta = std::fma(static_cast<long double>(design(i, j)), coefficients[j], eta);
        result[i] = output(std::exp(eta));
    }
    return result;
}

NegativeBinomialRegressionResult negative_binomial_regression(const linalg::DenseMatrix<double>& design,
    const std::vector<std::size_t>& counts, const NegativeBinomialRegressionOptions& options) {
    const auto n = design.rows(), p = design.cols();
    const auto finite = [](double x) { return std::isfinite(x); };
    if (p == 0 || n < p || counts.size() != n || (!options.offset.empty() && options.offset.size() != n) ||
        (!options.weights.empty() && options.weights.size() != n) || options.max_iterations == 0 ||
        !(options.tolerance > 0 && options.tolerance < 1) || !std::ranges::all_of(design.data(), finite) ||
        !std::ranges::all_of(options.offset, finite) ||
        !std::ranges::all_of(options.weights, [&](double x) { return x > 0 && finite(x); }) ||
        std::ranges::all_of(counts, [](auto count) { return count == 0; }))
        throw std::invalid_argument("negative_binomial_regression: invalid input, or all counts are zero");
    if (p == std::numeric_limits<std::size_t>::max() || p + 1 > std::numeric_limits<std::size_t>::max() / (p + 1))
        throw std::length_error("negative_binomial_regression: covariance dimensions exceed capacity");
    Problem problem{Matrix(n, p), counts, Vector(n), Vector(n, 1)};
    Vector scales(p), log_response(n);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i)
            scales[j] = std::max(scales[j], std::abs(static_cast<long double>(design(i, j))));
        if (scales[j] == 0)
            throw std::invalid_argument("negative_binomial_regression: design has a zero column");
        for (std::size_t i = 0; i < n; ++i)
            problem.x(i, j) = design(i, j) / scales[j];
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!options.offset.empty())
            problem.offsets[i] = options.offset[i];
        if (!options.weights.empty())
            problem.weights[i] = options.weights[i];
        log_response[i] = std::log(counts[i] + .5L) - problem.offsets[i];
    }
    const auto factor = linalg::svd(problem.x);
    if (factor.rank() != p || !factor.converged)
        throw std::invalid_argument("negative_binomial_regression: design is numerically rank deficient");
    auto poisson = fit_coefficients(problem, factor.solve(log_response), 0, options);
    if (!poisson)
        throw std::invalid_argument("negative_binomial_regression: initial Poisson information is singular");
    const auto fit = profile(problem, std::move(*poisson), options);
    NegativeBinomialRegressionResult result{{}, output(fit.alpha), {}, {}, output(fit.fit.value), fit.status};
    for (std::size_t j = 0; j < p; ++j)
        result.coefficients.push_back(output(fit.fit.beta[j] / scales[j]));
    if (result.status == Status::Converged || result.status == Status::PoissonBoundary) {
        auto transform = scales;
        for (auto& s : transform)
            s = 1 / s;
        auto information = fit.fit.factor;
        if (fit.alpha > 0) {
            transform.push_back(fit.alpha);
            information = linalg::cholesky(joint_information(problem, fit));
        }
        if (information.ok) {
            const auto d = transform.size();
            const auto inverse = information.solve(Matrix::identity(d));
            result.covariance = linalg::DenseMatrix<double>(d, d);
            for (std::size_t j = 0; j < d; ++j)
                for (std::size_t k = 0; k <= j; ++k)
                    result.covariance(j, k) = result.covariance(k, j) = output(transform[j] * inverse(j, k) * transform[k]);
        } else {
            result.status = Status::NumericalFailure;
        }
    }
    result.fitted = result.predict(design, options.offset);
    return result;
}
} // namespace datamunge::stats
