/// @file ordinal_regression.cpp
/// @brief Ordered cumulative logits and a joint Rao score test.

#include <datamunge/stats/ordinal_regression.hpp>

#include "likelihood_newton.hpp"

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Working coordinates and stable cumulative-logit arithmetic.
namespace {
/// @brief Working-precision vector.
using Vector = std::vector<long double>;
/// @brief Working-precision matrix.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Ordinal model data in centered unit-range predictor coordinates.
struct Problem {
    Matrix x;                       ///< Centered and column-scaled predictors.
    std::vector<std::size_t> y;    ///< Zero-based outcome indices.
    Vector weights;                 ///< Positive prior weights.
    std::size_t thresholds;        ///< One fewer than the number of observed categories.
};

/// @brief Likelihood and derivatives in either coefficient layout.
struct Evaluation {
    long double value; ///< Weighted log likelihood.
    Vector score;      ///< Score in working coordinates.
    Matrix info;       ///< Lower triangle of the information matrix.
};

/// @brief Contribution of one category and its two incident thresholds.
struct Category {
    long double log_probability;  ///< Log of the category probability.
    long double lower_base;       ///< Lower threshold score, before the gap term.
    long double upper_base;       ///< Upper threshold score, before the gap term.
    long double lower_variance;   ///< Logistic variance at the lower threshold.
    long double upper_variance;   ///< Logistic variance at the upper threshold.
    long double gap_score;        ///< Opposing score on the two thresholds when both are finite.
};

/// @brief Logistic cumulative probability computed without overflowing the exponential.
long double sigmoid(long double x) {
    const auto e = std::exp(-std::abs(x));
    return x >= 0 ? 1 / (1 + e) : e / (1 + e);
}

/// @brief Log logistic cumulative probability.
long double log_sigmoid(long double x) {
    return std::min(x, 0.0L) - std::log1p(std::exp(-std::abs(x)));
}

/// @brief Evaluates a category without subtracting nearly equal cumulative probabilities.
Category category(std::span<const long double> cutpoints, long double offset, std::size_t y) {
    Category result{};
    if (y > 0) {
        const auto a = cutpoints[y - 1] - offset;
        result.log_probability += log_sigmoid(-a);
        result.lower_base = -sigmoid(a);
        result.lower_variance = -result.lower_base * sigmoid(-a);
    }
    if (y < cutpoints.size()) {
        const auto b = cutpoints[y] - offset;
        result.log_probability += log_sigmoid(b);
        result.upper_base = sigmoid(-b);
        result.upper_variance = result.upper_base * sigmoid(b);
    }
    if (y > 0 && y < cutpoints.size()) {
        const auto gap = cutpoints[y] - cutpoints[y - 1];
        const auto difference = -std::expm1(-gap);
        result.log_probability += std::log(difference);
        result.gap_score = std::exp(-gap) / difference;
    }
    return result;
}

/// @brief Tests strict ordering of the candidate thresholds.
bool ordered(std::span<const long double> cutpoints) {
    return std::adjacent_find(cutpoints.begin(), cutpoints.end(), std::greater_equal<>()) == cutpoints.end();
}

/// @brief Narrows a finite working estimate.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("ordinal_regression: result exceeds finite double range");
    return static_cast<double>(value);
}

/// @brief Computes the common linear predictor for a training row.
long double linear_predictor(const Problem& problem, std::size_t row, const Vector& beta) {
    long double value = 0;
    for (std::size_t j = 0; j < problem.x.cols(); ++j)
        value = std::fma(problem.x(row, j), beta[problem.thresholds + j], value);
    return value;
}

/// @brief Evaluates the proportional model's score and observed information.
Evaluation evaluate(const Problem& problem, const Vector& beta) {
    const auto t = problem.thresholds, d = beta.size(), p = problem.x.cols();
    const auto cutpoints = std::span(beta).first(t);
    if (!ordered(cutpoints))
        return {-std::numeric_limits<long double>::infinity(), {}, {}};
    Evaluation result{0, Vector(d), Matrix(d, d)};
    for (std::size_t i = 0; i < problem.x.rows(); ++i) {
        const auto y = problem.y[i];
        const auto w = problem.weights[i];
        const auto term = category(cutpoints, linear_predictor(problem, i, beta), y);
        result.value += w * term.log_probability;
        if (y > 0) {
            result.score[y - 1] += w * (term.lower_base - term.gap_score);
            result.info(y - 1, y - 1) += w * term.lower_variance;
        }
        if (y < t) {
            result.score[y] += w * (term.upper_base + term.gap_score);
            result.info(y, y) += w * term.upper_variance;
        }
        if (y > 0 && y < t) {
            const auto gap_information = w * term.gap_score * (1 + term.gap_score);
            result.info(y - 1, y - 1) += gap_information;
            result.info(y, y) += gap_information;
            result.info(y, y - 1) -= gap_information;
        }
        const auto combined_variance = w * (term.lower_variance + term.upper_variance);
        for (std::size_t j = 0; j < p; ++j) {
            const auto x = problem.x(i, j);
            result.score[t + j] -= w * x * (term.lower_base + term.upper_base);
            if (y > 0)
                result.info(t + j, y - 1) -= w * x * term.lower_variance;
            if (y < t)
                result.info(t + j, y) -= w * x * term.upper_variance;
            for (std::size_t k = 0; k <= j; ++k)
                result.info(t + j, t + k) += combined_variance * x * problem.x(i, k);
        }
    }
    return result;
}

/// @brief Measures changes in cumulative logits and checks for a recession direction.
detail::LikelihoodStep step_effect(const Problem& problem, const Vector& step) {
    const auto t = problem.thresholds;
    long double maximum = 0, violation = 0, improvement = 0;
    for (std::size_t j = 1; j < t; ++j)
        violation = std::max(violation, step[j - 1] - step[j]);
    for (std::size_t i = 0; i < problem.x.rows(); ++i) {
        const auto shift = linear_predictor(problem, i, step);
        const auto y = problem.y[i];
        for (std::size_t j = 0; j < t; ++j)
            maximum = std::max(maximum, std::abs(step[j] - shift));
        if (y > 0) {
            violation = std::max(violation, step[y - 1] - shift);
            improvement = std::max(improvement, shift - step[y - 1]);
        }
        if (y < t) {
            violation = std::max(violation, shift - step[y]);
            improvement = std::max(improvement, step[y] - shift);
        }
    }
    const auto rounding = 64 * std::numeric_limits<long double>::epsilon() * maximum;
    return {maximum, violation <= rounding && improvement > rounding};
}

/// @brief Rejects dimensions that cannot be squared into a dense information matrix.
void check_square(std::size_t dimension) {
    if (dimension > std::numeric_limits<std::size_t>::max() / dimension)
        throw std::length_error("ordinal_regression: information dimensions exceed capacity");
}

/// @brief Computes the efficient Rao statistic using expected information for the full model.
std::optional<ParallelSlopesTest> test_parallel(const Problem& problem, const Vector& beta) {
    const auto t = problem.thresholds, p = problem.x.cols(), width = p + 1;
    if (p == 0 || t < 2)
        return std::nullopt;
    if (t > std::numeric_limits<std::size_t>::max() / width)
        throw std::length_error("ordinal_regression: score test dimensions exceed capacity");
    const auto full = t * width;
    check_square(full);
    const auto cutpoints = std::span(beta).first(t);
    Matrix info(full, full);
    Vector score(full), row(width);
    row[0] = 1;
    for (std::size_t i = 0; i < problem.x.rows(); ++i) {
        const auto eta = linear_predictor(problem, i, beta);
        const auto w = problem.weights[i];
        for (std::size_t j = 0; j < p; ++j)
            row[j + 1] = -problem.x(i, j);
        for (std::size_t y = 0; y <= t; ++y) {
            const auto term = category(cutpoints, eta, y);
            const auto lower = term.lower_base - term.gap_score;
            const auto upper = term.upper_base + term.gap_score;
            const auto probability = w * std::exp(term.log_probability);
            for (std::size_t j = 0; j < width; ++j) {
                if (y == problem.y[i]) {
                    if (y > 0)
                        score[(y - 1) * width + j] += w * lower * row[j];
                    if (y < t)
                        score[y * width + j] += w * upper * row[j];
                }
                for (std::size_t k = 0; k <= j; ++k) {
                    if (y > 0)
                        info((y - 1) * width + j, (y - 1) * width + k) += probability * lower * lower * row[j] * row[k];
                    if (y < t)
                        info(y * width + j, y * width + k) += probability * upper * upper * row[j] * row[k];
                }
                if (y > 0 && y < t)
                    for (std::size_t k = 0; k < width; ++k)
                        info(y * width + j, (y - 1) * width + k) += probability * upper * lower * row[j] * row[k];
            }
        }
    }
    const auto full_factor = linalg::cholesky(info);
    if (!full_factor.ok)
        return std::nullopt;
    /// @brief Maps a full coefficient index to the corresponding common-slope coefficient.
    const auto restricted_index = [&](std::size_t j) {
        return j % width == 0 ? j / width : t + j % width - 1;
    };
    Matrix reduced(t + p, t + p);
    Vector reduced_score(t + p);
    for (std::size_t j = 0; j < full; ++j) {
        const auto a = restricted_index(j);
        reduced_score[a] += score[j];
        for (std::size_t k = 0; k <= j; ++k) {
            const auto b = restricted_index(k);
            reduced(std::max(a, b), std::min(a, b)) += (j != k && a == b ? 2 : 1) * info(j, k);
        }
    }
    const auto reduced_factor = linalg::cholesky(reduced);
    if (!reduced_factor.ok)
        return std::nullopt;
    const auto full_solve = full_factor.solve(score);
    const auto reduced_solve = reduced_factor.solve(reduced_score);
    const auto full_stat = std::inner_product(score.begin(), score.end(), full_solve.begin(), 0.0L);
    const auto reduced_stat = std::inner_product(reduced_score.begin(), reduced_score.end(), reduced_solve.begin(), 0.0L);
    const auto delta = full_stat - reduced_stat;
    if (!std::isfinite(delta) || delta < -1e-10L * (1 + std::abs(full_stat)))
        return std::nullopt;
    const auto statistic = output(std::max(0.0L, delta));
    const auto df = p * (t - 1);
    return ParallelSlopesTest{statistic, df, random::gamma_cdf(statistic, static_cast<double>(df) / 2, 2, false)};
}
} // namespace

linalg::DenseMatrix<double> OrdinalRegressionResult::predict(const linalg::DenseMatrix<double>& design) const {
    const auto finite = [](double x) { return std::isfinite(x); };
    Vector cuts(cutpoints.begin(), cutpoints.end());
    if (cuts.empty() || cuts.size() + 1 != levels.size() || design.cols() != coefficients.size() ||
        !std::ranges::all_of(cutpoints, finite) || !ordered(cuts) ||
        !std::ranges::all_of(coefficients, finite) || !std::ranges::all_of(design.data(), finite))
        throw std::invalid_argument("ordinal_regression: invalid prediction design or fitted parameters");
    linalg::DenseMatrix<double> result(design.rows(), levels.size());
    for (std::size_t i = 0; i < design.rows(); ++i) {
        long double eta = 0;
        for (std::size_t j = 0; j < design.cols(); ++j)
            eta = std::fma(static_cast<long double>(design(i, j)), coefficients[j], eta);
        if (!std::isfinite(eta))
            throw std::overflow_error("ordinal_regression: nonfinite query predictor");
        for (std::size_t y = 0; y < levels.size(); ++y)
            result(i, y) = output(std::exp(category(cuts, eta, y).log_probability));
    }
    return result;
}

OrdinalRegressionResult ordinal_regression(const linalg::DenseMatrix<double>& design,
    const std::vector<std::int64_t>& labels, const OrdinalRegressionOptions& options) {
    using Status = OrdinalRegressionStatus;
    const auto n = design.rows(), p = design.cols();
    const auto finite = [](double x) { return std::isfinite(x); };
    if (n < 2 || p >= n || labels.size() != n || (!options.weights.empty() && options.weights.size() != n) ||
        options.max_iterations == 0 || !(options.tolerance > 0 && options.tolerance < 1) ||
        !std::ranges::all_of(design.data(), finite) ||
        !std::ranges::all_of(options.weights, [&](double x) { return x > 0 && finite(x); }))
        throw std::invalid_argument("ordinal_regression: invalid input dimensions or controls");
    auto levels = labels;
    std::ranges::sort(levels);
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    if (levels.size() < 2)
        throw std::invalid_argument("ordinal_regression: at least two levels are required");
    const auto t = levels.size() - 1;
    if (p > std::numeric_limits<std::size_t>::max() - t)
        throw std::length_error("ordinal_regression: parameter count exceeds capacity");
    check_square(p + t);
    Problem problem{Matrix(n, p), std::vector<std::size_t>(n), Vector(n, 1), t};
    Vector center(p), scales(p), totals(t + 1);
    for (std::size_t j = 0; j < p; ++j) {
        center[j] = design(0, j);
        for (std::size_t i = 0; i < n; ++i) {
            problem.x(i, j) = static_cast<long double>(design(i, j)) - center[j];
            scales[j] = std::max(scales[j], std::abs(problem.x(i, j)));
        }
        if (!(scales[j] > 0) || !std::isfinite(scales[j]))
            throw std::invalid_argument("ordinal_regression: predictor must vary and have finite range");
        for (std::size_t i = 0; i < n; ++i)
            problem.x(i, j) /= scales[j];
    }
    for (std::size_t i = 0; i < n; ++i) {
        problem.y[i] = std::ranges::lower_bound(levels, labels[i]) - levels.begin();
        if (!options.weights.empty())
            problem.weights[i] = options.weights[i];
        totals[problem.y[i]] += problem.weights[i];
    }
    Vector beta(p + t), right(t);
    long double tail = 0, head = 0;
    for (std::size_t j = t; j-- > 0;)
        right[j] = tail += totals[j + 1];
    for (std::size_t j = 0; j < t; ++j) {
        head += totals[j];
        beta[j] = std::log(head) - std::log(right[j]);
    }
    if (!ordered(std::span(beta).first(t)))
        throw std::overflow_error("ordinal_regression: marginal category spacings exceed working precision");
    const auto fit = detail::maximize_likelihood<Status>(std::move(beta), options.max_iterations, options.tolerance,
        [&](const Vector& candidate) { return evaluate(problem, candidate); },
        [&](const Vector& step) { return step_effect(problem, step); });
    if (!fit)
        throw std::invalid_argument("ordinal_regression: design is aliased or initial information is singular");
    OrdinalRegressionResult result{std::move(levels), {}, {}, {}, {}, output(fit->value),
                                    fit->iterations, fit->status, {}};
    long double shift = 0;
    for (std::size_t j = 0; j < p; ++j) {
        const auto coefficient = fit->beta[t + j] / scales[j];
        result.coefficients.push_back(output(coefficient));
        shift = std::fma(center[j], coefficient, shift);
    }
    for (std::size_t j = 0; j < t; ++j)
        result.cutpoints.push_back(output(fit->beta[j] + shift));
    if (std::adjacent_find(result.cutpoints.begin(), result.cutpoints.end(), std::greater_equal<>()) != result.cutpoints.end())
        throw std::overflow_error("ordinal_regression: fitted thresholds lose order in double precision");
    if (fit->status == Status::Converged) {
        auto jacobian = Matrix::identity(t + p);
        for (std::size_t j = 0; j < p; ++j) {
            jacobian(t + j, t + j) = 1 / scales[j];
            for (std::size_t k = 0; k < t; ++k)
                jacobian(k, t + j) = center[j] / scales[j];
        }
        const auto cov = jacobian * fit->factor.solve(Matrix::identity(t + p)) * jacobian.transpose();
        result.covariance = linalg::DenseMatrix<double>(t + p, t + p);
        for (std::size_t j = 0; j < t + p; ++j)
            for (std::size_t k = 0; k <= j; ++k)
                result.covariance(j, k) = result.covariance(k, j) = output(cov(j, k));
        if (options.test_parallel_slopes)
            result.parallel_slopes = test_parallel(problem, fit->beta);
    }
    result.fitted_probabilities = result.predict(design);
    return result;
}
} // namespace datamunge::stats
