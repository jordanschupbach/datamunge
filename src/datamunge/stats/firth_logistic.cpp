/// @file firth_logistic.cpp
/// @brief Jeffreys-penalized logistic likelihood, scoring, and interval inversion.

#include <datamunge/stats/firth_logistic.hpp>

#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Working-scale problem, adjusted score, and constrained fitting.
namespace {

/// @brief Working-precision vector.
using Vector = std::vector<long double>;

/// @brief Working-precision dense matrix.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Thin factorization of the information-weighted design.
using Factor = linalg::SVDDecomposition<long double>;

/// @brief Binomial counts and case weight for one design row.
struct Observation {
    long double successes; ///< Number of events.
    long double failures;  ///< Number of non-events.
    long double weight;    ///< Positive weight on the likelihood kernel.
};

/// @brief Column-scaled design and original binomial outcomes.
struct Problem {
    Matrix design;                            ///< Design with each column divided by its maximum absolute value.
    std::vector<Observation> observations;   ///< Counts in design row order.
};

/// @brief Penalized likelihood, adjusted score, and information at a candidate.
struct Evaluation {
    Vector probabilities; ///< Success probabilities.
    Vector score;         ///< Full adjusted score, in working parameter units.
    Factor factor;        ///< Factorization of the weighted full design.
    long double value;    ///< Penalized log-likelihood kernel in working coordinates.
};

/// @brief Parameters and likelihood at a solved model.
struct Solution {
    Vector beta;             ///< Coefficients of the working design.
    Evaluation evaluation;   ///< Scoring state at the fitted coefficients.
};

/// @brief Computation limit for long-double comparisons near a stationary point.
long double roundoff(long double value) {
    return 64 * std::numeric_limits<long double>::epsilon() * std::max(1.0L, std::abs(value));
}

/// @brief Converts a finite working quantity to a public scalar.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("firth_logistic: result exceeds finite double range");
    return static_cast<double>(value);
}

/// @brief Evaluates the logistic probability without an overflowing exponential.
long double probability(long double eta) {
    const auto tail = std::exp(-std::abs(eta));
    const auto smaller = tail / (1 + tail);
    return eta >= 0 ? 1 - smaller : smaller;
}

/// @brief Calculates the full Fisher information and leverage-adjusted gradient.
std::optional<Evaluation> evaluate(const Problem& problem, const Vector& beta) {
    const auto n = problem.design.rows(), p = problem.design.cols();
    Matrix weighted(n, p);
    Vector probabilities(n), raw_score(n), adjustment(n);
    long double log_likelihood = 0;
    for (std::size_t i = 0; i < n; ++i) {
        auto eta = 0.0L;
        for (std::size_t j = 0; j < p; ++j)
            eta = std::fma(problem.design(i, j), beta[j], eta);
        if (!std::isfinite(eta))
            return std::nullopt;
        const auto& row = problem.observations[i];
        const auto count = row.successes + row.failures;
        const auto exp_tail = std::exp(-std::abs(eta));
        const auto smaller = exp_tail / (1 + exp_tail);
        const auto loss = std::log1p(exp_tail);
        const auto adverse = eta >= 0 ? row.failures : row.successes;
        log_likelihood -= row.weight * (count * loss + adverse * std::abs(eta));
        probabilities[i] = eta >= 0 ? 1 - smaller : smaller;
        raw_score[i] = row.weight * (eta >= 0 ? count * smaller - row.failures : row.successes - count * smaller);
        adjustment[i] = eta >= 0 ? smaller - .5L : .5L - smaller;
        const auto root = std::sqrt(row.weight * count) * std::sqrt(exp_tail) / (1 + exp_tail);
        for (std::size_t j = 0; j < p; ++j)
            weighted(i, j) = root * problem.design(i, j);
    }
    if (!std::isfinite(log_likelihood))
        return std::nullopt;
    auto factor = linalg::svd(weighted);
    if (!factor.converged || factor.rank() != p)
        return std::nullopt;
    for (const auto singular_value : factor.s)
        log_likelihood += std::log(singular_value);
    Vector score(p);
    for (std::size_t i = 0; i < n; ++i) {
        long double leverage = 0;
        for (std::size_t j = 0; j < p; ++j)
            leverage += factor.U(i, j) * factor.U(i, j);
        const auto adjusted = raw_score[i] + leverage * adjustment[i];
        for (std::size_t j = 0; j < p; ++j)
            score[j] += problem.design(i, j) * adjusted;
    }
    return Evaluation{std::move(probabilities), std::move(score), std::move(factor), log_likelihood};
}

/// @brief Returns the inverse-information element in working coordinates.
long double inverse_entry(const Factor& factor, std::size_t row, std::size_t column) {
    long double value = 0;
    for (std::size_t j = 0; j < factor.s.size(); ++j)
        value += (factor.V(row, j) / factor.s[j]) * (factor.V(column, j) / factor.s[j]);
    return value;
}

/// @brief Initializes inverse curvature from the information for the free coefficients.
Matrix scoring_metric(const Evaluation& evaluation, std::size_t fixed) {
    const auto& factor = evaluation.factor;
    const auto p = factor.s.size();
    Matrix metric(p, p);
    Vector column(p);
    long double variance = 1;
    if (fixed < p) {
        for (std::size_t j = 0; j < p; ++j)
            column[j] = inverse_entry(factor, j, fixed);
        variance = column[fixed];
    }
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k <= j; ++k)
            if (j != fixed && k != fixed)
                metric(j, k) = metric(k, j) = inverse_entry(factor, j, k) - (column[j] / variance) * column[k];
    return metric;
}

/// @brief Updates the inverse curvature using the gradient change of the minimized negative log likelihood.
void update_metric(Matrix& metric, const Vector& old_beta, const Vector& new_beta,
                   const Vector& old_score, const Vector& new_score, std::size_t fixed) {
    const auto p = old_beta.size();
    Vector shift(p), change(p);
    long double curvature = 0, shift_norm = 0, change_norm = 0;
    for (std::size_t j = 0; j < p; ++j) {
        if (j == fixed)
            continue;
        shift[j] = new_beta[j] - old_beta[j];
        change[j] = old_score[j] - new_score[j];
        curvature += shift[j] * change[j];
        shift_norm = std::hypot(shift_norm, shift[j]);
        change_norm = std::hypot(change_norm, change[j]);
    }
    if (!(curvature > 64 * std::numeric_limits<long double>::epsilon() * shift_norm * change_norm))
        return;
    const auto projected = metric * change;
    long double quadratic = 0;
    for (std::size_t j = 0; j < p; ++j)
        quadratic += projected[j] * change[j];
    const auto coefficient = 1 + quadratic / curvature;
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k <= j; ++k) {
            const auto change_jk = shift[j] / curvature * (coefficient * shift[k] - projected[k])
                                 - projected[j] * (shift[k] / curvature);
            metric(k, j) = metric(j, k) = metric(j, k) + change_jk;
        }
}

/// @brief Maximum adjusted score in units of each coefficient's marginal information.
long double score_norm(const Evaluation& evaluation, std::size_t fixed) {
    const auto& factor = evaluation.factor;
    long double norm = 0;
    for (std::size_t j = 0; j < factor.s.size(); ++j) {
        if (j == fixed)
            continue;
        long double information = 0;
        for (std::size_t k = 0; k < factor.s.size(); ++k) {
            const auto value = factor.V(j, k) * factor.s[k];
            information += value * value;
        }
        norm = std::max(norm, std::abs(evaluation.score[j]) / std::sqrt(information));
    }
    return norm;
}

/// @brief Maximizes the penalized likelihood by safeguarded, Fisher-initialized BFGS.
Solution maximize(const Problem& problem, Vector beta, Evaluation current, std::size_t fixed,
                  const FirthLogisticOptions& options) {
    const auto p = beta.size();
    auto metric = scoring_metric(current, fixed);
    for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        const auto norm = score_norm(current, fixed);
        if (norm <= options.tolerance)
            return {std::move(beta), std::move(current)};
        bool accepted = false;
        for (int restart = 0; restart < 2 && !accepted; ++restart) {
            if (restart > 0)
                metric = scoring_metric(current, fixed);
            const auto direction = metric * current.score;
            long double predicted = 0, effect = 0;
            for (std::size_t j = 0; j < p; ++j)
                predicted += direction[j] * current.score[j];
            for (const auto shift : problem.design * direction)
                effect = std::max(effect, std::abs(shift));
            long double scale = effect > 5 ? 5 / effect : 1;
            for (int j = 0; j < 64 && predicted > 0 && std::isfinite(predicted); ++j, scale /= 2) {
                auto candidate = beta;
                for (std::size_t k = 0; k < p; ++k)
                    candidate[k] += scale * direction[k];
                if (candidate == beta)
                    break;
                auto trial = evaluate(problem, candidate);
                if (trial && trial->value - current.value >= 1e-4L * scale * predicted - roundoff(current.value)) {
                    update_metric(metric, beta, candidate, current.score, trial->score, fixed);
                    current = std::move(*trial);
                    beta = std::move(candidate);
                    accepted = true;
                    break;
                }
            }
        }
        if (!accepted) {
            const auto numerical_floor = 32 * std::sqrt(std::numeric_limits<long double>::epsilon());
            if (norm <= std::max(static_cast<long double>(options.tolerance), numerical_floor))
                return {std::move(beta), std::move(current)};
            throw std::runtime_error("firth_logistic: adjusted scoring could not find a stationary point");
        }
    }
    if (score_norm(current, fixed) <= options.tolerance)
        return {std::move(beta), std::move(current)};
    throw std::runtime_error("firth_logistic: iteration budget exhausted");
}

/// @brief A profile evaluation found a superior starting point for the unrestricted fit.
struct HigherProfile {
    Solution fit; ///< Constrained solution from which to restart the unrestricted fit.
};

/// @brief Inverts the profile of each coefficient while retaining the full information penalty.
std::vector<ProfileLikelihoodInterval> profile_intervals(const Problem& problem, const Solution& solution,
                                                        const Vector& scales, const FirthLogisticOptions& options) {
    const auto p = solution.beta.size();
    const auto& information = solution.evaluation.factor;
    const auto baseline = solution.evaluation.value;
    std::vector<ProfileLikelihoodInterval> result;
    for (std::size_t j = 0; j < p; ++j) {
        const auto estimate = output(solution.beta[j]);
        const auto variance = inverse_entry(information, j, j);
        Vector slope(p);
        for (std::size_t k = 0; k < p; ++k)
            slope[k] = inverse_entry(information, k, j) / variance;
        /// @brief Reoptimizes nuisance coefficients with the full Jeffreys penalty still in force.
        const auto profile = [&](double value) {
            if (value == estimate)
                return 0.0;
            auto beta = solution.beta;
            const auto change = static_cast<long double>(value) - beta[j];
            for (std::size_t k = 0; k < p; ++k)
                beta[k] += change * slope[k];
            beta[j] = value;
            auto current = evaluate(problem, beta);
            if (!current && p > 1) {
                beta.assign(p, 0);
                beta[j] = value;
                current = evaluate(problem, beta);
            }
            if (!current) {
                if (p == 1)
                    return -std::numeric_limits<double>::infinity();
                throw std::runtime_error("firth_logistic: cannot initialize a nuisance fit for this profile value");
            }
            auto restricted = maximize(problem, std::move(beta), std::move(*current), j, options);
            const auto difference = restricted.evaluation.value - baseline;
            const auto tolerance = std::max(roundoff(baseline), 32.0L * options.tolerance * options.tolerance);
            if (difference > tolerance)
                throw HigherProfile{std::move(restricted)};
            if (difference < -std::numeric_limits<double>::max())
                return -std::numeric_limits<double>::infinity();
            return output(std::min(0.0L, difference));
        };
        auto interval = profile_likelihood_interval(estimate, profile, {
            .confidence_level = options.confidence_level, .initial_step = output(std::sqrt(variance)),
        });
        interval.lower.value = output(static_cast<long double>(interval.lower.value) / scales[j]);
        interval.upper.value = output(static_cast<long double>(interval.upper.value) / scales[j]);
        result.push_back(interval);
    }
    return result;
}

} // namespace

std::vector<double> FirthLogisticResult::predict(const linalg::DenseMatrix<double>& design) const {
    const auto finite = [](double value) { return std::isfinite(value); };
    if (design.cols() != coefficients.size() || !std::ranges::all_of(design.data(), finite) ||
        !std::ranges::all_of(coefficients, finite))
        throw std::invalid_argument("firth_logistic: invalid prediction design or coefficients");
    std::vector<double> result(design.rows());
    for (std::size_t i = 0; i < design.rows(); ++i) {
        long double eta = 0;
        for (std::size_t j = 0; j < design.cols(); ++j)
            eta = std::fma(static_cast<long double>(design(i, j)), coefficients[j], eta);
        if (!std::isfinite(eta))
            throw std::overflow_error("firth_logistic: nonfinite query linear predictor");
        result[i] = static_cast<double>(probability(eta));
    }
    return result;
}

FirthLogisticResult firth_logistic(const linalg::DenseMatrix<double>& design, const std::vector<double>& successes,
                                 const FirthLogisticOptions& options) {
    const auto n = design.rows(), p = design.cols();
    const auto finite = [](double value) { return std::isfinite(value); };
    if (p == 0 || n < p || successes.size() != n || !std::ranges::all_of(design.data(), finite) ||
        (!options.trials.empty() && options.trials.size() != n) ||
        (!options.weights.empty() && options.weights.size() != n) || options.max_iterations == 0 ||
        !(options.tolerance > 0 && options.tolerance < 1) ||
        !(options.confidence_level > 0 && options.confidence_level < 1))
        throw std::invalid_argument("firth_logistic: invalid model shape or options");
    Problem problem{Matrix(n, p), {}};
    Vector scales(p);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i)
            scales[j] = std::max(scales[j], std::abs(static_cast<long double>(design(i, j))));
        if (scales[j] == 0)
            throw std::invalid_argument("firth_logistic: design is rank deficient");
        for (std::size_t i = 0; i < n; ++i)
            problem.design(i, j) = design(i, j) / scales[j];
    }
    problem.observations.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto count = options.trials.empty() ? 1.0 : options.trials[i];
        const auto weight = options.weights.empty() ? 1.0 : options.weights[i];
        const auto observed = successes[i];
        if (!(count > 0) || !finite(count) || count != std::floor(count) ||
            !(observed >= 0 && observed <= count) || observed != std::floor(observed) ||
            !(weight > 0) || !finite(weight))
            throw std::invalid_argument("firth_logistic: invalid counts or weights");
        problem.observations.push_back({observed, static_cast<long double>(count) - observed, weight});
    }
    Vector initial(p);
    auto starting = evaluate(problem, initial);
    if (!starting)
        throw std::invalid_argument("firth_logistic: design is numerically rank deficient at initialization");
    auto solution = maximize(problem, std::move(initial), std::move(*starting), p, options);
    std::vector<ProfileLikelihoodInterval> intervals;
    if (options.profile_intervals) {
        for (int attempt = 0;; ++attempt) {
            try {
                intervals = profile_intervals(problem, solution, scales, options);
                break;
            } catch (HigherProfile& higher) {
                if (attempt == 8)
                    throw std::runtime_error("firth_logistic: profiling repeatedly found a higher likelihood");
                solution = maximize(problem, std::move(higher.fit.beta), std::move(higher.fit.evaluation), p, options);
            }
        }
    }
    const auto& information = solution.evaluation.factor;
    const auto baseline = solution.evaluation.value;
    long double log_likelihood = baseline;
    for (auto scale : scales)
        log_likelihood += std::log(scale);
    FirthLogisticResult result{{}, linalg::DenseMatrix<double>(p, p), {}, output(log_likelihood), std::move(intervals)};
    for (std::size_t j = 0; j < p; ++j) {
        result.coefficients.push_back(output(solution.beta[j] / scales[j]));
        for (std::size_t k = 0; k <= j; ++k)
            result.covariance(j, k) = result.covariance(k, j) = output(inverse_entry(information, j, k) / scales[j] / scales[k]);
    }
    for (auto value : solution.evaluation.probabilities)
        result.fitted.push_back(output(value));
    return result;
}

} // namespace datamunge::stats
