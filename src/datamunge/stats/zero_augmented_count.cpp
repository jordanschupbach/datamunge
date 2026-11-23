/// @file zero_augmented_count.cpp
/// @brief Joint mixture and zero-truncated likelihoods with deterministic starts.

#include <datamunge/stats/zero_augmented_count.hpp>

#include "../random/count_likelihood.hpp"
#include "likelihood_bfgs.hpp"
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Working coordinates and joint likelihood for the two zero mechanisms.
namespace {
/// @brief Extended-precision matrix type.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Extended-precision vector type.
using Vector = std::vector<long double>;
/// @brief Model outcome alias.
using Status = ZeroAugmentedCountStatus;
/// @brief Zero mechanism alias.
using Model = ZeroAugmentedCountModel;

/// @brief Count and zero inputs in working coefficient coordinates.
struct Problem {
    Matrix x;                              ///< Scaled count design.
    Matrix z;                              ///< Scaled zero design.
    const std::vector<std::size_t>& counts; ///< Observed counts.
    Vector offset;                         ///< Fixed log exposure.
    Vector weights;                        ///< Case weights normalized to sum to one.
    Model model;                            ///< Zero mechanism being fitted.
};

/// @brief Specifies which coefficient blocks are present in a candidate.
struct Layout {
    bool zero;       ///< Estimate the logistic zero component.
    bool dispersion; ///< Estimate the log dispersion.
};

/// @brief Weighted log likelihood and its analytic or numerical score terms.
struct Evaluation {
    long double value; ///< Average weighted log likelihood.
    Vector score;      ///< Derivatives in coefficient order.
};

/// @brief One model candidate and its optimization outcome.
struct Fit {
    Vector beta;            ///< Working coefficients and optional log dispersion.
    Evaluation evaluation;  ///< Likelihood at beta.
    Layout layout;           ///< Present coefficient blocks.
    Status status;           ///< Optimization outcome before inference.
};

/// @brief Converts a finite working scalar to a public result.
double output(long double x) {
    if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<double>::max())
        throw std::overflow_error("zero-augmented count: result exceeds finite double range");
    return static_cast<double>(x);
}

/// @brief Log logistic probability with stable tails.
long double log_sigmoid(long double eta) {
    return std::min(0.0L, eta) - std::log1p(std::exp(-std::abs(eta)));
}

/// @brief Log of the sum of two positive masses.
long double log_add(long double a, long double b) {
    return std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
}

/// @brief Log mass of a zero in the ordinary count component.
long double log_zero(long double mu, long double alpha) {
    const auto product = alpha * mu;
    return product == 0 ? -mu : -std::log1p(product) / alpha;
}

/// @brief Log mass of a count, with a truncation normalization when needed.
long double count_log_likelihood(std::size_t y, long double eta, long double mu, long double alpha, Model model) {
    if (y == 0)
        return log_zero(mu, alpha);
    const auto mass = alpha == 0 ? random::detail::poisson_log_mass(y, eta, mu)
        : random::detail::negative_binomial_log_mass(y, mu, 1 / alpha, alpha * mu);
    return model == Model::Hurdle ? mass - std::log(-std::expm1(log_zero(mu, alpha))) : mass;
}

/// @brief Evaluates the observed likelihood with posterior mixture responsibilities.
Evaluation evaluate(const Problem& problem, const Vector& beta, Layout layout) {
    const auto p = problem.x.cols(), q = layout.zero ? problem.z.cols() : 0;
    const auto alpha = layout.dispersion ? std::exp(beta.back()) : 0;
    const auto invalid = -std::numeric_limits<long double>::infinity();
    if (layout.dispersion && (!(alpha > 0) || !std::isfinite(1 / alpha) || alpha > std::numeric_limits<double>::max()))
        return {invalid, {}};
    Evaluation result{0, Vector(beta.size())};
    constexpr long double h = .0001L;
    const auto alpha_lo = alpha * std::exp(-h), alpha_hi = alpha * std::exp(h);
    const auto alpha_outer_lo = alpha * std::exp(-2 * h), alpha_outer_hi = alpha * std::exp(2 * h);
    for (std::size_t i = 0; i < problem.x.rows(); ++i) {
        const auto y = problem.counts[i];
        const auto weight = problem.weights[i];
        long double gamma = 0;
        for (std::size_t j = 0; j < q; ++j)
            gamma = std::fma(problem.z(i, j), beta[p + j], gamma);
        const auto log_pi = q > 0 ? log_sigmoid(gamma) : invalid;
        const auto log_other = q > 0 ? log_sigmoid(-gamma) : 0;
        const auto pi = std::exp(log_pi);
        long double zero_score = -pi;
        if (y == 0 && problem.model == Model::Hurdle) {
            result.value += weight * log_pi;
            for (std::size_t j = 0; j < q; ++j)
                result.score[p + j] += weight * std::exp(log_other) * problem.z(i, j);
            continue;
        }
        long double eta = problem.offset[i];
        for (std::size_t j = 0; j < p; ++j)
            eta = std::fma(problem.x(i, j), beta[j], eta);
        const auto mu = std::exp(eta);
        if (!(mu > 0) || !std::isfinite(mu) || !std::isfinite(alpha * mu))
            return {invalid, {}};
        const auto count = count_log_likelihood(y, eta, mu, alpha, problem.model);
        auto likelihood = log_other + count;
        long double responsibility = 1;
        if (y == 0 && q > 0) {
            const auto both = log_add(log_pi, likelihood);
            responsibility = std::exp(likelihood - both);
            zero_score = std::exp(log_pi - both) * std::exp(log_other) - responsibility * pi;
            likelihood = both;
        }
        result.value += weight * likelihood;
        const auto reciprocal = 1 / (1 + alpha * mu);
        auto count_score = (y - mu) * reciprocal;
        if (problem.model == Model::Hurdle) {
            const auto zero_mass = log_zero(mu, alpha);
            count_score -= (mu / -std::expm1(zero_mass)) * std::exp(zero_mass) * reciprocal;
        }
        for (std::size_t j = 0; j < p; ++j)
            result.score[j] += weight * responsibility * count_score * problem.x(i, j);
        for (std::size_t j = 0; j < q; ++j)
            result.score[p + j] += weight * zero_score * problem.z(i, j);
        if (layout.dispersion) {
            const auto inner = count_log_likelihood(y, eta, mu, alpha_hi, problem.model)
                             - count_log_likelihood(y, eta, mu, alpha_lo, problem.model);
            const auto outer = count_log_likelihood(y, eta, mu, alpha_outer_hi, problem.model)
                             - count_log_likelihood(y, eta, mu, alpha_outer_lo, problem.model);
            result.score.back() += weight * responsibility * (8 * inner - outer) / (12 * h);
        }
    }
    return result;
}

/// @brief Maximizes one candidate with safeguarded BFGS steps.
Fit maximize(const Problem& problem, Vector beta, Layout layout, const ZeroAugmentedCountOptions& options) {
    const auto d = beta.size();
    auto metric = Matrix::identity(d);
    auto current = evaluate(problem, beta, layout);
    const auto finite = [](long double v) { return std::isfinite(v); };
    if (!finite(current.value) || !std::ranges::all_of(current.score, finite))
        return {std::move(beta), std::move(current), layout, Status::NumericalFailure};
    for (std::size_t iter = 0;; ++iter) {
        const auto norm = std::ranges::max(current.score | std::views::transform([](long double v) { return std::abs(v); }));
        if (norm <= options.tolerance)
            return {std::move(beta), std::move(current), layout, Status::Converged};
        if (iter == options.max_iterations)
            break;
        bool accepted = false;
        for (unsigned restart = 0; restart < 2 && !accepted; ++restart) {
            if (restart > 0)
                metric = Matrix::identity(d);
            const auto direction = metric * current.score;
            const auto predicted = std::inner_product(direction.begin(), direction.end(), current.score.begin(), 0.0L);
            const auto effect = std::ranges::max(direction | std::views::transform([](long double v) { return std::abs(v); }));
            auto scale = effect > 4 ? 4 / effect : 1;
            const auto rounding = 32 * std::numeric_limits<long double>::epsilon() * std::max(1.0L, std::abs(current.value));
            for (unsigned trial = 0; trial < 64 && predicted > 0 && finite(predicted); ++trial, scale /= 2) {
                auto candidate = beta;
                for (std::size_t j = 0; j < d; ++j)
                    candidate[j] += scale * direction[j];
                if (candidate == beta)
                    break;
                if (!std::ranges::all_of(candidate, finite))
                    continue;
                auto next = evaluate(problem, candidate, layout);
                if (next.value - current.value >= 1e-4L * scale * predicted - rounding &&
                    std::ranges::all_of(next.score, finite)) {
                    detail::update_likelihood_metric(metric, beta, candidate, current.score, next.score);
                    beta = std::move(candidate);
                    current = std::move(next);
                    accepted = true;
                    break;
                }
            }
        }
        if (!accepted)
            return {std::move(beta), std::move(current), layout, Status::NumericalFailure};
    }
    return {std::move(beta), std::move(current), layout, Status::IterationLimit};
}

/// @brief Inverts observed information obtained by differentiating the joint score.
linalg::DenseMatrix<double> covariance(const Problem& problem, const Fit& fit, const Vector& transform,
                                       long double weight_sum) {
    const auto d = fit.beta.size();
    Matrix derivatives(d, d);
    for (std::size_t j = 0; j < d; ++j) {
        const auto h = .001L * std::max(1.0L, std::abs(fit.beta[j]));
        auto candidate = fit.beta;
        /// @brief Evaluates one score in the symmetric fourth-order stencil.
        const auto at = [&](long double delta) {
            candidate[j] = fit.beta[j] + delta;
            return evaluate(problem, candidate, fit.layout).score;
        };
        const auto a = at(-2 * h), b = at(-h), c = at(h), e = at(2 * h);
        if (a.size() != d || b.size() != d || c.size() != d || e.size() != d)
            return {};
        for (std::size_t i = 0; i < d; ++i)
            derivatives(i, j) = ((e[i] - a[i]) - 8 * (c[i] - b[i])) / (12 * h);
    }
    Matrix corr(d, d);
    Vector scale(d);
    for (std::size_t i = 0; i < d; ++i) {
        const auto diag = derivatives(i, i);
        if (!(diag > 0) || !std::isfinite(diag))
            return {};
        scale[i] = std::sqrt(diag);
        corr(i, i) = 1;
        for (std::size_t j = 0; j < i; ++j)
            corr(i, j) = ((derivatives(i, j) + derivatives(j, i)) / 2) / scale[i] / scale[j];
    }
    const auto factor = linalg::cholesky(corr);
    if (!factor.ok)
        return {};
    for (std::size_t i = 0; i < d; ++i)
        if (!(factor.L(i, i) > 1e-7L))
            return {};
    const auto inverse = factor.solve(Matrix::identity(d));
    linalg::DenseMatrix<double> result(d, d);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            result(i, j) = result(j, i) = output(inverse(i, j) * (transform[i] / scale[i]) *
                                               (transform[j] / scale[j]) / weight_sum);
    return result;
}

/// @brief Copies and scales an active design, returning column scales.
Vector scale_design(const linalg::DenseMatrix<double>& raw, Matrix& scaled,
                    const std::vector<std::size_t>& counts, bool positive_only) {
    Vector scales(raw.cols());
    for (std::size_t j = 0; j < raw.cols(); ++j) {
        for (std::size_t i = 0; i < raw.rows(); ++i)
            if (!positive_only || counts[i] > 0)
                scales[j] = std::max(scales[j], std::abs(static_cast<long double>(raw(i, j))));
        if (scales[j] == 0)
            throw std::invalid_argument("zero-augmented count: an active design column is zero");
        for (std::size_t i = 0; i < raw.rows(); ++i)
            if (!positive_only || counts[i] > 0)
                scaled(i, j) = raw(i, j) / scales[j];
    }
    return scales;
}

/// @brief Fits both zero mechanisms and selects the strongest likelihood among applicable starts and limits.
ZeroAugmentedCountResult fit_count(const linalg::DenseMatrix<double>& count_design,
    const linalg::DenseMatrix<double>& zero_design, const std::vector<std::size_t>& counts,
    const ZeroAugmentedCountOptions& options, Model model) {
    const auto n = count_design.rows(), p = count_design.cols(), q = zero_design.cols();
    const auto finite = [](double x) { return std::isfinite(x); };
    const bool nb = options.family == ZeroAugmentedCountFamily::NegativeBinomial;
    if (p == 0 || q == 0 || n < std::max(p, q) || zero_design.rows() != n || counts.size() != n ||
        (!options.offset.empty() && options.offset.size() != n) || (!options.weights.empty() && options.weights.size() != n) ||
        options.max_iterations == 0 ||
        !(options.tolerance > 0 && options.tolerance < 1) ||
        (!nb && options.family != ZeroAugmentedCountFamily::Poisson) ||
        !std::ranges::all_of(count_design.data(), finite) || !std::ranges::all_of(zero_design.data(), finite) ||
        !std::ranges::all_of(options.offset, finite) ||
        !std::ranges::all_of(options.weights, [&](double x) { return x > 0 && finite(x); }) ||
        std::ranges::all_of(counts, [](auto y) { return y == 0; }))
        throw std::invalid_argument("zero-augmented count: invalid inputs or no observed positive counts");
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (p >= limit - q || p + q + 1 > limit / (p + q + 1))
        throw std::length_error("zero-augmented count: parameter matrix exceeds capacity");
    Problem problem{Matrix(n, p), Matrix(n, q), counts, Vector(n), Vector(n, 1), model};
    const auto x_scale = scale_design(count_design, problem.x, counts, model == Model::Hurdle);
    const auto z_scale = scale_design(zero_design, problem.z, counts, false);
    const auto count_factor = linalg::svd(problem.x);
    const auto zero_factor = linalg::svd(problem.z);
    if (count_factor.rank() != p || zero_factor.rank() != q || !count_factor.converged || !zero_factor.converged)
        throw std::invalid_argument("zero-augmented count: rank-deficient design");
    long double weight_sum = 0, zero_weight = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!options.offset.empty())
            problem.offset[i] = options.offset[i];
        if (!options.weights.empty())
            problem.weights[i] = options.weights[i];
        weight_sum += problem.weights[i];
        if (counts[i] == 0)
            zero_weight += problem.weights[i];
    }
    if (!std::isfinite(weight_sum))
        throw std::overflow_error("zero-augmented count: sum of weights exceeds working range");
    for (auto& w : problem.weights)
        w /= weight_sum;
    const auto zero_fraction = zero_weight / weight_sum;
    const auto constant = zero_factor.solve(Vector(n, 1));
    const auto projected = problem.z * constant;
    const bool supports_boundary = std::ranges::all_of(projected, [](long double x) { return x > 1e-10L; });
    Vector rough(n), adjusted(n);
    for (std::size_t i = 0; i < n; ++i) {
        rough[i] = std::log(counts[i] + .5L) - problem.offset[i];
        adjusted[i] = std::log(counts[i] == 0 ? 1.5L : counts[i] + .1L) - problem.offset[i];
    }
    const auto mild = count_factor.solve(rough);
    const auto strong = count_factor.solve(adjusted);
    Fit best{{}, {-std::numeric_limits<long double>::infinity(), {}}, {false, false}, Status::NumericalFailure};
    /// @brief Compares candidates at a common likelihood scale and prefers a converged solution on a numerical tie.
    const auto consider = [&](Vector start, Layout layout) {
        auto candidate = maximize(problem, std::move(start), layout, options);
        if (!std::isfinite(candidate.evaluation.value))
            return;
        const auto difference = candidate.evaluation.value - best.evaluation.value;
        const auto tolerance = 1e-11L * (1 + std::abs(candidate.evaluation.value));
        if (difference > tolerance ||
            (std::abs(difference) <= tolerance && best.status != Status::Converged && candidate.status == Status::Converged))
            best = std::move(candidate);
    };
    if (supports_boundary && (model == Model::ZeroInflated || zero_weight == 0)) {
        consider(mild, {false, false});
        if (nb) {
            for (const double alpha : {.15, 1.5}) {
                auto start = mild;
                start.push_back(std::log(alpha));
                consider(std::move(start), {false, true});
            }
        }
    }
    if (zero_weight > 0 || !supports_boundary) {
        for (const long double fraction : {.2L, .55L, .9L}) {
            const auto pi = std::clamp(zero_fraction * (model == Model::Hurdle ? 1 : fraction), .005L, .995L);
            auto start = fraction < .5L ? mild : strong;
            const auto logit = std::log(pi) - std::log1p(-pi);
            for (auto b : constant)
                start.push_back(b * logit);
            consider(start, {true, false});
            if (nb)
                for (const double alpha : {.15, 1.5}) {
                    auto dispersed = start;
                    dispersed.push_back(std::log(alpha));
                    consider(std::move(dispersed), {true, true});
                }
        }
    }
    if (best.beta.empty())
        throw std::overflow_error("zero-augmented count: no finite starting likelihood");
    const auto alpha = best.layout.dispersion ? std::exp(best.beta.back()) : 0;
    ZeroAugmentedCountResult result{model, {}, {}, output(alpha), {}, {}, output(weight_sum * best.evaluation.value),
                                    best.status};
    Vector transform;
    for (std::size_t j = 0; j < p; ++j) {
        transform.push_back(1 / x_scale[j]);
        result.count_coefficients.push_back(output(best.beta[j] * transform.back()));
    }
    if (best.layout.zero) {
        for (std::size_t j = 0; j < q; ++j) {
            transform.push_back(1 / z_scale[j]);
            result.zero_coefficients.push_back(output(best.beta[p + j] * transform.back()));
        }
    }
    if (best.layout.dispersion)
        transform.push_back(alpha);
    if (best.status == Status::Converged) {
        result.covariance = covariance(problem, best, transform, weight_sum);
        if (result.covariance.rows() == 0)
            result.status = Status::NumericalFailure;
        else if (!best.layout.zero)
            result.status = Status::NoZeroComponent;
        else if (nb && !best.layout.dispersion)
            result.status = Status::PoissonBoundary;
    }
    result.fitted = result.predict(count_design, zero_design, options.offset);
    return result;
}
} // namespace

ZeroAugmentedCountPrediction ZeroAugmentedCountResult::predict(const linalg::DenseMatrix<double>& count_design,
    const linalg::DenseMatrix<double>& zero_design, const std::vector<double>& offset) const {
    const auto n = count_design.rows();
    const auto finite = [](double x) { return std::isfinite(x); };
    if (count_design.cols() != count_coefficients.size() || zero_design.rows() != n ||
        (!zero_coefficients.empty() && zero_design.cols() != zero_coefficients.size()) ||
        (!offset.empty() && offset.size() != n) || !(dispersion >= 0) || !finite(dispersion) ||
        (model != Model::ZeroInflated && model != Model::Hurdle) ||
        !std::ranges::all_of(count_design.data(), finite) || !std::ranges::all_of(zero_design.data(), finite) ||
        !std::ranges::all_of(count_coefficients, finite) || !std::ranges::all_of(zero_coefficients, finite) ||
        !std::ranges::all_of(offset, finite))
        throw std::invalid_argument("zero-augmented count: incompatible prediction inputs");
    ZeroAugmentedCountPrediction result;
    for (std::size_t i = 0; i < n; ++i) {
        long double eta = offset.empty() ? 0 : offset[i];
        for (std::size_t j = 0; j < count_coefficients.size(); ++j)
            eta = std::fma(static_cast<long double>(count_design(i, j)), count_coefficients[j], eta);
        long double gamma = 0;
        for (std::size_t j = 0; j < zero_coefficients.size(); ++j)
            gamma = std::fma(static_cast<long double>(zero_design(i, j)), zero_coefficients[j], gamma);
        if (!std::isfinite(eta) || !std::isfinite(gamma))
            throw std::overflow_error("zero-augmented count: predictor exceeds working range");
        const auto pi = zero_coefficients.empty() ? 0.0L : std::exp(log_sigmoid(gamma));
        const auto other = zero_coefficients.empty() ? 1.0L : std::exp(log_sigmoid(-gamma));
        const auto mu = std::exp(eta);
        const auto ordinary_zero = log_zero(mu, dispersion);
        const auto truncated_mean = mu == 0 ? 1 : mu / -std::expm1(ordinary_zero);
        result.count_mean.push_back(output(mu));
        result.zero_component_probability.push_back(output(pi));
        result.zero_probability.push_back(output(model == Model::Hurdle ? pi : pi + other * std::exp(ordinary_zero)));
        result.mean.push_back(output(other * (model == Model::Hurdle ? truncated_mean : mu)));
    }
    return result;
}

ZeroAugmentedCountResult zero_inflated_count_regression(
    const linalg::DenseMatrix<double>& count_design, const linalg::DenseMatrix<double>& zero_design,
    const std::vector<std::size_t>& counts, const ZeroAugmentedCountOptions& options) {
    return fit_count(count_design, zero_design, counts, options, Model::ZeroInflated);
}

ZeroAugmentedCountResult hurdle_count_regression(
    const linalg::DenseMatrix<double>& count_design, const linalg::DenseMatrix<double>& zero_design,
    const std::vector<std::size_t>& counts, const ZeroAugmentedCountOptions& options) {
    return fit_count(count_design, zero_design, counts, options, Model::Hurdle);
}
} // namespace datamunge::stats
