/// @file likelihood_glmm.cpp
/// @brief Adaptive group integrals and their derivatives for a single Gaussian random intercept.

#include <datamunge/stats/likelihood_glmm.hpp>

#include "../random/count_likelihood.hpp"
#include "likelihood_bfgs.hpp"
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Conditional modes, quadrature, and optimization working coordinates.
namespace {
/// @brief Working vector.
using Vector = std::vector<long double>;
/// @brief Working matrix.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Fit outcome alias.
using Status = LikelihoodGLMMStatus;

/// @brief Quadrature node and weight for the normalized exp(-x*x) measure.
struct Node {
    long double x;          ///< Hermite node.
    long double log_weight; ///< Logarithm of the normalized weight.
};

/// @brief Compiled data for repeated marginal likelihood evaluations.
struct Problem {
    Matrix x;                                    ///< Scaled fixed design.
    const std::vector<std::size_t>& counts;      ///< Success or count values.
    Vector trials;                               ///< Number of binomial trials per row.
    Vector offset;                               ///< Fixed offsets on the linear predictor.
    Vector constants;                            ///< Binomial log combinations.
    std::vector<std::vector<std::size_t>> rows;   ///< Observed groups in label order.
    std::vector<Node> rule;                       ///< Hermite rule including the order-one case.
    bool poisson;                                ///< Selects Poisson in place of binomial.
};

/// @brief Row log mass and its derivatives with respect to the link predictor.
struct Row {
    long double value;    ///< Log probability of the observed count.
    long double residual; ///< First derivative of the log mass.
    long double weight;   ///< Negative second derivative.
    long double change;   ///< Derivative of the weight.
};

/// @brief Standard-normal mode and negative log posterior curvature in one group.
struct Mode {
    long double u;         ///< Conditional mode of the standardized random intercept.
    long double curvature; ///< Curvature at u, including the prior.
};

/// @brief Average log likelihood and its working-parameter score.
struct Evaluation {
    long double value{-std::numeric_limits<long double>::infinity()}; ///< Per-observation marginal log likelihood.
    Vector score;                                                     ///< Derivatives of the adaptive approximation.
};

/// @brief One fitted candidate, including its optimization outcome.
struct Candidate {
    Vector beta;          ///< Working fixed coefficients and optional log SD.
    Evaluation evaluated; ///< Objective and score at this candidate.
    Status status;        ///< Optimization status before computing covariance.
    bool random;           ///< Includes a positive random-intercept SD.
};

/// @brief Converts a working number to a finite public output.
double output(long double x) {
    if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<double>::max())
        throw std::overflow_error("likelihood GLMM: result exceeds finite reporting range");
    return static_cast<double>(x);
}

/// @brief Inverse logit evaluated from the smaller tail exponential.
long double logistic(long double eta) {
    const auto z = std::exp(-std::abs(eta));
    return (eta >= 0 ? 1 : z) / (1 + z);
}

/// @brief Constructs a Gauss–Hermite rule from the Hermite Jacobi matrix.
std::vector<Node> quadrature(std::size_t points) {
    if (points == 1)
        return {{0, 0}};
    Matrix jacobi(points, points);
    for (std::size_t j = 1; j < points; ++j)
        jacobi(j, j - 1) = jacobi(j - 1, j) = std::sqrt(j / 2.0L);
    const auto eigen = linalg::symmetric_eigen(jacobi);
    std::vector<Node> result;
    for (std::size_t j = 0; j < points; ++j) {
        const auto weight = std::abs(eigen.eigenvectors(0, j));
        if (!(weight > 0) || !std::isfinite(weight) || !std::isfinite(eigen.eigenvalues[j]))
            throw std::runtime_error("likelihood GLMM: invalid Gauss–Hermite rule");
        result.push_back({eigen.eigenvalues[j], 2 * std::log(weight)});
    }
    return result;
}

/// @brief Evaluates an observed outcome and its link-scale likelihood derivatives.
Row evaluate_row(const Problem& problem, std::size_t i, long double eta) {
    const long double y = problem.counts[i];
    if (!std::isfinite(eta))
        return {-std::numeric_limits<long double>::infinity(), -eta, 0, 0};
    if (problem.poisson) {
        const auto mean = std::exp(eta);
        const auto value = std::isfinite(mean) ? random::detail::poisson_log_mass(y, eta, mean)
                                              : -std::numeric_limits<long double>::infinity();
        return {value, y - mean, mean, mean};
    }
    const auto e = std::exp(-std::abs(eta)), a = std::log1p(e);
    const auto p = (eta >= 0 ? 1 : e) / (1 + e), q = (eta >= 0 ? e : 1) / (1 + e);
    const auto failures = problem.trials[i] - y;
    const auto weight = problem.trials[i] * p * q;
    return {problem.constants[i] + y * (std::min(0.0L, eta) - a) + failures * (std::min(0.0L, -eta) - a),
            y*q - failures*p, weight, weight * (q - p)};
}

/// @brief Concave score and information along the standardized random intercept.
Mode conditional_mode(const Problem& problem, const Vector& eta, const std::vector<std::size_t>& rows,
                      long double sigma) {
    /// @brief Computes the group score and negative second derivative.
    const auto evaluate = [&](long double u) {
        long double residual = 0, weight = 0;
        for (const auto i : rows) {
            const auto row = evaluate_row(problem, i, eta[i] + sigma * u);
            residual += row.residual;
            weight += row.weight;
        }
        return std::pair{sigma * residual - u, 1 + sigma * sigma * weight};
    };
    auto [score, curvature] = evaluate(0);
    if (score == 0 && std::isfinite(curvature))
        return {0, curvature};
    if (std::isnan(score))
        return {0, 0};
    long double lower = 0, upper = 0;
    if (score > 0) {
        upper = 1;
        for (int j = 0; j < 200 && evaluate(upper).first > 0; ++j)
            upper *= 2;
        if (!(evaluate(upper).first <= 0))
            return {0, 0};
    } else {
        lower = -1;
        for (int j = 0; j < 200 && evaluate(lower).first < 0; ++j)
            lower *= 2;
        if (!(evaluate(lower).first >= 0))
            return {0, 0};
    }
    long double u = 0;
    for (int j = 0; j < 256; ++j) {
        const auto [s, h] = evaluate(u);
        const auto change = s / h;
        const auto margin = 32 * std::numeric_limits<long double>::epsilon() *
            std::max({1.0L, std::abs(u), 1 / sigma});
        if (std::isfinite(h) && (s == 0 || std::abs(change) <= margin / std::max(1.0L, sigma)))
            return {u, h};
        if (s > 0)
            lower = u;
        else
            upper = u;
        const auto trial = u + change;
        const auto next = trial > lower && trial < upper ? trial : std::midpoint(lower, upper);
        if (next == u)
            return {u, std::isfinite(h) ? h : 0};
        u = next;
    }
    return {0, 0};
}

/// @brief Adds one group's adaptive integral and full score, differentiating mode and scale.
bool integrate_group(const Problem& problem, const Vector& eta, const std::vector<std::size_t>& rows,
                     long double sigma, Evaluation& result) {
    const auto mode = conditional_mode(problem, eta, rows, sigma);
    if (!(mode.curvature > 0))
        return false;
    const auto p = problem.x.cols();
    const auto variance = sigma * sigma;
    Vector mode_score(p + 1), log_curvature_score(p + 1);
    long double sum_weight = 0, sum_change = 0;
    for (const auto i : rows) {
        const auto row = evaluate_row(problem, i, eta[i] + sigma * mode.u);
        sum_weight += row.weight;
        sum_change += row.change;
        for (std::size_t j = 0; j < p; ++j) {
            mode_score[j] -= sigma * row.weight * problem.x(i, j);
            log_curvature_score[j] += row.change * problem.x(i, j);
        }
    }
    for (std::size_t j = 0; j < p; ++j) {
        mode_score[j] /= mode.curvature;
        log_curvature_score[j] = (variance / mode.curvature) *
            (log_curvature_score[j] + sigma * mode_score[j] * sum_change);
    }
    mode_score[p] = mode.u * (1 - variance * sum_weight) / mode.curvature;
    log_curvature_score[p] = (variance / mode.curvature) *
        (2 * sum_weight + sigma * (mode.u + mode_score[p]) * sum_change);

    const auto spread = std::sqrt(2 / mode.curvature);
    auto peak = -std::numeric_limits<long double>::infinity();
    long double sum = 0;
    Vector score(p + 1);
    for (const auto& rule : problem.rule) {
        const auto shift = spread * rule.x, u = mode.u + shift;
        auto value = rule.log_weight + rule.x * rule.x - .5L * u * u;
        Vector node_score(p + 1);
        long double residual = 0;
        for (const auto i : rows) {
            const auto row = evaluate_row(problem, i, eta[i] + sigma * u);
            value += row.value;
            residual += row.residual;
            for (std::size_t j = 0; j < p; ++j)
                node_score[j] += row.residual * problem.x(i, j);
        }
        if (!std::isfinite(value))
            continue;
        node_score[p] = sigma * u * residual;
        const auto derivative_u = sigma * residual - u;
        for (std::size_t j = 0; j <= p; ++j)
            node_score[j] += derivative_u * (mode_score[j] - .5L * shift * log_curvature_score[j]);
        if (value > peak) {
            const auto rescale = std::exp(peak - value);
            sum *= rescale;
            for (auto& s : score)
                s *= rescale;
            peak = value;
        }
        const auto weight = std::exp(value - peak);
        sum += weight;
        for (std::size_t j = 0; j <= p; ++j)
            score[j] += weight * node_score[j];
    }
    if (!(sum > 0))
        return false;
    result.value += peak + std::log(sum) - .5L * std::log(mode.curvature);
    for (std::size_t j = 0; j <= p; ++j)
        result.score[j] += score[j] / sum - .5L * log_curvature_score[j];
    return true;
}

/// @brief Forms fixed-effect predictors with their known offsets.
Vector linear_predictor(const Problem& problem, const Vector& beta) {
    auto eta = problem.offset;
    for (std::size_t i = 0; i < problem.x.rows(); ++i)
        for (std::size_t j = 0; j < problem.x.cols(); ++j)
            eta[i] = std::fma(problem.x(i, j), beta[j], eta[i]);
    return eta;
}

/// @brief Evaluates the fixed-effect boundary or integrated likelihood.
Evaluation evaluate(const Problem& problem, const Vector& beta, bool random) {
    const auto eta = linear_predictor(problem, beta);
    Evaluation result{0, Vector(beta.size())};
    if (random) {
        const auto sigma = std::exp(beta.back());
        if (!(sigma > 0) || !std::isfinite(sigma * sigma))
            return {};
        for (const auto& group : problem.rows)
            if (!integrate_group(problem, eta, group, sigma, result))
                return {};
    } else {
        for (std::size_t i = 0; i < eta.size(); ++i) {
            const auto row = evaluate_row(problem, i, eta[i]);
            result.value += row.value;
            for (std::size_t j = 0; j < beta.size(); ++j)
                result.score[j] += row.residual * problem.x(i, j);
        }
    }
    if (!std::isfinite(result.value) || !std::ranges::all_of(result.score, [](auto x) { return std::isfinite(x); }))
        return {};
    result.value /= eta.size();
    for (auto& x : result.score)
        x /= eta.size();
    return result;
}

/// @brief Maximizes either the adaptive random-intercept or fixed-only likelihood.
Candidate maximize(const Problem& problem, Vector beta, bool random, const LikelihoodGLMMOptions& options) {
    /// @brief Evaluates the selected likelihood and score at the trial coordinates.
    const auto profile = [&](const Vector& trial) { return evaluate(problem, trial, random); };
    auto fit = detail::maximize_likelihood<Status>(std::move(beta), options.max_iterations, options.tolerance, profile);
    return {std::move(fit.parameters), std::move(fit.state), fit.status, random};
}

/// @brief Inverts the observed information and transforms the log SD to SD.
linalg::DenseMatrix<double> covariance(const Problem& problem, const Candidate& fit, const Vector& scales) {
    const auto d = fit.beta.size();
    Matrix jacobian(d, d);
    for (std::size_t j = 0; j < d; ++j) {
        auto beta = fit.beta;
        const auto h = .0002L * std::max(1.0L, std::abs(beta[j]));
        /// @brief Re-evaluates one point of the fourth-order score stencil.
        const auto at = [&](long double v) {
            beta[j] = fit.beta[j] + v;
            return evaluate(problem, beta, fit.random).score;
        };
        const auto a = at(-2*h), b = at(-h), c = at(h), e = at(2*h);
        if (a.empty() || b.empty() || c.empty() || e.empty())
            return {};
        for (std::size_t i = 0; i < d; ++i)
            jacobian(i, j) = ((e[i] - a[i]) - 8 * (c[i] - b[i])) / (12*h);
    }
    Matrix correlation(d, d);
    Vector coordinate(d);
    for (std::size_t i = 0; i < d; ++i) {
        if (!(jacobian(i, i) > 0) || !std::isfinite(jacobian(i, i)))
            return {};
        coordinate[i] = std::sqrt(jacobian(i, i));
        correlation(i, i) = 1;
        for (std::size_t j = 0; j < i; ++j) {
            correlation(i, j) = .5L * (jacobian(i, j) + jacobian(j, i)) / coordinate[i] / coordinate[j];
            if (!std::isfinite(correlation(i, j)))
                return {};
        }
    }
    const auto factor = linalg::cholesky(correlation);
    if (!factor.ok)
        return {};
    for (std::size_t i = 0; i < d; ++i)
        if (!(factor.L(i, i) > 1e-7L))
            return {};
    const auto inverse = factor.solve(Matrix::identity(d));
    Vector restore(d);
    for (std::size_t j = 0; j < scales.size(); ++j)
        restore[j] = 1 / scales[j] / coordinate[j];
    if (fit.random)
        restore.back() = std::exp(fit.beta.back()) / coordinate.back();
    linalg::DenseMatrix<double> result(d, d);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            result(i, j) = result(j, i) = output(inverse(i, j) * restore[i] * restore[j] / problem.x.rows());
    return result;
}

} // namespace

std::vector<double> LikelihoodGLMMFit::predict(const linalg::DenseMatrix<double>& design,
    const std::vector<std::size_t>& groups, const std::vector<double>& offset) const {
    const auto n = design.rows();
    const auto finite = [](double x) { return std::isfinite(x); };
    if (design.cols() != coefficients.size() || (!groups.empty() && groups.size() != n) ||
        (!offset.empty() && offset.size() != n) || group_levels.size() != group_modes.size() ||
        !std::ranges::is_sorted(group_levels) || !std::ranges::all_of(group_modes, finite) ||
        !std::ranges::all_of(coefficients, finite) || !std::ranges::all_of(design.data(), finite) ||
        !std::ranges::all_of(offset, finite) || (family != GLMMFamily::Binomial && family != GLMMFamily::Poisson))
        throw std::invalid_argument("likelihood GLMM: invalid prediction input");
    std::vector<double> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        long double eta = offset.empty() ? 0 : offset[i];
        for (std::size_t j = 0; j < design.cols(); ++j)
            eta = std::fma(static_cast<long double>(design(i, j)), coefficients[j], eta);
        if (!groups.empty()) {
            const auto level = std::ranges::lower_bound(group_levels, groups[i]);
            if (level != group_levels.end() && *level == groups[i])
                eta += group_modes[level - group_levels.begin()];
        }
        result[i] = output(family == GLMMFamily::Poisson ? std::exp(eta) : logistic(eta));
    }
    return result;
}

LikelihoodGLMMFit fit_likelihood_glmm(const linalg::DenseMatrix<double>& design,
    const std::vector<std::size_t>& counts, const std::vector<std::size_t>& groups, const LikelihoodGLMMOptions& options) {
    const auto n = design.rows(), p = design.cols();
    const bool poisson = options.family == GLMMFamily::Poisson;
    const auto finite = [](double x) { return std::isfinite(x); };
    if (p == 0 || n < p || n != counts.size() || n != groups.size() ||
        (!poisson && options.family != GLMMFamily::Binomial) ||
        (!options.trials.empty() && (poisson || options.trials.size() != n)) ||
        (!options.offset.empty() && options.offset.size() != n) ||
        options.quadrature_points == 0 || options.quadrature_points > 48 || options.max_iterations == 0 ||
        !(options.tolerance > 0 && options.tolerance < 1) ||
        !std::ranges::all_of(design.data(), finite) || !std::ranges::all_of(options.offset, finite))
        throw std::invalid_argument("likelihood GLMM: invalid designs, family or controls");
    const auto capacity = Vector{}.max_size();
    if (p >= capacity || p + 1 > capacity / (p + 1))
        throw std::length_error("likelihood GLMM: coefficient information exceeds storage capacity");
    std::map<std::size_t, std::vector<std::size_t>> group_rows;
    Problem problem{Matrix(n, p), counts, Vector(n, 1), Vector(n), Vector(n), {},
                    quadrature(options.quadrature_points), poisson};
    Vector rough(n), scales(p);
    bool positive = false, failure = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (!options.offset.empty())
            problem.offset[i] = options.offset[i];
        const auto trials = options.trials.empty() ? 1 : options.trials[i];
        if (!poisson && (trials == 0 || counts[i] > trials))
            throw std::invalid_argument("likelihood GLMM: binomial successes exceed trials or trials are zero");
        positive |= counts[i] > 0;
        failure |= counts[i] < trials;
        if (poisson) {
            rough[i] = std::log(counts[i] + .5L) - problem.offset[i];
        } else {
            problem.trials[i] = trials;
            const auto other = trials - counts[i];
            rough[i] = std::log(counts[i] + .5L) - std::log(other + .5L) - problem.offset[i];
            const long double small = std::min(counts[i], other), large = std::max(counts[i], other);
            problem.constants[i] = random::detail::gamma_increment(large + 1, small) - std::lgamma(small + 1);
        }
        group_rows[groups[i]].push_back(i);
    }
    if (!positive || (!poisson && !failure) || group_rows.size() < 2)
        throw std::invalid_argument("likelihood GLMM: need outcome variation and at least two groups");

    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i)
            scales[j] = std::max(scales[j], std::abs(static_cast<long double>(design(i, j))));
        if (scales[j] == 0)
            throw std::invalid_argument("likelihood GLMM: zero fixed design column");
        for (std::size_t i = 0; i < n; ++i)
            problem.x(i, j) = design(i, j) / scales[j];
    }
    const auto factor = linalg::svd(problem.x);
    if (!factor.converged || factor.rank() != p)
        throw std::invalid_argument("likelihood GLMM: rank-deficient fixed design");
    LikelihoodGLMMFit result{options.family, {}, 0, {}, {}, {}, 0, {}, Status::NumericalFailure};
    for (auto& [label, rows] : group_rows) {
        result.group_levels.push_back(label);
        problem.rows.push_back(std::move(rows));
    }

    auto fixed = maximize(problem, factor.solve(rough), false, options);
    if (fixed.evaluated.score.empty())
        fixed = maximize(problem, Vector(p), false, options);
    auto best = fixed;
    for (const long double log_sd : {-.7L, .4L, 1.25L}) {
        auto start = fixed.beta;
        start.push_back(log_sd);
        auto trial = maximize(problem, std::move(start), true, options);
        const auto gap = trial.evaluated.value - best.evaluated.value;
        const auto near = 1e-12L * (1 + std::abs(trial.evaluated.value));
        if (!trial.evaluated.score.empty() && (best.evaluated.score.empty() || gap > near ||
            (std::abs(gap) <= near && best.status != Status::Converged && trial.status == Status::Converged)))
            best = std::move(trial);
    }
    if (best.evaluated.score.empty())
        throw std::runtime_error("likelihood GLMM: no usable fixed or random-effect candidate");

    const auto sigma = best.random ? std::exp(best.beta.back()) : 0;
    result.random_intercept_sd = output(sigma);
    result.log_likelihood = output(best.evaluated.value * n);
    for (std::size_t j = 0; j < p; ++j)
        result.coefficients.push_back(output(best.beta[j] / scales[j]));
    result.group_modes.resize(problem.rows.size());
    if (best.random) {
        const auto eta = linear_predictor(problem, best.beta);
        for (std::size_t j = 0; j < problem.rows.size(); ++j)
            result.group_modes[j] = output(sigma * conditional_mode(problem, eta, problem.rows[j], sigma).u);
    }
    result.status = best.status;
    if (best.status == Status::Converged) {
        result.covariance = covariance(problem, best, scales);
        if (result.covariance.rows() == 0)
            result.status = Status::NumericalFailure;
        else if (!best.random)
            result.status = Status::ZeroVariance;
    }
    result.fitted = result.predict(design, groups, options.offset);
    return result;
}

} // namespace datamunge::stats
