#pragma once

/// @file distributions.hpp
/// @brief Log densities and log mass functions for writing Bayesian models with autodiff.

#include <datamunge/autodiff/tape.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

// A small log-density/log-mass-function library for building Bayesian models: every
// distribution has a plain `double` overload (direct evaluation) and an
// `autodiff::Var`-valued overload for the model's *random variable* argument (so its
// gradient w.r.t. that argument is available via reverse-mode automatic differentiation --
// see autodiff_model.hpp). Distribution *hyperparameters* (shape, rate, degrees of freedom,
// ...) are always plain doubles here: this sidesteps ever needing a differentiable lgamma
// (whose derivative is the digamma function, not implemented in this library) while still
// covering the common case of a Var random variable under fixed-hyperparameter priors, or a
// Var linear predictor as a likelihood's location/rate parameter.

namespace datamunge::bayes {

namespace detail {
/// @brief A scalar supported by the log-density functions.
/// @tparam Real Candidate scalar type.
template <class Real>
concept LogDensityScalar = std::is_same_v<Real, double> || std::is_same_v<Real, autodiff::Var>;

constexpr double kPi = 3.14159265358979323846; ///< pi, for the log densities.
} // namespace detail

// ---- Normal ----

/// @brief Normal log density \f$-\log\sigma-\tfrac12\log2\pi-\tfrac12((x-\mu)/\sigma)^2\f$.
/// @param x Value.
/// @param mu Mean.
/// @param sigma Standard deviation (positive).
/// @return The log density.
inline double normal_lpdf(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    return -std::log(sigma) - 0.5 * std::log(2.0 * detail::kPi) - 0.5 * z * z;
}
/// @brief Normal log density of a parameter-valued random variable, e.g. a prior \f$\mu\sim N(0,5^2)\f$.
/// @param x Random variable.
/// @param mu Mean.
/// @param sigma Standard deviation (positive).
/// @return The log density as a differentiable Var.
inline autodiff::Var normal_lpdf(const autodiff::Var& x, double mu, double sigma) {
    const autodiff::Var z = (x - mu) / sigma;
    return (z * z) * (-0.5) - std::log(sigma) - 0.5 * std::log(2.0 * detail::kPi);
}
/// @brief Normal log density with parameter-valued value and mean, e.g. \f$\theta_j\sim N(\mu,\sigma^2)\f$ with known \f$\sigma\f$.
/// @param x Random variable.
/// @param mu Mean.
/// @param sigma Standard deviation (positive).
/// @return The log density as a differentiable Var.
inline autodiff::Var normal_lpdf(const autodiff::Var& x, const autodiff::Var& mu, double sigma) {
    const autodiff::Var z = (x - mu) / sigma;
    return (z * z) * (-0.5) - std::log(sigma) - 0.5 * std::log(2.0 * detail::kPi);
}
/// @brief Normal log density with a parameter-valued scale, as in a hierarchical model
///        \f$\theta_j\sim N(\mu,\tau)\f$ where \f$\tau\f$ is itself sampled.
/// @param x Random variable.
/// @param mu Location.
/// @param sigma Scale; must be positive.
/// @return \f$-\log\sigma-\tfrac12\log 2\pi-\tfrac12((x-\mu)/\sigma)^2\f$.
inline autodiff::Var normal_lpdf(const autodiff::Var& x, const autodiff::Var& mu, const autodiff::Var& sigma) {
    const autodiff::Var z = (x - mu) / sigma;
    return (z * z) * (-0.5) - autodiff::log(sigma) - 0.5 * std::log(2.0 * detail::kPi);
}
/// @brief The common regression-likelihood shape: a fixed observation x against a
///        parameter-valued mean mu (e.g. a linear predictor).
inline autodiff::Var normal_lpdf(double x, const autodiff::Var& mu, double sigma) {
    const autodiff::Var z = (mu - x) / sigma;
    return (z * z) * (-0.5) - std::log(sigma) - 0.5 * std::log(2.0 * detail::kPi);
}

// ---- Cauchy ----

/// @brief Cauchy log density \f$-\log(\pi\gamma)-\log(1+((x-x_0)/\gamma)^2)\f$. Restricted to \f$x\ge0\f$ with \f$x_0=0\f$ it is a half-Cauchy prior up to the constant \f$\log2\f$.
/// @param x Value.
/// @param x0 Location.
/// @param gamma Scale (positive).
/// @return The log density.
inline double cauchy_lpdf(double x, double x0, double gamma) {
    const double z = (x - x0) / gamma;
    return -std::log(detail::kPi * gamma) - std::log(1.0 + z * z);
}
/// @brief Cauchy log density of a parameter-valued random variable (see the double overload).
/// @param x Random variable.
/// @param x0 Location.
/// @param gamma Scale (positive).
/// @return The log density as a differentiable Var.
inline autodiff::Var cauchy_lpdf(const autodiff::Var& x, double x0, double gamma) {
    const autodiff::Var z = (x - x0) / gamma;
    return autodiff::log(z * z + 1.0) * (-1.0) - std::log(detail::kPi * gamma);
}

// ---- Laplace (double exponential) ----

/// @brief Laplace log density \f$-\log(2b)-|x-\mu|/b\f$: the error law of least-absolute-deviation
///        (median) regression. It is not differentiable at \f$x=\mu\f$.
/// @param x Value.
/// @param mu Location (the median).
/// @param b Scale (positive); the standard deviation is \f$\sqrt2\,b\f$.
/// @return The log density.
inline double laplace_lpdf(double x, double mu, double b) { return -std::log(2.0 * b) - std::abs(x - mu) / b; }
/// @brief Laplace log density with a parameter-valued location, as in a regression likelihood.
/// @param x Observation.
/// @param mu Location (e.g. a linear predictor).
/// @param b Scale (positive).
/// @return The log density as a differentiable Var (its derivative jumps at \f$x=\mu\f$).
inline autodiff::Var laplace_lpdf(double x, const autodiff::Var& mu, double b) {
    return autodiff::abs(mu - x) * (-1.0 / b) - std::log(2.0 * b);
}

// ---- Exponential ----

/// @brief Exponential log density \f$\log\lambda-\lambda x\f$ for \f$x\ge0\f$.
/// @param x Value (non-negative).
/// @param rate Rate \f$\lambda\f$ (positive).
/// @return The log density.
inline double exponential_lpdf(double x, double rate) { return std::log(rate) - rate * x; }
/// @brief Exponential log density of a parameter-valued random variable.
/// @param x Random variable (non-negative).
/// @param rate Rate (positive).
/// @return The log density as a differentiable Var.
inline autodiff::Var exponential_lpdf(const autodiff::Var& x, double rate) { return x * (-rate) + std::log(rate); }

// ---- Gamma (shape, rate) ----

/// @brief Gamma log density \f$\alpha\log\beta-\log\Gamma(\alpha)+(\alpha-1)\log x-\beta x\f$.
/// @param x Value (positive).
/// @param shape Shape \f$\alpha\f$ (positive).
/// @param rate Rate \f$\beta\f$ (positive).
/// @return The log density.
inline double gamma_lpdf(double x, double shape, double rate) {
    return shape * std::log(rate) - std::lgamma(shape) + (shape - 1.0) * std::log(x) - rate * x;
}
/// @brief Gamma log density of a parameter-valued random variable.
/// @param x Random variable (positive).
/// @param shape Shape (positive).
/// @param rate Rate (positive).
/// @return The log density as a differentiable Var.
inline autodiff::Var gamma_lpdf(const autodiff::Var& x, double shape, double rate) {
    const double const_term = shape * std::log(rate) - std::lgamma(shape);
    return autodiff::log(x) * (shape - 1.0) - x * rate + const_term;
}

// ---- Beta (alpha, beta) ----

/// @brief Beta log density \f$(\alpha-1)\log x+(\beta-1)\log(1-x)-\log B(\alpha,\beta)\f$.
/// @param x Value in (0, 1).
/// @param alpha First shape (positive).
/// @param beta Second shape (positive).
/// @return The log density.
inline double beta_lpdf(double x, double alpha, double beta) {
    const double lbeta = std::lgamma(alpha) + std::lgamma(beta) - std::lgamma(alpha + beta);
    return (alpha - 1.0) * std::log(x) + (beta - 1.0) * std::log(1.0 - x) - lbeta;
}
/// @brief Beta log density of a parameter-valued random variable.
/// @param x Random variable in (0, 1).
/// @param alpha First shape (positive).
/// @param beta Second shape (positive).
/// @return The log density as a differentiable Var.
inline autodiff::Var beta_lpdf(const autodiff::Var& x, double alpha, double beta) {
    const double lbeta = std::lgamma(alpha) + std::lgamma(beta) - std::lgamma(alpha + beta);
    return autodiff::log(x) * (alpha - 1.0) + autodiff::log(1.0 - x) * (beta - 1.0) - lbeta;
}

// ---- Student's t (degrees of freedom nu, location mu, scale sigma) ----

/// @brief Student-t log density with \f$\nu\f$ degrees of freedom, location \f$\mu\f$ and scale \f$\sigma\f$.
/// @param x Value.
/// @param nu Degrees of freedom (positive).
/// @param mu Location.
/// @param sigma Scale (positive).
/// @return The log density.
inline double student_t_lpdf(double x, double nu, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    const double const_term =
        std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0) - 0.5 * std::log(nu * detail::kPi) - std::log(sigma);
    return const_term - (nu + 1.0) / 2.0 * std::log(1.0 + z * z / nu);
}
/// @brief Student-t log density of a parameter-valued random variable.
/// @param x Random variable.
/// @param nu Degrees of freedom (positive).
/// @param mu Location.
/// @param sigma Scale (positive).
/// @return The log density as a differentiable Var.
inline autodiff::Var student_t_lpdf(const autodiff::Var& x, double nu, double mu, double sigma) {
    const double const_term =
        std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0) - 0.5 * std::log(nu * detail::kPi) - std::log(sigma);
    const autodiff::Var z = (x - mu) / sigma;
    const autodiff::Var inner = (z * z) / nu + 1.0;
    return autodiff::log(inner) * (-(nu + 1.0) / 2.0) + const_term;
}

/// @brief Computes log(1 + exp(x)) without overflowing the exponential.
///
/// @param x Real argument.
/// @return The softplus of @p x.
/// @par Example
/// @code
/// const double y = dm::bayes::softplus(1000.0); // 1000
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/real-valued_basic_functions.html
/// @see bernoulli_logit_lpmf()
inline double softplus(double x) {
    return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
}

/// @brief Differentiable softplus (same value as the scalar overload).
/// @param x Real argument on a reverse-mode tape.
/// @return Softplus with the inverse-logit local derivative.
/// @par Example
/// @code
/// dm::autodiff::Tape tape;
/// const auto y = dm::bayes::softplus(dm::autodiff::Var(tape, 1000)); // value 1000
/// @endcode
/// @see softplus(double)
inline autodiff::Var softplus(const autodiff::Var& x) {
    const double value = softplus(x.value());
    const double t = std::exp(-std::abs(x.value()));
    const double slope = (x.value() >= 0.0 ? 1.0 : t) / (1.0 + t);
    auto& tape = x.tape();
    return {tape, tape.record_unary(value, x.index(), slope), value};
}

/// @brief Adds two finite log weights and returns the natural log of their sum.
/// @tparam Real double or autodiff::Var.
/// @param a First finite log weight.
/// @param b Second finite log weight, on the same tape for Vars.
/// @return The log of exp(a) + exp(b), without exponentiating large positive weights.
/// @par Example
/// @code
/// const double z = dm::bayes::log_add_exp(1000.0, 1000.0); // 1000 + log(2)
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/real-valued_basic_functions.html
/// @see softplus()
template <detail::LogDensityScalar Real>
Real log_add_exp(const Real& a, const Real& b) {
    if constexpr (std::is_same_v<Real, autodiff::Var>) {
        return a.value() >= b.value() ? a + softplus(b - a) : b + softplus(a - b);
    } else {
        return std::max(a, b) + softplus(-std::abs(a - b));
    }
}

/// @brief Bernoulli log mass for a finite log-odds predictor.
/// @param y Outcome, 0 or 1.
/// @param logit_p Log-odds.
/// @return The normalized log probability.
/// @par Example
/// @code
/// const double lp = dm::bayes::bernoulli_logit_lpmf(1, 1000.0); // 0
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/binary_distributions.html
/// @see binomial_logit_lpmf()
inline double bernoulli_logit_lpmf(double y, double logit_p) {
    return -y * softplus(-logit_p) - (1.0 - y) * softplus(logit_p);
}

/// @brief Differentiable Bernoulli log mass with a finite log-odds predictor.
/// @param y Outcome, 0 or 1.
/// @param logit_p Log-odds on a reverse-mode tape.
/// @return The normalized log probability.
/// @par Example
/// @code
/// dm::autodiff::Tape tape;
/// const auto lp = dm::bayes::bernoulli_logit_lpmf(0, dm::autodiff::Var(tape, -1000)); // value 0
/// @endcode
/// @see softplus()
inline autodiff::Var bernoulli_logit_lpmf(double y, const autodiff::Var& logit_p) {
    return -y * softplus(-logit_p) - (1.0 - y) * softplus(logit_p);
}

// ---- Poisson, log-rate-parameterized (y observed non-negative count, log_rate the linear predictor) ----

/// @brief Poisson log mass with log-scale rate: \f$y\eta-e^\eta-\log y!\f$.
/// @param y Count (non-negative integer value).
/// @param log_rate Log rate \f$\eta\f$.
/// @return The log probability.
inline double poisson_log_lpmf(double y, double log_rate) {
    return y * log_rate - std::exp(log_rate) - std::lgamma(y + 1.0);
}
/// @brief Poisson log mass with a parameter-valued linear predictor, as in Poisson regression.
/// @param y Count.
/// @param log_rate Log rate.
/// @return The log probability as a differentiable Var.
inline autodiff::Var poisson_log_lpmf(double y, const autodiff::Var& log_rate) {
    const double const_term = -std::lgamma(y + 1.0);
    return log_rate * y - autodiff::exp(log_rate) + const_term;
}

/// @brief Binomial log mass parameterized by log-odds.
/// @tparam Real double or autodiff::Var.
/// @param successes Integer count from zero through @p trials.
/// @param trials Non-negative integer number of trials.
/// @param logit_p Finite log-odds.
/// @return The normalized log mass. Counts are assumed valid.
/// @par Example
/// @code
/// const auto lp = dm::bayes::binomial_logit_lpmf(1, 2, 0.0); // log(0.5)
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/bounded_discrete_distributions.html
/// @see bernoulli_logit_lpmf()
template <detail::LogDensityScalar Real>
Real binomial_logit_lpmf(double successes, double trials, const Real& logit_p) {
    const double failures = trials - successes;
    const double log_choose = std::lgamma(trials + 1.0) - std::lgamma(successes + 1.0) -
                              std::lgamma(failures + 1.0);
    return log_choose - successes * softplus(-logit_p) - failures * softplus(logit_p);
}

/// @brief Negative binomial (NB2) log mass with variance mean + mean squared / precision.
/// @tparam Real double or autodiff::Var.
/// @param y Non-negative integer count.
/// @param log_mean Finite log mean.
/// @param precision Positive finite shape (inverse overdispersion), fixed data.
/// @return The normalized log mass. Inputs are assumed valid.
/// @par Example
/// @code
/// const auto lp = dm::bayes::neg_binomial_2_log_lpmf(0, 0.0, 1.0); // log(0.5)
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/unbounded_discrete_distributions.html
/// @see poisson_log_lpmf()
template <detail::LogDensityScalar Real>
Real neg_binomial_2_log_lpmf(double y, const Real& log_mean, double precision) {
    const auto log_ratio = log_mean - std::log(precision);
    const double log_coefficient = std::lgamma(y + precision) - std::lgamma(precision) -
                                   std::lgamma(y + 1.0);
    return log_coefficient - precision * softplus(log_ratio) - y * softplus(-log_ratio);
}

/// @brief Gamma log density parameterized by log mean and fixed shape.
/// @tparam Real double or autodiff::Var.
/// @param y Positive finite observation.
/// @param log_mean Finite log mean.
/// @param shape Positive finite shape, fixed data.
/// @return The normalized log density, with rate shape / mean. Inputs are assumed valid.
/// @par Example
/// @code
/// const auto lp = dm::bayes::gamma_log_mean_lpdf(1, 0.0, 1.0); // -1
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/positive_continuous_distributions.html
/// @see gamma_lpdf()
template <detail::LogDensityScalar Real>
Real gamma_log_mean_lpdf(double y, const Real& log_mean, double shape) {
    using std::exp;
    const auto log_scaled_y = std::log(y) - log_mean + std::log(shape);
    return shape * log_scaled_y - exp(log_scaled_y) - std::lgamma(shape) - std::log(y);
}

/// @brief Zero-inflated Poisson log mass, marginalizing the structural-zero indicator.
/// @tparam Real double or autodiff::Var.
/// @param y Non-negative integer count.
/// @param log_rate Finite log rate in the Poisson component.
/// @param zero_logit Finite log-odds of a structural zero, on the same tape as @p log_rate for Vars.
/// @return The normalized log mass. Inputs are assumed valid.
/// @par Example
/// @code
/// const auto lp = dm::bayes::zero_inflated_poisson_log_lpmf(1, 0.0, 0.0); // -1 - log(2)
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html
/// @see poisson_log_lpmf()
template <detail::LogDensityScalar Real>
Real zero_inflated_poisson_log_lpmf(double y, const Real& log_rate, const Real& zero_logit) {
    if (y != 0.0) {
        return bernoulli_logit_lpmf(0, zero_logit) + poisson_log_lpmf(y, log_rate);
    }
    const double eta = [&] {
        if constexpr (std::is_same_v<Real, autodiff::Var>) return log_rate.value();
        else return log_rate;
    }();
    const auto structural = bernoulli_logit_lpmf(1, zero_logit);
    // If the Poisson rate overflows, its zero log mass rounds to negative infinity.
    // Avoid recording a path with an infinite local derivative.
    if (eta > std::log(std::numeric_limits<double>::max())) {
        return structural;
    }
    const auto ordinary = bernoulli_logit_lpmf(0, zero_logit) + poisson_log_lpmf(0, log_rate);
    return log_add_exp(structural, ordinary);
}

} // namespace datamunge::bayes
