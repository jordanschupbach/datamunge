#pragma once

/// @file distributions.hpp
/// @brief Log densities and log mass functions for writing Bayesian models with autodiff.

#include <datamunge/autodiff/tape.hpp>

#include <cmath>

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

// ---- Bernoulli, logit-parameterized (y observed in {0, 1}, logit_p the linear predictor) ----

/// @brief Bernoulli log mass with logit-scale probability: \f$y\eta-\log(1+e^\eta)\f$.
/// @param y Outcome, 0 or 1.
/// @param logit_p Log-odds \f$\eta\f$.
/// @return The log probability.
inline double bernoulli_logit_lpmf(double y, double logit_p) {
    return y * logit_p - std::log(1.0 + std::exp(logit_p));
}
/// @brief Bernoulli log mass with a parameter-valued linear predictor, as in logistic regression.
/// @param y Outcome, 0 or 1.
/// @param logit_p Log-odds.
/// @return The log probability as a differentiable Var.
inline autodiff::Var bernoulli_logit_lpmf(double y, const autodiff::Var& logit_p) {
    return logit_p * y - autodiff::log(autodiff::exp(logit_p) + 1.0);
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

} // namespace datamunge::bayes
