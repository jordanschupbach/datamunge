#pragma once

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
constexpr double kPi = 3.14159265358979323846;
} // namespace detail

// ---- Normal ----

inline double normal_lpdf(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    return -std::log(sigma) - 0.5 * std::log(2.0 * detail::kPi) - 0.5 * z * z;
}
inline autodiff::Var normal_lpdf(const autodiff::Var& x, double mu, double sigma) {
    const autodiff::Var z = (x - mu) / sigma;
    return (z * z) * (-0.5) - std::log(sigma) - 0.5 * std::log(2.0 * detail::kPi);
}
inline autodiff::Var normal_lpdf(const autodiff::Var& x, const autodiff::Var& mu, double sigma) {
    const autodiff::Var z = (x - mu) / sigma;
    return (z * z) * (-0.5) - std::log(sigma) - 0.5 * std::log(2.0 * detail::kPi);
}
/// @brief The common regression-likelihood shape: a fixed observation x against a
///        parameter-valued mean mu (e.g. a linear predictor).
inline autodiff::Var normal_lpdf(double x, const autodiff::Var& mu, double sigma) {
    const autodiff::Var z = (mu - x) / sigma;
    return (z * z) * (-0.5) - std::log(sigma) - 0.5 * std::log(2.0 * detail::kPi);
}

// ---- Cauchy ----

inline double cauchy_lpdf(double x, double x0, double gamma) {
    const double z = (x - x0) / gamma;
    return -std::log(detail::kPi * gamma) - std::log(1.0 + z * z);
}
inline autodiff::Var cauchy_lpdf(const autodiff::Var& x, double x0, double gamma) {
    const autodiff::Var z = (x - x0) / gamma;
    return autodiff::log(z * z + 1.0) * (-1.0) - std::log(detail::kPi * gamma);
}

// ---- Exponential ----

inline double exponential_lpdf(double x, double rate) { return std::log(rate) - rate * x; }
inline autodiff::Var exponential_lpdf(const autodiff::Var& x, double rate) { return x * (-rate) + std::log(rate); }

// ---- Gamma (shape, rate) ----

inline double gamma_lpdf(double x, double shape, double rate) {
    return shape * std::log(rate) - std::lgamma(shape) + (shape - 1.0) * std::log(x) - rate * x;
}
inline autodiff::Var gamma_lpdf(const autodiff::Var& x, double shape, double rate) {
    const double const_term = shape * std::log(rate) - std::lgamma(shape);
    return autodiff::log(x) * (shape - 1.0) - x * rate + const_term;
}

// ---- Beta (alpha, beta) ----

inline double beta_lpdf(double x, double alpha, double beta) {
    const double lbeta = std::lgamma(alpha) + std::lgamma(beta) - std::lgamma(alpha + beta);
    return (alpha - 1.0) * std::log(x) + (beta - 1.0) * std::log(1.0 - x) - lbeta;
}
inline autodiff::Var beta_lpdf(const autodiff::Var& x, double alpha, double beta) {
    const double lbeta = std::lgamma(alpha) + std::lgamma(beta) - std::lgamma(alpha + beta);
    return autodiff::log(x) * (alpha - 1.0) + autodiff::log(1.0 - x) * (beta - 1.0) - lbeta;
}

// ---- Student's t (degrees of freedom nu, location mu, scale sigma) ----

inline double student_t_lpdf(double x, double nu, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    const double const_term =
        std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0) - 0.5 * std::log(nu * detail::kPi) - std::log(sigma);
    return const_term - (nu + 1.0) / 2.0 * std::log(1.0 + z * z / nu);
}
inline autodiff::Var student_t_lpdf(const autodiff::Var& x, double nu, double mu, double sigma) {
    const double const_term =
        std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0) - 0.5 * std::log(nu * detail::kPi) - std::log(sigma);
    const autodiff::Var z = (x - mu) / sigma;
    const autodiff::Var inner = (z * z) / nu + 1.0;
    return autodiff::log(inner) * (-(nu + 1.0) / 2.0) + const_term;
}

// ---- Bernoulli, logit-parameterized (y observed in {0, 1}, logit_p the linear predictor) ----

inline double bernoulli_logit_lpmf(double y, double logit_p) {
    return y * logit_p - std::log(1.0 + std::exp(logit_p));
}
inline autodiff::Var bernoulli_logit_lpmf(double y, const autodiff::Var& logit_p) {
    return logit_p * y - autodiff::log(autodiff::exp(logit_p) + 1.0);
}

// ---- Poisson, log-rate-parameterized (y observed non-negative count, log_rate the linear predictor) ----

inline double poisson_log_lpmf(double y, double log_rate) {
    return y * log_rate - std::exp(log_rate) - std::lgamma(y + 1.0);
}
inline autodiff::Var poisson_log_lpmf(double y, const autodiff::Var& log_rate) {
    const double const_term = -std::lgamma(y + 1.0);
    return log_rate * y - autodiff::exp(log_rate) + const_term;
}

} // namespace datamunge::bayes
