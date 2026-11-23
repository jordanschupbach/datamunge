#pragma once

/// @file glm_family.hpp
/// @brief Shared GLM family arithmetic for unpenalized and penalized fitting.

#include <datamunge/stats/glm.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

/// @brief Internal statistics fitting helpers.
namespace datamunge::stats::detail {

/// @brief Maps a valid mean to its link scale.
/// @param f Response family.
/// @param mu Response-scale mean.
/// @param binomial Link for the binomial family.
/// @return Linear predictor.
inline double glm_link(GLMFamily f, double mu, GLMBinomialLink binomial = GLMBinomialLink::Logit) {
  switch (f) {
    case GLMFamily::Gaussian:
      return mu;
    case GLMFamily::Binomial:
      if (binomial == GLMBinomialLink::Probit)
        return random::normal_quantile(mu);
      if (binomial == GLMBinomialLink::Cloglog)
        return std::log(-std::log1p(-mu));
      return std::log(mu) - std::log1p(-mu);
    case GLMFamily::Poisson:
      return std::log(mu);
    case GLMFamily::Gamma:
      return 1.0 / mu;
  }
  return mu;
}

/// @brief Inverse link, with epsilon-bounded binomial probabilities.
/// @param f Response family.
/// @param eta Linear predictor.
/// @param binomial Link for the binomial family.
/// @return Response-scale mean.
inline double glm_inverse_link(GLMFamily f, double eta, GLMBinomialLink binomial = GLMBinomialLink::Logit) {
  switch (f) {
    case GLMFamily::Gaussian:
      return eta;
    case GLMFamily::Binomial: {
      double p;
      if (binomial == GLMBinomialLink::Probit)
        p = .5 * std::erfc(-eta / std::numbers::sqrt2);
      else if (binomial == GLMBinomialLink::Cloglog)
        p = -std::expm1(-std::exp(eta));
      else {
        const double e = std::exp(-std::fabs(eta));
        p = eta >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
      }
      return std::clamp(
          p, std::numeric_limits<double>::epsilon(), 1.0 - std::numeric_limits<double>::epsilon());
    }
    case GLMFamily::Poisson:
      return std::exp(eta);
    case GLMFamily::Gamma:
      return 1.0 / eta;
  }
  return eta;
}

/// @brief Derivative of the mean with respect to the linear predictor for a selected link.
/// @param f Response family.
/// @param eta Linear predictor.
/// @param mu Response-scale mean.
/// @param binomial Link for the binomial family.
/// @return Inverse-link derivative with a machine-epsilon floor for the alternate binomial links.
inline double glm_mean_derivative(GLMFamily f, double eta, double mu, GLMBinomialLink binomial) {
  switch (f) {
    case GLMFamily::Gaussian:
      return 1;
    case GLMFamily::Binomial:
      if (binomial == GLMBinomialLink::Probit)
        return std::max(std::numeric_limits<double>::epsilon(),
                        std::exp(-.5 * eta * eta) / std::sqrt(2 * std::numbers::pi));
      if (binomial == GLMBinomialLink::Cloglog) {
        const double intensity = std::exp(eta);
        return std::isfinite(intensity)
            ? std::max(std::numeric_limits<double>::epsilon(), std::exp(eta - intensity))
            : std::numeric_limits<double>::epsilon();
      }
      return mu * (1 - mu);
    case GLMFamily::Poisson:
      return mu;
    case GLMFamily::Gamma:
      return -mu * mu;
  }
  return 1;
}

// d(eta)/d(mu) = g'(mu)
/// @brief Evaluates d eta/d mu.
/// @param f Response family.
/// @param mu Valid mean.
/// @return Link derivative.
inline double glm_link_derivative(GLMFamily f, double mu) {
  switch (f) {
    case GLMFamily::Gaussian:
      return 1.0;
    case GLMFamily::Binomial:
      return 1.0 / (mu * (1.0 - mu));
    case GLMFamily::Poisson:
      return 1.0 / mu;
    case GLMFamily::Gamma:
      return -1.0 / (mu * mu);
  }
  return 1.0;
}

/// @brief Unscaled exponential-family variance function V(mu).
/// @param f Response family.
/// @param mu Valid mean.
/// @return Unscaled variance.
inline double glm_variance(GLMFamily f, double mu) {
  switch (f) {
    case GLMFamily::Gaussian:
      return 1.0;
    case GLMFamily::Binomial:
      return mu * (1.0 - mu);
    case GLMFamily::Poisson:
      return mu;
    case GLMFamily::Gamma:
      return mu * mu;
  }
  return 1.0;
}

/// @brief Evaluates x log(x / mean) + mean - x, preserving small differences.
inline long double binomial_deviance_part(long double x, long double mean) {
  if (x == 0)
    return mean;
  if (mean == 0)
    return std::numeric_limits<long double>::infinity();
  const auto delta = x - mean;
  if (std::abs(delta) > .1L * (x + mean))
    return x * std::log(x / mean) - delta;
  const auto v = delta / (x + mean);
  const auto square = v * v;
  auto sum = delta * v;
  auto term = 2 * x * v;
  for (int k = 3; k <= 19; k += 2) {
    term *= square;
    sum += term / k;
  }
  return sum;
}

/// @brief Unweighted observation contribution to deviance.
/// @param f Response family.
/// @param y Observed response.
/// @param mu Mean in the family domain.
/// @return Unit deviance, possibly infinity at a null-model boundary.
inline double glm_unit_deviance(GLMFamily f, double y, double mu) {
  switch (f) {
    case GLMFamily::Gaussian: {
      const double d = y - mu;
      return d * d;
    }
    case GLMFamily::Binomial: {
      if (y == 0)
        return -2 * std::log1p(-mu);
      if (y == 1)
        return -2 * std::log(mu);
      const long double observed = y, expected = mu;
      return static_cast<double>(2 * (binomial_deviance_part(observed, expected) +
                                      binomial_deviance_part(1 - observed, 1 - expected)));
    }
    case GLMFamily::Poisson: {
      const double a = (y > 0.0) ? y * std::log(y / mu) : 0.0;
      return 2.0 * (a - (y - mu));
    }
    case GLMFamily::Gamma:
      if (std::isinf(mu))
        return std::numeric_limits<double>::infinity();
      return 2.0 * (-std::log(y / mu) + (y - mu) / mu);
  }
  return 0.0;
}

/// @brief Rejects nonfinite or out-of-family responses.
/// @param f Response family.
/// @param y Response value.
inline void glm_validate_response(GLMFamily f, double y) {
  if (!std::isfinite(y))
    throw std::invalid_argument("GLM: response must be finite");
  switch (f) {
    case GLMFamily::Gaussian:
      return;
    case GLMFamily::Binomial:
      if (y != 0.0 && y != 1.0)
        throw std::invalid_argument(
            "GLM: binomial family requires a response of exactly 0 or 1, found " +
            std::to_string(y));
      return;
    case GLMFamily::Poisson:
      if (y < 0.0)
        throw std::invalid_argument(
            "GLM: poisson family requires a non-negative response, found " + std::to_string(y));
      return;
    case GLMFamily::Gamma:
      if (y <= 0.0)
        throw std::invalid_argument(
            "GLM: Gamma family requires a strictly positive response, found " + std::to_string(y));
      return;
  }
}

} // namespace datamunge::stats::detail
