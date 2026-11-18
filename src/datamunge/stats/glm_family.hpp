#pragma once

/// @file glm_family.hpp
/// @brief Shared GLM family arithmetic for unpenalized and penalized fitting.

#include <datamunge/stats/glm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

/// @brief Internal statistics fitting helpers.
namespace datamunge::stats::detail {

/// @brief Maps a valid mean to its fixed link scale.
/// @param f Response family.
/// @param mu Response-scale mean.
/// @return Linear predictor.
inline double glm_link(GLMFamily f, double mu) {
  switch (f) {
    case GLMFamily::Gaussian:
      return mu;
    case GLMFamily::Binomial:
      return std::log(mu / (1.0 - mu));
    case GLMFamily::Poisson:
      return std::log(mu);
    case GLMFamily::Gamma:
      return 1.0 / mu;
  }
  return mu;
}

/// @brief Inverse fixed link, with epsilon-bounded binomial probabilities.
/// @param f Response family.
/// @param eta Linear predictor.
/// @return Response-scale mean.
inline double glm_inverse_link(GLMFamily f, double eta) {
  switch (f) {
    case GLMFamily::Gaussian:
      return eta;
    case GLMFamily::Binomial: {
      const double e = std::exp(-std::fabs(eta));
      const double p = eta >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
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
      const double a = (y > 0.0) ? y * std::log(y / mu) : 0.0;
      const double b = (y < 1.0) ? (1.0 - y) * std::log((1.0 - y) / (1.0 - mu)) : 0.0;
      return 2.0 * (a + b);
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
