#pragma once
/// @file gamma_sampling.hpp
/// @brief Internal gamma sampling for normalized weights and matrix laws.

#include <datamunge/random/distributions.hpp>
#include <datamunge/random/random.hpp>
#include <cmath>

namespace datamunge::random::detail {

/// @brief Samples a uniform value strictly between zero and one.
/// @param rng Engine supplying 53-bit uniform values.
/// @return An open-interval uniform variate.
inline long double positive_uniform(RandomGenerator& rng) {
  double sample;
  do {
    sample = rng.uniform01();
  } while (sample == 0);
  return sample;
}

/// @brief Uses the Marsaglia–Tsang sampler for the logarithm of a unit-scale gamma draw.
/// @param rng Engine supplying uniform and normal draws.
/// @param shape Positive gamma shape.
/// @param log_scale Factor applied to the log draw to prevent overflow at tiny shapes.
/// @return Scaled log gamma variate.
inline long double gamma_draw_log(RandomGenerator& rng, long double shape, long double log_scale) {
  if (shape < 1) {
    const auto boosted = gamma_draw_log(rng, 1 + shape, log_scale);
    return boosted + (log_scale / shape) * std::log(positive_uniform(rng));
  }
  const auto d = shape - 1.L / 3;
  const auto c = (1.L / 3) / std::sqrt(d);
  for (;;) {
    const long double z = rng.normal();
    const auto w = c * z;
    if (w <= -1)
      continue;
    const auto u = rng.uniform01();
    const auto square = z * z;
    const auto delta = w * (3 + w * (3 + w));
    if (u < 1 - .0331L * square * square ||
        std::log(u) < square / 2 - d * gamma_divergence(delta))
      return log_scale * (std::log(d) + 3 * std::log1p(w));
  }
}

} // namespace datamunge::random::detail
