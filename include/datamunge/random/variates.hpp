#pragma once

/// @file variates.hpp
/// @brief Gamma and chi-square random variates for samplers that need them (posterior draws, imputation).

#include <datamunge/random/random.hpp>

namespace datamunge::random {

/// @brief Draws a gamma variate with the given shape and scale (Marsaglia–Tsang).
/// @param rng Random generator.
/// @param shape Positive shape.
/// @param scale Positive scale (mean = shape * scale).
/// @return A gamma(shape, scale) draw.
/// @throws std::invalid_argument If shape or scale is not positive and finite.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(3);
/// std::println("{}", dm::random::gamma_variate(rng, 2.5, 2.0) > 0); // true
/// @endcode
/// @par References
/// - G. Marsaglia and W. W. Tsang (2000). "A simple method for generating gamma variables." ACM
///   Transactions on Mathematical Software 26(3), 363-372. doi:10.1145/358407.358414
/// @see chi_squared_variate()
[[nodiscard]] double gamma_variate(RandomGenerator& rng, double shape, double scale = 1.0);

/// @brief Draws a chi-square variate, a gamma(df / 2, 2) draw.
/// @param rng Random generator.
/// @param df Positive degrees of freedom (need not be an integer).
/// @return A chi-square(df) draw.
/// @throws std::invalid_argument If df is not positive and finite.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(3);
/// std::println("{}", dm::random::chi_squared_variate(rng, 4) > 0); // true
/// @endcode
/// @see gamma_variate()
[[nodiscard]] double chi_squared_variate(RandomGenerator& rng, double df);

} // namespace datamunge::random
