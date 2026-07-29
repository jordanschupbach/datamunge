#pragma once

#include <cstdint>

namespace datamunge::algorithms {

/// @brief Result of the extended Euclidean algorithm: the greatest common divisor of two integers
///        together with a pair of *Bezout coefficients* that witness it. The three fields satisfy
///        the identity a*x + b*y == gcd, where a and b are the inputs handed to extended_gcd.
struct ExtendedGcd {
    /// @brief The (non-negative) greatest common divisor of a and b. By convention gcd(0, 0) = 0.
    std::int64_t gcd;
    /// @brief Bezout coefficient of a: a*x + b*y == gcd.
    std::int64_t x;
    /// @brief Bezout coefficient of b: a*x + b*y == gcd.
    std::int64_t y;
};

/// @brief The extended Euclidean algorithm. Ordinary Euclid finds gcd(a, b) by replacing the
///        larger argument with its remainder modulo the smaller and repeating until one becomes
///        zero; the surviving value is the gcd. The *extended* version additionally tracks, for
///        each running remainder r, a pair (s, t) with s*a + t*b == r. When the remainder reaches
///        zero the previous remainder is the gcd and its (s, t) are the Bezout coefficients, so
///        the returned triple satisfies a*x + b*y == gcd. The whole thing runs in
///        O(log min(|a|, |b|)) arithmetic operations.
///
///        Signs and zeros are handled as follows: the returned gcd is always non-negative
///        (truncating division can transiently produce a negative gcd when an input is negative,
///        which is normalised away by negating all three terms). gcd(0, 0) == 0 with x == y == 0;
///        gcd(0, b) == |b|; negative inputs are accepted and the Bezout identity still holds.
///
/// @param a first integer.
/// @param b second integer.
/// @return the gcd of a and b together with Bezout coefficients x, y such that a*x + b*y == gcd.
ExtendedGcd extended_gcd(std::int64_t a, std::int64_t b);

/// @brief The modular multiplicative inverse of @p a modulo @p m: the unique residue in [0, m)
///        whose product with a is congruent to 1 modulo m. It exists iff gcd(a, m) == 1, in which
///        case extended_gcd yields x with a*x + m*y == 1, hence a*x == 1 (mod m); reducing x into
///        [0, m) gives the inverse. Negative or large a are accepted (they are reduced modulo m).
///
/// @param a the value to invert.
/// @param m the modulus.
/// @return a^{-1} mod m as a representative in [0, m).
/// @throws std::invalid_argument if m <= 0, or if gcd(a, m) != 1 (a is not invertible modulo m).
std::int64_t modular_inverse(std::int64_t a, std::int64_t m);

} // namespace datamunge::algorithms
