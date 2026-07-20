#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace datamunge::algebra {

namespace detail {
[[nodiscard]] inline std::int64_t to_integer(double v) { return static_cast<std::int64_t>(std::llround(v)); }
} // namespace detail

/// @brief (base^exponent) mod modulus via binary exponentiation. All three arguments and the
///        result are doubles holding exact integer values -- this module avoids std::size_t
///        and other unsigned fixed-width types in binding-facing signatures, so plain `double`
///        (exact up to 2^53) is used for modular-integer arithmetic throughout.
/// @throws std::invalid_argument if modulus <= 0 or exponent < 0.
[[nodiscard]] inline double mod_pow(double base, double exponent, double modulus) {
    std::int64_t b = detail::to_integer(base);
    std::int64_t e = detail::to_integer(exponent);
    const std::int64_t m = detail::to_integer(modulus);
    if (m <= 0) throw std::invalid_argument("modulus must be positive");
    if (e < 0) throw std::invalid_argument("exponent must be nonnegative");

    b %= m;
    if (b < 0) b += m;
    std::int64_t result = 1 % m;
    while (e > 0) {
        if (e & 1) result = (result * b) % m;
        b = (b * b) % m;
        e >>= 1;
    }
    return static_cast<double>(result);
}

/// @brief Modular multiplicative inverse of a mod m, via the extended Euclidean algorithm.
/// @throws std::invalid_argument if gcd(a, m) != 1 (no inverse exists) or m <= 0.
[[nodiscard]] inline double mod_inverse(double a, double m) {
    const std::int64_t modulus = detail::to_integer(m);
    if (modulus <= 0) throw std::invalid_argument("modulus must be positive");

    std::int64_t r0 = modulus;
    std::int64_t r1 = detail::to_integer(a) % modulus;
    if (r1 < 0) r1 += modulus;
    std::int64_t s0 = 0, s1 = 1;
    while (r1 != 0) {
        const std::int64_t q = r0 / r1;
        const std::int64_t r2 = r0 - q * r1;
        r0 = r1;
        r1 = r2;
        const std::int64_t s2 = s0 - q * s1;
        s0 = s1;
        s1 = s2;
    }
    if (r0 != 1) throw std::invalid_argument("a has no inverse mod m (gcd(a, m) != 1)");

    std::int64_t inv = s0 % modulus;
    if (inv < 0) inv += modulus;
    return static_cast<double>(inv);
}

/// @brief Chinese Remainder Theorem: the unique x in [0, product(moduli)) such that
///        `x == remainders[i] (mod moduli[i])` for every i, given pairwise-coprime positive
///        moduli.
/// @throws std::invalid_argument if sizes disagree, are empty, a modulus is non-positive, the
///         moduli aren't pairwise coprime, or the combined modulus would exceed 2^53 (past the
///         point a double can represent every integer exactly).
[[nodiscard]] inline double crt(const std::vector<double>& remainders, const std::vector<double>& moduli) {
    if (remainders.size() != moduli.size() || remainders.empty())
        throw std::invalid_argument("remainders/moduli must be the same nonzero size");

    constexpr std::int64_t kMaxExactModulus = std::int64_t{1} << 53;

    std::int64_t m = detail::to_integer(moduli[0]);
    if (m <= 0) throw std::invalid_argument("moduli must be positive");
    std::int64_t x = detail::to_integer(remainders[0]) % m;
    if (x < 0) x += m;

    for (std::size_t i = 1; i < remainders.size(); ++i) {
        const std::int64_t mi = detail::to_integer(moduli[i]);
        if (mi <= 0) throw std::invalid_argument("moduli must be positive");
        if (m > kMaxExactModulus / mi) throw std::invalid_argument("combined modulus exceeds exact double range (2^53)");
        const std::int64_t new_m = m * mi;

        std::int64_t ri = detail::to_integer(remainders[i]) % mi;
        if (ri < 0) ri += mi;

        const std::int64_t m_mod_mi = ((m % mi) + mi) % mi;
        const std::int64_t inv = detail::to_integer(mod_inverse(static_cast<double>(m_mod_mi), static_cast<double>(mi)));
        const std::int64_t diff = ((ri - x) % mi + mi) % mi;
        const std::int64_t t = (diff * inv) % mi;

        x = x + m * t;
        m = new_m;
        x %= m;
        if (x < 0) x += m;
    }
    return static_cast<double>(x);
}

} // namespace datamunge::algebra
