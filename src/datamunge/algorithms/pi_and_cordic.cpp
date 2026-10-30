/// @file pi_and_cordic.cpp
/// @brief BBP digit extraction, exact Machin hex digits, AGM and Chudnovsky pi, and CORDIC.

#include <datamunge/algorithms/pi_and_cordic.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// @brief Modular-exponentiation and fixed-point helpers.
namespace {

constexpr double kPi = 3.14159265358979323846; ///< pi in double precision.

/// @brief 16^e mod m by fast modular exponentiation (for the BBP digit extraction).
long long modpow16(long long e, long long m) {
    long long r = 1 % m;
    long long b = 16 % m;
    while (e > 0) {
        if (e & 1) r = (r * b) % m;
        b = (b * b) % m;
        e >>= 1;
    }
    return r;
}

/// @brief Fractional part of 16^n * sum_{k>=0} 1 / ((8k+j) 16^k): the integer-part terms are handled
// modularly (16^{n-k} mod (8k+j)) and the tail (k>n) is a fast-converging series.
double bbp_series(int j, int n) {
    double s = 0.0;
    for (int k = 0; k <= n; ++k) {
        const long long denom = 8LL * k + j;
        s += static_cast<double>(modpow16(n - k, denom)) / static_cast<double>(denom);
        s -= std::floor(s); // keep only the fractional part
    }
    for (int k = n + 1; k <= n + 24; ++k) {
        const double term = std::pow(16.0, static_cast<double>(n - k)) / static_cast<double>(8 * k + j);
        s += term;
        if (term < 1e-18) break;
    }
    return s - std::floor(s);
}

/// @brief n! in double precision (for the Chudnovsky terms).
double factorial(int n) {
    double r = 1.0;
    for (int i = 2; i <= n; ++i) r *= i;
    return r;
}

/// @brief Fixed-point number: limb 0 is the integer part, limbs 1.. the fraction in base 2^32.
using Limbs = std::vector<std::uint32_t>;

/// @brief x /= d in place, from limb `from` on (limbs before it must be zero). Truncates.
void divide_small(Limbs& x, const std::uint32_t d, const std::size_t from) {
    std::uint64_t rem = 0;
    for (std::size_t i = from; i < x.size(); ++i) {
        const std::uint64_t cur = (rem << 32) | x[i];
        x[i] = static_cast<std::uint32_t>(cur / d);
        rem = cur % d;
    }
}

/// @brief acc += sign * t, both fixed-point of the same length.
void add_signed(Limbs& acc, const Limbs& t, const bool subtract) {
    std::int64_t carry = 0;
    for (std::size_t i = acc.size(); i-- > 0;) {
        std::int64_t v = static_cast<std::int64_t>(acc[i]) + (subtract ? -static_cast<std::int64_t>(t[i]) : t[i]) + carry;
        carry = 0;
        if (v < 0) {
            v += (std::int64_t{1} << 32);
            carry = -1;
        } else if (v >= (std::int64_t{1} << 32)) {
            v -= (std::int64_t{1} << 32);
            carry = 1;
        }
        acc[i] = static_cast<std::uint32_t>(v);
    }
}

/// @brief m * arctan(1/x) in fixed point with `limbs` fraction limbs.
Limbs scaled_arctan_inverse(const std::uint32_t x, const std::uint32_t m, const std::size_t limbs) {
    Limbs sum(limbs + 1, 0), power(limbs + 1, 0), term(limbs + 1, 0);
    power[0] = m;
    divide_small(power, x, 0);                    // m / x
    const std::uint32_t x2 = x * x;
    std::size_t lead = 0;                         // first possibly non-zero limb of power
    for (std::uint32_t k = 0;; ++k) {
        while (lead < power.size() && power[lead] == 0) ++lead;
        if (lead == power.size()) break;
        std::fill(term.begin(), term.begin() + static_cast<std::ptrdiff_t>(lead), 0U);
        std::copy(power.begin() + static_cast<std::ptrdiff_t>(lead), power.end(), term.begin() + static_cast<std::ptrdiff_t>(lead));
        divide_small(term, 2 * k + 1, lead);      // m / ((2k+1) x^(2k+1))
        add_signed(sum, term, (k & 1U) != 0);
        divide_small(power, x2, lead);
    }
    return sum;
}

} // namespace

double bbp_pi_fraction(const int n) {
    if (n < 0) throw std::invalid_argument("bbp_pi_fraction: n must be non-negative");
    double x = 4.0 * bbp_series(1, n) - 2.0 * bbp_series(4, n) - bbp_series(5, n) - bbp_series(6, n);
    return x - std::floor(x);
}

std::string bbp_pi_hex_digits(const int n, const int count) {
    if (n < 0 || count < 1 || count > 12) throw std::invalid_argument("bbp_pi_hex_digits: need n >= 0 and 1 <= count <= 12");
    static constexpr char kHex[] = "0123456789ABCDEF";
    double x = bbp_pi_fraction(n);
    std::string out;
    for (int i = 0; i < count; ++i) {
        x *= 16.0;
        const int d = static_cast<int>(x);
        out += kHex[d];
        x -= d;
    }
    return out;
}

std::string pi_hex_digits(const int count) {
    if (count < 1 || count > 1000000) throw std::invalid_argument("pi_hex_digits: count must be in [1, 1000000]");
    const std::size_t limbs = static_cast<std::size_t>(count) / 8 + 4;  // 3+ guard limbs
    Limbs pi = scaled_arctan_inverse(5, 16, limbs);
    add_signed(pi, scaled_arctan_inverse(239, 4, limbs), true);
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(static_cast<std::size_t>(count));
    for (std::size_t i = 1; i < pi.size() && out.size() < static_cast<std::size_t>(count); ++i)
        for (int shift = 28; shift >= 0 && out.size() < static_cast<std::size_t>(count); shift -= 4)
            out += kHex[(pi[i] >> shift) & 0xFU];
    return out;
}

int bbp_pi_hex_digit(int n) {
    double x = 4.0 * bbp_series(1, n) - 2.0 * bbp_series(4, n) - bbp_series(5, n) - bbp_series(6, n);
    x -= std::floor(x); // reduce to [0,1)
    return static_cast<int>(x * 16.0);
}

double gauss_legendre_pi(int iterations) {
    double a = 1.0;
    double b = 1.0 / std::sqrt(2.0);
    double t = 0.25;
    double p = 1.0;
    for (int i = 0; i < iterations; ++i) {
        const double a_next = 0.5 * (a + b);
        const double b_next = std::sqrt(a * b);
        t -= p * (a - a_next) * (a - a_next);
        a = a_next;
        b = b_next;
        p *= 2.0;
    }
    return (a + b) * (a + b) / (4.0 * t);
}

double chudnovsky_pi(int terms) {
    double sum = 0.0;
    for (int k = 0; k < terms; ++k) {
        const double numerator   = factorial(6 * k) * (13591409.0 + 545140134.0 * k);
        const double denominator = factorial(3 * k) * std::pow(factorial(k), 3.0) *
                                   std::pow(640320.0, 3.0 * k + 1.5);
        double term = 12.0 * numerator / denominator;
        if (k & 1) term = -term;
        sum += term;
    }
    return 1.0 / sum;
}

SinCos cordic_sincos(double theta, int iterations) {
    // Range-reduce theta into [-pi/2, pi/2], within CORDIC's convergence radius (~1.743).
    double z    = std::remainder(theta, 2.0 * kPi); // now in [-pi, pi]
    bool   flip = false;
    if (z > kPi / 2.0) {
        z -= kPi;
        flip = true;
    } else if (z < -kPi / 2.0) {
        z += kPi;
        flip = true;
    }

    double x    = 1.0;
    double y    = 0.0;
    double gain = 1.0;
    double p2   = 1.0; // 2^{-i}
    for (int i = 0; i < iterations; ++i) {
        const double angle = std::atan(p2); // the precomputed arctangent table
        const double d     = (z >= 0.0) ? 1.0 : -1.0;
        const double nx    = x - d * y * p2; // shift-add pseudo-rotation
        const double ny    = y + d * x * p2;
        x = nx;
        y = ny;
        z -= d * angle;
        gain *= 1.0 / std::sqrt(1.0 + p2 * p2);
        p2 *= 0.5;
    }
    SinCos out{y * gain, x * gain};
    if (flip) {
        out.sin = -out.sin;
        out.cos = -out.cos;
    }
    return out;
}

} // namespace datamunge::algorithms
