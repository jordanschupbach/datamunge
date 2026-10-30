#pragma once

/// @file pi_and_cordic.hpp
/// @brief Computing pi (BBP digit extraction, exact hex digits, Gauss-Legendre, Chudnovsky) and
///        CORDIC sine and cosine.

#include <string>

namespace datamunge::algorithms {

/// @brief The *Bailey-Borwein-Plouffe (BBP) formula* as a *spigot* for a single hexadecimal digit
///        of pi. The BBP series
///        @c pi = sum_k 16^{-k} ( 4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6) )
///        has the remarkable property that the @p n-th hex digit (0-indexed, the digit just after
///        the point being @c n=0) can be computed *directly*, without the preceding digits, using
///        only modular exponentiation and a short convergent tail -- so it needs no arbitrary-
///        precision arithmetic. It famously computes deep digits of pi in near-constant space.
///
/// @param n the zero-based index of the hexadecimal digit after the point (n=0 gives '2').
/// @return the hexadecimal digit value in @c [0,15].
int bbp_pi_hex_digit(int n);

/// @brief The fractional part \f$\{16^n\pi\}\f$ by BBP digit extraction, in double precision.
///
/// Its hexadecimal expansion is the digits of pi from position @p n on, so one evaluation yields
/// several digits. Each of the four series accumulates about \f$n\f$ rounding errors of order
/// \f$2^{-53}\f$, so the number of trustworthy digits falls slowly as @p n grows.
/// @param n Zero-based position of the first hex digit after the point (n = 0 gives 0.1415...·16 -> '2').
/// @return \f$\{16^n\pi\}\in[0,1)\f$.
/// @throws std::invalid_argument if @p n is negative.
double bbp_pi_fraction(int n);

/// @brief @p count hex digits of pi starting at zero-based position @p n, read off one
///        bbp_pi_fraction() evaluation (uppercase, e.g. n = 0, count = 6 gives "243F6A").
/// @param n Zero-based position of the first digit.
/// @param count Number of digits, 1 to 12 (a double holds about 13 hex digits; later ones are noise).
/// @return The digits as a string.
/// @throws std::invalid_argument if @p n is negative or @p count is outside [1, 12].
std::string bbp_pi_hex_digits(int n, int count);

/// @brief The first @p count hex digits of pi after the point, computed exactly.
///
/// Uses Machin's formula \f$\pi=16\arctan\tfrac15-4\arctan\tfrac1{239}\f$ with
/// \f$\arctan\tfrac1x=\sum_k(-1)^k/((2k+1)x^{2k+1})\f$ in fixed-point arithmetic on base-\f$2^{32}\f$
/// limbs, with three guard limbs to absorb truncation. Cost grows as \f$O(\text{count}^2)\f$:
/// tens of thousands of digits take well under a second. Intended as a reference for testing
/// digit-extraction methods.
/// @param count Number of digits (1 to 1,000,000).
/// @return Uppercase hex digits, "243F6A8885A3..." for pi = 3.243F6A8885A3...
/// @throws std::invalid_argument if @p count is outside [1, 1000000].
std::string pi_hex_digits(int count);

/// @brief The *Gauss-Legendre algorithm* for pi: an arithmetic-geometric-mean iteration that
///        *doubles* the number of correct digits each step (quadratic convergence). Starting from
///        @c a=1, @c b=1/sqrt2, @c t=1/4, @c p=1, it repeats
///        @c a'=(a+b)/2, @c b=sqrt(a b), @c t-=p(a-a')^2, @c p*=2, and returns
///        @c pi ~ (a+b)^2/(4t). In double precision three to four iterations already reach full
///        machine accuracy; more iterations need arbitrary precision to show further digits.
///
/// @param iterations the number of AGM iterations.
/// @return the estimate of pi (double precision saturates after ~4 iterations).
double gauss_legendre_pi(int iterations);

/// @brief The *Chudnovsky algorithm* for pi: sums the rapidly converging hypergeometric series
///        @c 1/pi = 12 sum_k (-1)^k (6k)! (545140134 k + 13591409) / ( (3k)! (k!)^3 640320^{3k+3/2} ),
///        each term of which adds about 14 decimal digits -- the method behind modern record pi
///        computations. Evaluated here in double precision, so a couple of terms already saturate
///        the ~15-16 significant digits a @c double can hold.
///
/// @param terms the number of series terms to sum (>= 1).
/// @return the estimate of pi.
double chudnovsky_pi(int terms);

/// @brief Sine and cosine returned together by @ref cordic_sincos.
struct SinCos {
    double sin{0.0}; ///< Sine of the angle.
    double cos{0.0}; ///< Cosine of the angle.
};

/// @brief *CORDIC* (COordinate Rotation DIgital Computer) in circular rotation mode: computes
///        @c sin and @c cos of @p theta using only *additions, subtractions, and bit shifts* plus a
///        small table of arctangents -- no multiplies or divides, which is why it powered early
///        calculators and lives in FPGA/DSP hardware. It rotates the vector @c (1/K, 0) by a sum of
///        ever-smaller angles @c atan(2^{-i}), each step choosing the sign that drives a running
///        angle accumulator toward @p theta; @c K is the fixed gain of the pseudo-rotations. The
///        input is range-reduced into @c [-pi/2, pi/2] first.
///
/// @param theta the angle in radians.
/// @param iterations the number of CORDIC rotations (more -> more accurate; ~40 for double).
/// @return the @ref SinCos of @p theta.
SinCos cordic_sincos(double theta, int iterations = 40);

} // namespace datamunge::algorithms
