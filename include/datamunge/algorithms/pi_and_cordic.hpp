#pragma once

/// @file pi_and_cordic.hpp
/// @brief Computing pi (BBP digit extraction, exact hex digits, Gauss-Legendre, Chudnovsky) and
///        CORDIC sine and cosine.

#include <string>
#include <datamunge/algorithms/chudnovsky.hpp>

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

/// @brief Legacy fixed-count double AGM pi estimate, with bounded weight use.
/// Preserves ordered arithmetic and initialization(1,1/sqrt2,1/4,1), including
/// left-associated p*delta*delta. Negative counts retain zero-stage compatibility.
/// Counts above gauss_legendre_pi_limit() are rejected before work instead of
/// returning a NaN suffix after weight overflow. Executes every admitted stage;
/// stagnation is not early stopping or a correct-rounding certificate. At the
/// maximum count the final unused p overflows, while all used weights/estimate
/// remain finite. Requires radix-two double at build time. Only one final
/// estimate expression is evaluated; O(max(0,N)) work/O(1) storage, no shared
/// mutable state. Zero stages returns the initialized approximation, not pi.
/// @param iterations Stage count; negatives act as0, positive bound is format-dependent.
/// @return Actual final rounded estimate; no last-bit or mathematical-error guarantee.
/// @throws std::invalid_argument Count exceeds the safe used-weight range.
/// @throws std::overflow_error A used arithmetic result or estimate is nonfinite.
/// @par Example
/// @code
/// std::cout << (dm::algorithms::gauss_legendre_pi(-1)==dm::algorithms::gauss_legendre_pi(0)); // 1
/// @endcode
/// @par References
/// - R. P. Brent(1976), Fast Multiple-Precision Evaluation of Elementary Functions,
///   J. ACM23(2),242–251, Section5, doi:10.1145/321941.321944,
///   https://maths-people.anu.edu.au/~brent/pd/rpb034.pdf . Fixed-count double
///   variant; the paper instead discusses precision stopping and variable precision.
/// @see gauss_legendre_pi_run(), gauss_legendre_pi_trace()
double gauss_legendre_pi(int iterations);

/// @brief Owned sine/cosine approximation pair returned by the circular CORDIC model.
/// Original fields and their order/defaults are preserved. Default storage alone
/// has no associated angle; producing calls document count and reduction limits.
/// @par Example
/// @code
/// const auto s=dm::algorithms::cordic_sincos(0,0);
/// std::cout << s.sin << ' ' << s.cos; // 0 1
/// @endcode
/// @see cordic_sincos(), cordic_sincos_trace()
struct SinCos {
    double sin{0.0}; ///< Computed sine component, default zero storage.
    double cos{0.0}; ///< Computed cosine component, default zero storage.
};

/// @brief Compute circular CORDIC sine/cosine with checked floating-model controls.
/// Preserves the original recurrence/result layout: raw initial (1,0), residual
/// sign >=0 selects +1, N dyadic pseudo-rotations, accumulated inverse gain and
/// final simultaneous sine/cosine sign restoration. Finite theta and counts in
/// [0,1075] are required; zero stages is an unrotated approximation, not an error
/// certificate. No early stop or exact-identity special cases. Uses floating
/// multiplications and cached atan/sqrt constants, not an integer shift engine.
/// Reduction uses remainder(theta,2*pi_double); large angles can have large phase
/// error against true sin(theta)/cos(theta). Theta(1+N) time/O(1) call scratch,
/// after thread-safe one-time immutable table construction. No input mutation.
/// @param theta Finite angle in radians.
/// @param iterations Completed stage count in [0,1075], default forty.
/// @return Owned compensated approximation pair, with original field layout.
/// @throws std::invalid_argument Nonfinite theta or count outside [0,1075].
/// @par Example
/// @code
/// std::cout << dm::algorithms::cordic_sincos(0,0).cos; // 1
/// @endcode
/// @par References
/// - J. E. Volder (1959), The CORDIC Trigonometric Computing Technique,
///   https://doi.org/10.1109/TEC.1959.5222693 . This API is a floating circular model.
/// @see cordic_sincos_analyze(), cordic_sincos_trace(), cordic_reduce()
SinCos cordic_sincos(double theta,int iterations=40);

} // namespace datamunge::algorithms
