#pragma once

#include <vector>

namespace datamunge::algorithms {

/// @brief The *Kahan summation algorithm* (compensated summation): adds a sequence of
///        floating-point numbers while carrying a running *compensation* for the low-order bits lost
///        to rounding at each step. A naive running sum can lose @c O(n·eps) relative accuracy as
///        magnitudes vary; Kahan's correction term recovers almost all of it, giving an error bound
///        independent of @c n (essentially @c 2·eps). It is the standard way to sum many terms
///        accurately without resorting to a wider type.
///
/// @param xs the values to sum.
/// @return the compensated sum of @p xs.
double kahan_sum(const std::vector<double>& xs);

/// @brief *Newton-Raphson division*: computes @c numerator/denominator *without a divide* by first
///        finding the *reciprocal* @c 1/denominator with Newton's iteration @c x <- x·(2 - d·x) --
///        which doubles the number of correct bits each step (quadratic convergence) -- and then
///        multiplying. The denominator is scaled into @c [0.5,1) (via the exponent) so a fixed linear
///        initial guess converges in a handful of steps; the result is rescaled. This is how hardware
///        dividers and many soft-float libraries implement division.
///
/// @param numerator the dividend.
/// @param denominator the divisor (must be nonzero).
/// @return the quotient @c numerator/denominator.
double newton_raphson_division(double numerator, double denominator);

/// @brief The *nth root algorithm*: computes @c a^(1/n) for @c a >= 0 and integer @c n >= 1 with
///        Newton's method on @c x^n - a = 0, giving the iteration
///        @c x <- ((n-1)·x + a/x^(n-1)) / n. Because @c x^n is convex for @c x>0, the iteration
///        converges *quadratically* from a good start; the initial guess is taken from the binary
///        exponent of @p a so only a few iterations are needed.
///
/// @param a the radicand (@c a >= 0).
/// @param n the root degree (@c n >= 1).
/// @return the principal @c n-th root of @p a.
double nth_root(double a, int n);

/// @brief The *alpha max plus beta min algorithm*: a fast, division- and square-root-free
///        approximation of the Euclidean magnitude @c sqrt(x^2+y^2), as
///        @c alpha·max(|x|,|y|) + beta·min(|x|,|y|). With the peak-error-minimizing coefficients
///        @c alpha≈0.96043, @c beta≈0.39782 the worst-case relative error is about @c 3.96%. It is a
///        classic DSP/embedded trick for estimating vector magnitude (or complex modulus) where a true
///        @c hypot is too expensive.
///
/// @param x,y the two components.
/// @return the approximate magnitude @c ~sqrt(x^2+y^2).
double alpha_max_beta_min(double x, double y);

} // namespace datamunge::algorithms
