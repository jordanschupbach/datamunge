#pragma once

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief *Goldschmidt division*: computes @c n/d by iteratively scaling *both* numerator and
///        denominator by the same factor @c f = 2 - d, driving the denominator toward 1 (so the
///        numerator tends to the quotient). Like Newton-Raphson division it converges *quadratically*
///        and uses only multiplies, but its two multiplications each step are *independent* and can
///        run in parallel -- which is why pipelined and superscalar hardware often prefer it. The
///        denominator is first scaled into @c (0,1] so the iteration contracts.
///
/// @param numerator the dividend.
/// @param denominator the divisor (nonzero).
/// @return the quotient @c numerator/denominator.
double goldschmidt_division(double numerator, double denominator);

/// @brief *Montgomery multiplication* (via Montgomery reduction / REDC): computes @c a·b mod n
///        *without a trial division by n*. Working in the Montgomery domain (numbers scaled by
///        @c R=2^64 mod n), a product is reduced using only multiplications, additions, and shifts by
///        @c R -- replacing the expensive @c mod by cheap bit operations. It is the workhorse behind
///        fast modular exponentiation in RSA and elliptic-curve cryptography, where millions of
///        modular multiplications are done under a fixed odd modulus.
///
/// @param a,b the operands (each @c < n).
/// @param n an odd modulus.
/// @return @c (a·b) mod n, computed by Montgomery reduction.
std::uint64_t montgomery_multiply(std::uint64_t a, std::uint64_t b, std::uint64_t n);

/// @brief *Cipolla's algorithm*: finds a *square root modulo a prime* -- an @c r with
///        @c r^2 ≡ n (mod p) -- by computing in the quadratic field @c F_{p^2}. It picks @c a so that
///        @c a^2-n is a *non-residue*, then raises @c (a+\sqrt{a^2-n}) to the power @c (p+1)/2 in
///        @c F_{p^2}; the result is a real (residue-class) square root. It is an elegant alternative to
///        Tonelli-Shanks and is often faster when @c p-1 is divisible by a large power of two.
///
/// @param n the value whose square root mod @p p is sought (@c 0 <= n < p).
/// @param p an odd prime.
/// @return an @c r with @c r^2 ≡ n (mod p) if @p n is a quadratic residue, else @c -1.
long long cipolla_sqrt(std::uint64_t n, std::uint64_t p);

/// @brief *Addition-chain exponentiation*: computes @c x^e using a short *addition chain* -- a
///        sequence @c 1=c_0,c_1,\dots,c_L=e in which each term is a sum of two earlier terms, so each
///        corresponds to one multiplication. The *shortest* chain minimizes the multiplications needed
///        for @c x^e (fewer than the standard binary square-and-multiply for many exponents, e.g.
///        @c x^{15} in 5 rather than 6). This routine returns a chain: the binary-method chain for
///        general @c e, refined by a bounded search that finds a provably shortest chain for small
///        @c e.
///
/// @param e the exponent (@c e >= 1).
/// @return an addition chain from 1 to @p e (each element the sum of two earlier ones); its length
///         minus one is the multiplication count.
std::vector<std::uint64_t> addition_chain(std::uint64_t e);

} // namespace datamunge::algorithms
