#pragma once

#include <cstdint>

namespace datamunge::algorithms {

/// @brief The Miller-Rabin primality test, made *deterministic* for every 64-bit integer. A prime
///        modulus @f$n@f$ has, by Fermat's little theorem, @f$a^{n-1}\equiv1\pmod n@f$ for every
///        base @f$a@f$ coprime to it -- but so do the Carmichael numbers, which is why the plain
///        Fermat test fails. Miller-Rabin strengthens it: writing @f$n-1 = d\cdot2^{s}@f$ with
///        @f$d@f$ odd, a prime @f$n@f$ forces the sequence @f$a^{d}, a^{2d}, \dots, a^{2^{s-1}d}@f$
///        to reach @f$1@f$ only *through* @f$\pm1@f$, since modulo a prime the only square roots of
///        @f$1@f$ are @f$\pm1@f$. Concretely @f$a^{d}\equiv1@f$, or @f$a^{2^{r}d}\equiv n-1@f$ for
///        some @f$r\in[0,s)@f$; a base @f$a@f$ for which neither holds is a *witness* proving @f$n@f$
///        composite. Every composite has many witnesses, and it is a theorem (Jaeschke; Sinclair)
///        that the twelve smallest primes @f$\{2,3,5,7,11,13,17,19,23,29,31,37\}@f$ already witness
///        every composite below @f$3.3\times10^{24}@f$ -- comfortably past @f$2^{64}@f$. Testing
///        exactly that fixed set therefore decides primality with *certainty* for any
///        @c std::uint64_t, in @f$O(\log^{3} n)@f$-ish time (12 modular exponentiations).
///
///        The name keeps the conventional @c is_probable_prime, but note that over the whole
///        @c std::uint64_t range this implementation is not probabilistic: it never errs.
///
/// @param n the number to test.
/// @return true iff @p n is prime. Correctly returns false for n < 2, for even n > 2, and -- unlike
///         the Fermat test -- for every Carmichael number (561, 1105, 1729, ...).
[[nodiscard]] bool is_probable_prime(std::uint64_t n);

/// @brief A single Miller-Rabin round: is @p n a *strong probable prime* to base @p a? Writing
///        @f$n-1 = d\cdot2^{s}@f$, this returns true iff @f$a^{d}\equiv1\pmod n@f$ or
///        @f$a^{2^{r}d}\equiv n-1\pmod n@f$ for some @f$r\in[0,s)@f$ -- i.e. iff @p a is *not* a
///        witness against @p n. A single round can be fooled (2047 = 23*89 is a strong probable
///        prime to base 2); is_probable_prime runs the fixed twelve-base set, which cannot be.
///        Handy for the report and tests, where the trace of one round is what exposes a Carmichael
///        number.
///
/// @param n the number to test (n < 2 -> false; even n -> true only for n == 2).
/// @param a the base; reduced modulo @p n, and a base that is a multiple of @p n passes vacuously.
/// @return true iff @p n is a strong probable prime to base @p a.
[[nodiscard]] bool is_strong_probable_prime_base(std::uint64_t n, std::uint64_t a);

} // namespace datamunge::algorithms
