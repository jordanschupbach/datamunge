#pragma once

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief Pollard's rho integer factorization with Brent's cycle-detection improvement (Pollard
///        1975; Brent 1980). Given a composite @p n it returns a single *nontrivial* divisor
///        @f$d@f$ with @f$1 < d < n@f$ and @f$n \bmod d = 0@f$, in expected @f$O(n^{1/4})@f$ time
///        rather than the @f$O(\sqrt{n})@f$ of trial division. The method iterates the map
///        @f$g(x) = (x^2 + c) \bmod n@f$; considered modulo an unknown prime factor @f$p \mid n@f$
///        the sequence is a functional-graph iteration over @f$\mathbb{Z}_p@f$ and so, by the
///        birthday paradox, collides after only @f$\approx\sqrt{p} \le n^{1/4}@f$ steps. A
///        collision @f$x_i \equiv x_j \pmod p@f$ makes @f$p \mid |x_i - x_j|@f$ while typically
///        @f$n \nmid |x_i - x_j|@f$, so @f$\gcd(|x_i - x_j|,\, n)@f$ reveals @f$p@f$. Brent's
///        variant of the same rho cycle detection (see @c brent_cycle_detection) advances a single
///        pointer in blocks of doubling length and batches the differences into one product before
///        taking a gcd, cutting the number of modular gcds. Products use a 128-bit intermediate
///        (@c unsigned @c __int128), so the routine is overflow-safe for @p n up to @f$2^{63}@f$.
///
///        The constant @f$c@f$ is drawn deterministically from the fixed sequence
///        @f$c = 1, 2, 3, \dots@f$, retrying on the (rare) failure where a run exposes only the
///        trivial gcd @f$n@f$; results are therefore fully reproducible. Even @p n short-circuit to
///        the factor @c 2. As a documented convenience, if @p n is prime the function returns
///        @p n unchanged (it has no nontrivial factor), and @p n in @f$\{0,1\}@f$ return @p n.
///
/// @param n the number to split; intended to be a composite greater than 1.
/// @return a nontrivial divisor of @p n when @p n is composite; @p n itself when @p n is prime
///         or less than 2.
[[nodiscard]] std::uint64_t pollard_rho_factor(std::uint64_t n);

/// @brief Full prime factorization of @p n, ascending and with multiplicity. Small even factors
///        are peeled off directly; the remainder is split recursively by testing primality with an
///        inline deterministic Miller-Rabin (exact for all 64-bit inputs) and, when composite,
///        cutting @p n at a factor from @c pollard_rho_factor and recursing on both parts. The
///        returned vector multiplies back to @p n exactly, every entry is prime, and the entries
///        are sorted.
///
/// @param n the number to factor.
/// @return the multiset of prime factors of @p n in nondecreasing order; empty when @p n is 1.
/// @throws std::invalid_argument if @p n is 0 (zero has no prime factorization).
[[nodiscard]] std::vector<std::uint64_t> factorize(std::uint64_t n);

} // namespace datamunge::algorithms
