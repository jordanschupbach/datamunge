#pragma once

#include <cstdint>

namespace datamunge::algorithms {

/// @brief *Pollard's rho algorithm for discrete logarithms*: solves @c g^x ≡ h (mod p) in
///        *expected* @c O(sqrt(order)) time and @c O(1) space -- the low-memory counterpart to
///        baby-step giant-step. It runs a pseudo-random walk over group elements while tracking each
///        element as @c g^a·h^b, and Floyd cycle detection finds a collision @c g^{a_1}h^{b_1} =
///        g^{a_2}h^{b_2}; the exponents then give @c (a_1-a_2) ≡ x·(b_2-b_1) (mod order), a linear
///        congruence solved for @c x. It is the discrete-log analogue of Pollard's rho factorization.
///
/// @param g the base (a generator of a subgroup of order @p order modulo @p p).
/// @param h the target, assumed to be a power of @p g.
/// @param p the prime modulus.
/// @param order the order of @p g (e.g. @c p-1 for a primitive root).
/// @return an @c x in @c [0,order) with @c g^x ≡ h (mod p), or @c -1 if the walk failed to resolve it.
long long pollard_rho_log(std::uint64_t g, std::uint64_t h, std::uint64_t p, std::uint64_t order);

/// @brief The *Pohlig-Hellman algorithm* for discrete logarithms: when the group order is *smooth*
///        (factors into small primes), it solves @c g^x ≡ h (mod p) by solving a small discrete log in
///        each prime-power subgroup and reassembling @c x with the Chinese Remainder Theorem. If
///        @c order = ∏ q_i^{e_i}, each subgroup log costs @c O(e_i·sqrt(q_i)), so a smooth order makes
///        the whole discrete log easy -- which is why cryptographic groups are chosen to have order
///        divisible by a large prime.
///
/// @param g the base of order @p order.
/// @param h the target (a power of @p g).
/// @param p the prime modulus.
/// @param order the order of @p g.
/// @return an @c x in @c [0,order) with @c g^x ≡ h (mod p), or @c -1 on failure.
long long pohlig_hellman_log(std::uint64_t g, std::uint64_t h, std::uint64_t p, std::uint64_t order);

/// @brief The *Baillie-PSW primality test*: combines a *strong (Miller-Rabin) probable-prime test to
///        base 2* with a *strong Lucas probable-prime test* (Selfridge's parameters). Individually each
///        has infinitely many pseudoprimes, but the two families are believed disjoint: *no composite
///        is known to pass both*, and none exists below @c 2^64. So although Baillie-PSW is technically
///        a probabilistic test, it is *deterministic in practice* for 64-bit integers and is what many
///        libraries use as their primality check.
///
/// @param n the integer to test.
/// @return true if @p n is a (Baillie-PSW) probable prime, false if it is definitely composite.
bool baillie_psw(std::uint64_t n);

} // namespace datamunge::algorithms
