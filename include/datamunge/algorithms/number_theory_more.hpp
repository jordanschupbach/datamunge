#pragma once
#include <datamunge/algorithms/discrete_log.hpp>

/// @file number_theory_more.hpp
/// @brief Discrete logarithms (Pollard rho, Pohlig-Hellman), Lucas sequences, Lucas probable-prime
///        tests and the Baillie-PSW primality test.

#include <cstdint>

namespace datamunge::algorithms {

/// @brief The *Baillie-PSW primality test*: combines a *strong (Miller-Rabin) probable-prime test to
///        base 2* with a *strong Lucas probable-prime test* (Selfridge's parameters, strong_lucas_probable_prime()). Individually each
///        has infinitely many pseudoprimes, but the two families are believed disjoint: *no composite
///        is known to pass both*, and none exists below @c 2^64. So although Baillie-PSW is technically
///        a probabilistic test, it is *deterministic in practice* for 64-bit integers and is what many
///        libraries use as their primality check.
///
/// @param n the integer to test.
/// @return true if @p n is a (Baillie-PSW) probable prime, false if it is definitely composite.
bool baillie_psw(std::uint64_t n);

/// @brief The Jacobi symbol \f$(a/n)\f$ for odd \f$n>0\f$: the product of Legendre symbols over the
///        prime factors of @p n; 0 when \f$\gcd(a,n)>1\f$.
/// @throws std::invalid_argument if @p n is even or zero.
int jacobi_symbol(long long a, std::uint64_t n);

/// Selfridge's "method A" parameters for a Lucas test of @p n.
struct SelfridgeParameters {
    long long D              = 0;      ///< first of 5, -7, 9, -11, ... with Jacobi symbol (D/n) = -1.
    long long P              = 1;      ///< always 1.
    long long Q              = 0;      ///< (1 - D) / 4.
    bool      perfect_square = false;  ///< true when n is a square (no such D exists); D is then 0.
};

/// @brief Selfridge's parameters for the odd number @p n > 1: the first D in 5, -7, 9, -11, ... with
///        \f$(D/n)=-1\f$, P = 1 and Q = (1 - D)/4.
/// @throws std::invalid_argument if @p n is even or less than 3.
SelfridgeParameters selfridge_parameters(std::uint64_t n);

/// Terms of a Lucas sequence pair modulo n.
struct LucasTerms {
    std::uint64_t U  = 0;  ///< \f$U_k \bmod n\f$.
    std::uint64_t V  = 0;  ///< \f$V_k \bmod n\f$.
    std::uint64_t Qk = 0;  ///< \f$Q^k \bmod n\f$.
};

/// @brief \f$U_k\f$, \f$V_k\f$ and \f$Q^k\f$ modulo the odd number @p n for the Lucas sequences with
///        parameters P, Q: \f$U_0=0, U_1=1, V_0=2, V_1=P\f$ and \f$X_{j+1}=P X_j-Q X_{j-1}\f$.
///
/// Uses \f$O(\log k)\f$ doublings \f$U_{2j}=U_jV_j\f$, \f$V_{2j}=V_j^2-2Q^j\f$ and increments
/// \f$U_{j+1}=(PU_j+V_j)/2\f$, \f$V_{j+1}=(DU_j+PV_j)/2\f$ with \f$D=P^2-4Q\f$, so @p n must be odd.
/// @throws std::invalid_argument if @p n is even.
LucasTerms lucas_sequence_mod(std::uint64_t k, long long P, long long Q, std::uint64_t n);

/// @brief The (standard) Lucas probable-prime test with Selfridge's parameters: @p n passes when
///        \f$U_{n+1}\equiv0 \pmod n\f$, which holds for every prime with \f$(D/n)=-1\f$.
bool lucas_probable_prime(std::uint64_t n);

/// @brief The strong Lucas probable-prime test with Selfridge's parameters: writing
///        \f$n+1=d\cdot2^s\f$ with d odd, @p n passes when \f$U_d\equiv0\f$ or
///        \f$V_{d2^r}\equiv0 \pmod n\f$ for some \f$0\le r<s\f$. Perfect squares fail.
bool strong_lucas_probable_prime(std::uint64_t n);

} // namespace datamunge::algorithms
