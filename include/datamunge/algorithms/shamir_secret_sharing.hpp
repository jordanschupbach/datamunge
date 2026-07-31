#pragma once

/// \file shamir_secret_sharing.hpp
/// \brief Shamir's (k, n) threshold secret sharing over a prime field.
///
/// Shamir (1979) splits a secret \f$s\f$ into \f$n\f$ shares so that *any* \f$k\f$ of them
/// reconstruct \f$s\f$ while any \f$k-1\f$ reveal *nothing* about it. The idea rests on
/// polynomial interpolation over a finite field \f$\mathbb{Z}_p\f$: pick a random degree
/// \f$k-1\f$ polynomial whose constant term is the secret,
/// \f[
///   P(x) = s + a_1 x + a_2 x^2 + \dots + a_{k-1} x^{k-1} \pmod p,
/// \f]
/// and hand out the points \f$(i, P(i))\f$ for \f$i = 1,\dots,n\f$. A degree \f$k-1\f$
/// polynomial is uniquely determined by \f$k\f$ points, so any \f$k\f$ shares recover
/// \f$P\f$ (and thus \f$s = P(0)\f$) by Lagrange interpolation:
/// \f[
///   s = P(0) = \sum_{i} y_i \prod_{j\ne i} \frac{x_j}{x_j - x_i} \pmod p.
/// \f]
/// With fewer than \f$k\f$ points, every candidate secret remains equally consistent, giving
/// information-theoretic security. Used for key escrow, distributed custody, and
/// multi-party cryptography.
///
/// This implementation works over a prime field with 64-bit values (128-bit intermediate
/// products) and takes the random polynomial coefficients as an argument so results are
/// reproducible.

#include <cstdint>
#include <vector>

#include <datamunge/algorithms/diffie_hellman.hpp>  // mul_mod, mod_pow

namespace datamunge::algorithms {

/// \brief A single share: the evaluation point \p x and the polynomial value \p y = P(x).
struct SecretShare {
    std::uint64_t x;
    std::uint64_t y;
};

/// \brief Split \p secret into \p n shares with threshold \f$k = coeffs.size()+1\f$.
/// \param coeffs the random higher-degree coefficients \f$a_1,\dots,a_{k-1}\f$ (the constant
///        term is the secret). Their count sets the threshold.
/// \param prime a prime modulus larger than the secret and than \p n.
/// \returns the \p n shares \f$(i, P(i))\f$ for \f$i = 1,\dots,n\f$.
inline std::vector<SecretShare> shamir_split(std::uint64_t secret, unsigned n,
                                             const std::vector<std::uint64_t>& coeffs,
                                             std::uint64_t prime) {
    std::vector<SecretShare> shares;
    shares.reserve(n);
    for (unsigned i = 1; i <= n; ++i) {
        std::uint64_t x = i;
        std::uint64_t y = secret % prime;
        std::uint64_t x_pow = 1;  // x^0
        for (std::uint64_t a : coeffs) {
            x_pow = mul_mod(x_pow, x, prime);      // x^degree
            y = (y + mul_mod(a % prime, x_pow, prime)) % prime;
        }
        shares.push_back({x, y});
    }
    return shares;
}

/// \brief Reconstruct the secret \f$P(0)\f$ from any \f$k\f$ shares by Lagrange interpolation.
/// \param prime the same prime field used to split.
inline std::uint64_t shamir_combine(const std::vector<SecretShare>& shares,
                                    std::uint64_t prime) {
    std::uint64_t secret = 0;
    const std::size_t k = shares.size();
    for (std::size_t i = 0; i < k; ++i) {
        // Lagrange basis L_i(0) = prod_{j!=i} x_j / (x_j - x_i).
        std::uint64_t num = 1, den = 1;
        for (std::size_t j = 0; j < k; ++j) {
            if (j == i) continue;
            num = mul_mod(num, shares[j].x % prime, prime);  // product of x_j
            // (x_j - x_i) mod prime, kept nonnegative
            std::uint64_t diff = (shares[j].x % prime + prime - shares[i].x % prime) % prime;
            den = mul_mod(den, diff, prime);
        }
        // Modular inverse of den via Fermat: den^(prime-2) mod prime.
        std::uint64_t inv_den = mod_pow(den, prime - 2, prime);
        std::uint64_t term = mul_mod(shares[i].y % prime, mul_mod(num, inv_den, prime), prime);
        secret = (secret + term) % prime;
    }
    return secret;
}

}  // namespace datamunge::algorithms
