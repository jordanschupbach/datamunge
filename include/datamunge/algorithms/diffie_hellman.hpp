#pragma once

/// \file diffie_hellman.hpp
/// \brief Diffie-Hellman key exchange over a prime field.
///
/// Diffie-Hellman (1976) lets two parties agree on a shared secret over a public channel
/// without ever transmitting it. Fix a large prime modulus \f$p\f$ and a generator \f$g\f$
/// of the multiplicative group \f$\mathbb{Z}_p^\ast\f$. Alice picks a secret \f$a\f$ and
/// publishes \f$A = g^a \bmod p\f$; Bob picks a secret \f$b\f$ and publishes
/// \f$B = g^b \bmod p\f$. Each raises the other's public value to their own secret:
/// \f[
///   B^a = (g^b)^a = g^{ab} = (g^a)^b = A^b \pmod p,
/// \f]
/// so both arrive at the same \f$s = g^{ab}\bmod p\f$. An eavesdropper sees \f$g,p,A,B\f$
/// but recovering \f$s\f$ requires solving the *discrete logarithm* problem, believed hard
/// for well-chosen \f$p\f$.
///
/// This implementation works with 64-bit values (using 128-bit intermediate products), so
/// it demonstrates the protocol on toy/moderate primes rather than the 2048-bit primes of
/// production use; the arithmetic is identical.

#include <cstdint>

#include <datamunge/algorithms/modular_exponentiation.hpp>  // mod_pow (square-and-multiply)

namespace datamunge::algorithms {

/// \brief Modular multiplication \f$(a\cdot b)\bmod m\f$ without overflow (128-bit intermediate).
inline std::uint64_t mul_mod(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
    return static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b)) % m);
}

/// \brief Compute the public value \f$g^{secret}\bmod p\f$ to publish to the peer.
inline std::uint64_t dh_public_key(std::uint64_t g, std::uint64_t secret, std::uint64_t p) {
    return mod_pow(g, secret, p);
}

/// \brief Derive the shared secret \f$(\text{peer\_public})^{secret}\bmod p\f$.
inline std::uint64_t dh_shared_secret(std::uint64_t peer_public, std::uint64_t secret,
                                      std::uint64_t p) {
    return mod_pow(peer_public, secret, p);
}

}  // namespace datamunge::algorithms
