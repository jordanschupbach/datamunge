#pragma once

/// @file number_theory_advanced.hpp
/// @brief Chakravala's method for Pell's equation, Lenstra's elliptic-curve factorisation, and
///        the AKS primality test (traced step by step in aks.hpp).

#include <cstdint>
#include <datamunge/algorithms/chakravala.hpp>

namespace datamunge::algorithms {

/// @brief Lenstra elliptic-curve factorisation.
/// @param n The number to factor.
/// @param seed Seed for the random curves, so results are reproducible.
/// @return A nontrivial factor of a composite @p n, or @p n itself if none is found (for
///         example when @p n is prime).
std::uint64_t lenstra_ecm(std::uint64_t n, std::uint64_t seed = 0x243F6A8885A308D3ULL);

/// @brief The AKS primality test: deterministic and polynomial in \f$\log n\f$.
/// @return true if and only if @p n is prime; aks_trace() shows which step decided.
bool aks_is_prime(std::uint64_t n);

} // namespace datamunge::algorithms
