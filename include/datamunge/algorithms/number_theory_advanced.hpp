#pragma once

/// @file number_theory_advanced.hpp
/// @brief Chakravala's method for Pell's equation, Lenstra's elliptic-curve factorisation, and
///        the AKS primality test (traced step by step in aks.hpp).

#include <cstdint>

namespace datamunge::algorithms {

/// @brief A solution (x, y) of Pell's equation \f$x^2 - N y^2 = 1\f$.
struct PellSolution {
    __int128 x{0};      ///< x of the fundamental solution.
    __int128 y{0};      ///< y of the fundamental solution.
    bool     found{false}; ///< false only when N is a perfect square.
};

/// @brief Chakravala's method: the fundamental solution of Pell's equation \f$x^2 - N y^2 = 1\f$
///        for a non-square \f$N > 1\f$.
/// @param N The non-square coefficient.
/// @return The smallest positive solution; `found` is false if @p N is a perfect square.
PellSolution chakravala(std::uint64_t N);

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
