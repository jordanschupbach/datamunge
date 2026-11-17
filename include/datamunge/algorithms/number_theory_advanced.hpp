#pragma once
#include <datamunge/algorithms/lenstra_ecm.hpp>

/// @file number_theory_advanced.hpp
/// @brief Chakravala's method for Pell's equation, Lenstra's elliptic-curve factorisation, and
///        the AKS primality test (traced step by step in aks.hpp).

#include <cstdint>
#include <datamunge/algorithms/chakravala.hpp>

namespace datamunge::algorithms {

/// @brief The AKS primality test: deterministic and polynomial in \f$\log n\f$.
/// @return true if and only if @p n is prime; aks_trace() shows which step decided.
bool aks_is_prime(std::uint64_t n);

} // namespace datamunge::algorithms
