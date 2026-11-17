/// @file multiplication.hpp
/// @brief Compatibility aggregation of focused checked native producers.
#pragma once
#include <datamunge/algorithms/toom3.hpp>

// Fast integer multiplication algorithms.
//
// (Karatsuba lives in number_theory.hpp as karatsuba_multiply; this header adds
// Toom-Cook and checked fixed-prime NTT decimal multipliers plus
// Booth's signed hardware multiplication.)
//
// Big non-negative integers are passed and returned as ordinary base-10 decimal
// strings (no sign, no leading zeros except the single "0"). Booth's algorithm
// operates on machine words and multiplies signed values.

#include <cstdint>
#include <datamunge/algorithms/booth_multiplication.hpp>
#include <datamunge/algorithms/ntt_multiplication.hpp>
#include <string>

namespace datamunge::algorithms {} // namespace datamunge::algorithms
