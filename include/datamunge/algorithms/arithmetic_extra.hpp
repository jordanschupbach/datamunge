#pragma once

/// @file arithmetic_extra.hpp
/// @brief Arithmetic entry points, retaining the historical Goldschmidt include location.

#include <cstdint>
#include <datamunge/algorithms/cipolla.hpp>
#include <datamunge/algorithms/goldschmidt_division.hpp>
#include <datamunge/algorithms/montgomery.hpp>
#include <vector>

namespace datamunge::algorithms {

/// @brief *Addition-chain exponentiation*: computes @c x^e using a short *addition chain* -- a
///        sequence @c 1=c_0,c_1,\dots,c_L=e in which each term is a sum of two earlier terms, so each
///        corresponds to one multiplication. The *shortest* chain minimizes the multiplications needed
///        for @c x^e (fewer than the standard binary square-and-multiply for many exponents, e.g.
///        @c x^{15} in 5 rather than 6). This routine returns a chain: the binary-method chain for
///        general @c e, refined by a bounded search that finds a provably shortest chain for small
///        @c e.
///
/// @param e the exponent (@c e >= 1).
/// @return an addition chain from 1 to @p e (each element the sum of two earlier ones); its length
///         minus one is the multiplication count.
std::vector<std::uint64_t> addition_chain(std::uint64_t e);

} // namespace datamunge::algorithms
