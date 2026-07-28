#pragma once

// Berlekamp's root-finding algorithm (Berlekamp-Rabin) over GF(p).
//
// Finds all roots in GF(p) of a polynomial by the algebraic split
// gcd(f(x), (x + d)^((p-1)/2) - 1), which separates the roots r for which
// r + d is a quadratic residue from the rest; recursing with fresh shifts d
// isolates every root. As a special case it computes modular square roots
// (the roots of x^2 - a).

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

// Polynomial over GF(p): coefficient i is the coefficient of x^i (little-endian).
using BPoly = std::vector<std::int64_t>;

// All roots in GF(p) (odd prime) of f, ascending. `seed` makes the randomised
// shifts deterministic.
std::vector<std::int64_t> berlekamp_roots(const BPoly& f, std::int64_t p,
                                          std::uint64_t seed = 0xC2B2AE3D27D4EB4FULL);

} // namespace datamunge::algorithms
