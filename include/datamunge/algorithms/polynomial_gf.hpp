#pragma once

// Polynomials over a prime field GF(p).
//
// A polynomial is a coefficient vector in little-endian order: index i holds the
// coefficient of x^i, each in [0, p). Trailing zeros are trimmed, so the zero
// polynomial is the empty vector and deg(f) = size - 1.

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

using GFPoly = std::vector<std::int64_t>;

struct PolyDivMod {
    GFPoly quotient;
    GFPoly remainder;
};

// Long division of a by b over GF(p) (p prime, b != 0): returns quotient and
// remainder with deg(remainder) < deg(b) and a == quotient*b + remainder (mod p).
PolyDivMod poly_divmod_gf(const GFPoly& a, const GFPoly& b, std::int64_t p);

// Chien search: all roots r in GF(p) with f(r) == 0, found by walking the field
// elements (0 and the successive powers of a primitive root) and updating each
// term incrementally -- the recursive evaluation that names the method.
std::vector<std::int64_t> chien_search(const GFPoly& f, std::int64_t p);

// Cantor-Zassenhaus: factor a monic squarefree polynomial over GF(p) (odd p)
// into its irreducible factors. The product of the returned factors equals the
// input (mod p). `seed` makes the randomised equal-degree split deterministic.
std::vector<GFPoly> cantor_zassenhaus(const GFPoly& f, std::int64_t p,
                                      std::uint64_t seed = 0x9E3779B97F4A7C15ULL);

} // namespace datamunge::algorithms
