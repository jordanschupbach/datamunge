#pragma once

// The index-calculus algorithm for discrete logarithms in the multiplicative
// group modulo a prime p. It collects "relations" -- random powers g^k that are
// smooth over a factor base of small primes -- solves a linear system for the
// logarithms of the factor-base primes, then expresses the target's logarithm
// through one more smooth relation. Subexponential in the size of p.

#include <cstdint>

namespace datamunge::algorithms {

struct DiscreteLogResult {
    std::uint64_t x{0};
    bool          found{false};
};

// Solve g^x == h (mod p) for x modulo the order of g (p prime, g a generator).
// `seed` makes the randomised relation search deterministic.
DiscreteLogResult index_calculus(std::uint64_t g, std::uint64_t h, std::uint64_t p,
                                 std::uint64_t seed = 0x1D2C3B4A59687766ULL);

} // namespace datamunge::algorithms
