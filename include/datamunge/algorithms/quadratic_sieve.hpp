#pragma once

// The quadratic sieve: factor a composite odd integer by finding a congruence of
// squares X^2 == Y^2 (mod n) with X != +-Y, so gcd(X - Y, n) is a nontrivial
// factor. Smooth values of Q(x) = x^2 - n are located with a factor base of
// primes modulo which n is a quadratic residue, and a linear dependency among
// their exponent vectors mod 2 assembles the square.

#include <cstdint>

namespace datamunge::algorithms {

// Return a nontrivial factor of composite n, or n itself if none is found
// (e.g. n prime). n should be odd and not a prime power for the sieve proper;
// small factors and perfect squares are handled directly.
std::uint64_t quadratic_sieve(std::uint64_t n);

} // namespace datamunge::algorithms
