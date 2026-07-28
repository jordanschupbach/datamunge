#pragma once

// Advanced number-theory algorithms: a Diophantine solver, an elliptic-curve
// factoriser, and a deterministic primality test.

#include <cstdint>

namespace datamunge::algorithms {

// Chakravala's method: fundamental solution of Pell's equation x^2 - N y^2 = 1
// for a non-square N > 1. `found` is false only if N is a perfect square.
struct PellSolution {
    __int128 x{0};
    __int128 y{0};
    bool     found{false};
};
PellSolution chakravala(std::uint64_t N);

// Lenstra elliptic-curve factorisation: return a nontrivial factor of a
// composite n, or n itself if none is found (e.g. n prime). `seed` makes the
// randomised curve choices deterministic.
std::uint64_t lenstra_ecm(std::uint64_t n, std::uint64_t seed = 0x243F6A8885A308D3ULL);

// AKS primality test: deterministic, polynomial-time. Returns true iff n is prime.
bool aks_is_prime(std::uint64_t n);

} // namespace datamunge::algorithms
