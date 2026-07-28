#pragma once

// Pollard's kangaroo (lambda) algorithm for the discrete logarithm in a known
// interval. Given a generator g, a target h = g^x (mod P), and bounds a <= x <= b,
// it recovers x in O(sqrt(b - a)) group operations using a "tame" and a "wild"
// kangaroo whose deterministic jumps eventually collide.

#include <cstdint>

namespace datamunge::algorithms {

struct KangarooResult {
    std::uint64_t x{0};
    bool          found{false};
    std::uint64_t jumps{0}; // total kangaroo jumps taken
};

// Solve g^x == h (mod P) for x in [a, b]. `seed` makes the jump set deterministic.
KangarooResult pollard_kangaroo(std::uint64_t g, std::uint64_t h, std::uint64_t P,
                                std::uint64_t a, std::uint64_t b,
                                std::uint64_t seed = 0x2545F4914F6CDD1DULL);

} // namespace datamunge::algorithms
