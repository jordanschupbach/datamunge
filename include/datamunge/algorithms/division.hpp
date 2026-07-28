#pragma once

// Sequential integer division algorithms (unsigned, 64-bit).
//
// Each computes quotient = dividend / divisor and remainder = dividend % divisor
// by a different classic mechanism, and reports how much work it took.

#include <cstdint>

namespace datamunge::algorithms {

struct DivResultEx {
    std::uint64_t quotient{0};
    std::uint64_t remainder{0};
    int           iterations{0}; // bit steps or Newton iterations, per algorithm
};

// Restoring division: at each bit, trial-subtract the divisor; if the partial
// remainder goes negative, add it back (restore) and emit a 0 quotient bit.
DivResultEx restoring_divide(std::uint64_t dividend, std::uint64_t divisor);

// Non-restoring division: never "puts back" an over-subtraction. Each step
// adds or subtracts the divisor depending on the sign of the running partial
// remainder, with a single correction at the end -- one add/subtract per bit
// instead of restoring's occasional second add.
DivResultEx non_restoring_divide(std::uint64_t dividend, std::uint64_t divisor);

// Newton-Raphson division: computes the reciprocal 1/divisor by the quadratically
// convergent iteration x <- x(2 - divisor*x) in fixed point, then multiplies by
// the dividend. Multiplication-based, so it converges in O(log n) steps.
DivResultEx newton_raphson_divide(std::uint64_t dividend, std::uint64_t divisor);

} // namespace datamunge::algorithms
