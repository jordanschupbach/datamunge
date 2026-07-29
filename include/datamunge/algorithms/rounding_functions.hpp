#pragma once

// The classic rounding functions -- the ways to map a real number to an integer,
// differing only in how they break *ties* at the exact half. The choice matters:
// rounding halves always up biases sums upward, while round-half-to-even
// (banker's rounding, the IEEE-754 default) cancels that bias by sending ties to
// the nearest even integer. This header collects the standard modes and a helper
// to measure their bias.

#include <cmath>
#include <vector>

namespace datamunge::algorithms {

// Round half to even (banker's rounding): ties go to the nearest even integer.
inline double round_half_to_even(double x) {
    const double f = std::floor(x);
    const double diff = x - f;
    if (diff < 0.5) return f;
    if (diff > 0.5) return f + 1.0;
    // exactly halfway: pick the even neighbor
    return (std::fmod(f, 2.0) == 0.0) ? f : f + 1.0;
}

// Round half away from zero: 0.5 -> 1, -0.5 -> -1, 2.5 -> 3.
inline double round_half_away_from_zero(double x) {
    return (x >= 0.0) ? std::floor(x + 0.5) : std::ceil(x - 0.5);
}

// Round half up (toward +infinity on ties): 0.5 -> 1, -0.5 -> 0, 2.5 -> 3.
inline double round_half_up(double x) { return std::floor(x + 0.5); }

// Round half down (toward -infinity on ties): 0.5 -> 0, -0.5 -> -1, 2.5 -> 2.
inline double round_half_down(double x) { return std::ceil(x - 0.5); }

// Round toward zero (truncate): 0.9 -> 0, -0.9 -> 0, 2.5 -> 2.
inline double round_toward_zero(double x) { return std::trunc(x); }

enum class RoundMode { HalfToEven, HalfAwayFromZero, HalfUp, HalfDown, TowardZero, Ceil, Floor };

inline double round_with(double x, RoundMode mode) {
    switch (mode) {
        case RoundMode::HalfToEven:        return round_half_to_even(x);
        case RoundMode::HalfAwayFromZero:  return round_half_away_from_zero(x);
        case RoundMode::HalfUp:            return round_half_up(x);
        case RoundMode::HalfDown:          return round_half_down(x);
        case RoundMode::TowardZero:        return round_toward_zero(x);
        case RoundMode::Ceil:              return std::ceil(x);
        case RoundMode::Floor:             return std::floor(x);
    }
    return round_half_to_even(x);
}

// Mean signed rounding error (rounded - exact) over `values` under a mode -- a
// measure of a rounding rule's bias. Round-half-to-even keeps this near zero on
// symmetric tie-heavy data; round-half-up drifts positive.
inline double rounding_bias(const std::vector<double>& values, RoundMode mode) {
    if (values.empty()) return 0.0;
    double acc = 0.0;
    for (double x : values) acc += round_with(x, mode) - x;
    return acc / static_cast<double>(values.size());
}

} // namespace datamunge::algorithms
