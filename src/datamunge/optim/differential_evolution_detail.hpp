#pragma once

/// @file
/// @brief Shared sampling primitives for differential evolution.

#include <algorithm>
#include <array>
#include <random>

namespace datamunge::optim::detail {

/// @brief Rejection-sample distinct population indices, excluding the target.
/// @param rng Caller-owned local engine; advanced in place.
/// @param population_size Population size, validated to allow count distinct candidates.
/// @param exclude Target index that must not be selected.
/// @param count Number of leading entries to fill, at most five.
/// @return Fixed-size array with count distinct valid indices; unused entries remain zero.
inline std::array<std::size_t, 5> random_distinct_indices(std::mt19937_64& rng, const std::size_t population_size,
                                                   const std::size_t exclude, const std::size_t count) {
    std::uniform_int_distribution<std::size_t> dist(0, population_size - 1);
    std::array<std::size_t, 5> result{};
    std::size_t used = 0;
    while (used < count) {
        const std::size_t r = dist(rng);
        if (r == exclude) continue;
        if (std::find(result.begin(), result.begin() + used, r) != result.begin() + used) continue;
        result[used++] = r;
    }
    return result;
}

} // namespace datamunge::optim::detail
