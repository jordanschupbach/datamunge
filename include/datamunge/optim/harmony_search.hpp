#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct HarmonySearchOptions {
    /// @brief HMS: harmony memory size.
    std::size_t population_size{30};
    /// @brief HMCR: probability a dimension is drawn from the harmony memory rather than
    ///        uniformly at random, in (0, 1).
    double memory_consideration_rate{0.9};
    /// @brief PAR: probability a memory-considered dimension is further pitch-adjusted, in [0, 1].
    double pitch_adjustment_rate{0.3};
    /// @brief Pitch adjustment bandwidth, as a fraction of each dimension's [lower, upper] range.
    double bandwidth_fraction{0.05};
    /// @brief HS improvises exactly one new candidate per iteration (unlike generational methods
    ///        that evaluate population_size candidates per generation), so it typically needs a
    ///        much larger iteration budget for a comparable total evaluation count.
    std::size_t max_iterations{5000};
    /// @brief Stops after the best-ever value has improved by less than this for
    ///        20 * population_size consecutive iterations.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief Harmony search on an ArbitraryFunction within a box constraint: a fixed-size harmony
///        memory of candidate solutions is improvised one new vector at a time, mixing memory
///        consideration, pitch adjustment, and random selection per dimension, with greedy
///        replacement of the worst memory member. Derivative-free.
class HarmonySearch {
public:
    explicit HarmonySearch(HarmonySearchOptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed one member of the harmony memory.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    HarmonySearchOptions options_;
};

} // namespace datamunge::optim
