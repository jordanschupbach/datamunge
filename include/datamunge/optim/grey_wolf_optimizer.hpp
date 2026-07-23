#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct GreyWolfOptimizerOptions {
    std::size_t population_size{30};
    std::size_t max_iterations{500};
    /// @brief Stops after the best-ever value has improved by less than this for 20 consecutive iterations.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief Grey Wolf Optimizer (GWO) on an ArbitraryFunction within a box constraint: a
///        population of wolves is guided each iteration by its three best members (alpha,
///        beta, delta), moving every wolf to the elementwise average of three independently
///        randomized pulls toward those leaders. Derivative-free. Mirjalili, Mirjalili &
///        Lewis (2014), "Grey wolf optimizer", Advances in Engineering Software, 69, 46-61.
class GreyWolfOptimizer {
public:
    explicit GreyWolfOptimizer(GreyWolfOptimizerOptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed one wolf of the population.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    GreyWolfOptimizerOptions options_;
};

} // namespace datamunge::optim
