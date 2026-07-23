#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct CrossEntropyMethodOptions {
    /// @brief Candidates sampled from the search distribution each iteration.
    std::size_t population_size{60};
    /// @brief Fraction of population_size kept as the elite set;
    ///        elite_count = max(2, ceil(population_size * elite_ratio)).
    double elite_ratio{0.2};
    /// @brief Per-dimension standard deviation of the initial sampling distribution.
    double initial_std_dev{1.0};
    /// @brief Exponential smoothing factor in (0, 1] blending the new elite statistics with the
    ///        previous generation's: new = smoothing * elite_stat + (1 - smoothing) * old.
    ///        smoothing == 1 means full replacement each iteration (no smoothing at all).
    double smoothing{0.7};
    std::size_t max_iterations{300};
    /// @brief Stops after the best-ever value has improved by less than this for 20 consecutive
    ///        iterations, or once every per-dimension standard deviation has collapsed below it.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief The Cross-Entropy Method (CEM) for derivative-free continuous optimization on an
///        ArbitraryFunction within a box constraint. A diagonal-covariance Gaussian search
///        distribution is repeatedly refit from the best ("elite") fraction of each generation's
///        samples, with the new distribution parameters exponentially smoothed against the
///        previous generation's for stability (Rubinstein 1997; Rubinstein & Kroese 2004).
class CrossEntropyMethod {
public:
    explicit CrossEntropyMethod(CrossEntropyMethodOptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed the initial distribution mean.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    CrossEntropyMethodOptions options_;
};

} // namespace datamunge::optim
