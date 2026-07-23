#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct EstimationOfDistributionOptions {
    /// @brief Lambda: candidates sampled from the Gaussian model each generation.
    std::size_t population_size{60};
    /// @brief Fraction of population_size kept as "elite" each generation to re-estimate the
    ///        Gaussian from; mu = max(2, ceil(population_size * selection_ratio)).
    double selection_ratio{0.5};
    /// @brief Per-dimension standard deviation of the initial diagonal covariance.
    double initial_std_dev{1.0};
    /// @brief Diagonal regularization added to the re-estimated covariance every generation, to
    ///        keep it numerically well-conditioned (needed whenever mu <= dimension, or whenever
    ///        the elite set happens to be nearly coplanar).
    double covariance_regularization{1e-9};
    std::size_t max_generations{300};
    /// @brief Stops after the best-ever value has improved by less than this for 20 consecutive
    ///        generations, or once the covariance has collapsed to near-zero.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief Estimation of Distribution Algorithm (EMNA-style, full-covariance Gaussian) on an
///        ArbitraryFunction within a box constraint. Unlike explicit-operator methods (GA, DE)
///        or covariance-adapting methods (CMA-ES), EMNA builds an explicit multivariate Gaussian
///        model of where the good solutions are each generation -- re-estimating its mean and
///        covariance from scratch as the empirical mean/covariance of the current generation's
///        elite, with no smoothing or evolution paths -- and samples the next generation directly
///        from that model. Derivative-free.
class EstimationOfDistribution {
public:
    explicit EstimationOfDistribution(EstimationOfDistributionOptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed the initial Gaussian mean.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    EstimationOfDistributionOptions options_;
};

} // namespace datamunge::optim
