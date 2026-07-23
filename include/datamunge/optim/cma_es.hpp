#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct CMAESOptions {
    /// @brief Offspring sampled per generation; zero selects the standard 4 + floor(3 log(n)).
    std::size_t population_size{0};
    /// @brief Initial global search standard deviation.
    double initial_step_size{0.5};
    std::size_t max_generations{500};
    /// @brief Stops when the global step size times the largest covariance axis is below this.
    double tolerance{1e-8};
    std::uint64_t seed{42};
};

/// @brief Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for unconstrained continuous
///        derivative-free optimization.  CMA-ES adapts a full Gaussian covariance matrix from
///        the best sampled candidates each generation, making it effective on rotated and
///        ill-conditioned objectives.
class CMAES {
public:
    explicit CMAES(CMAESOptions options = {});

    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates) const;

private:
    CMAESOptions options_;
};

} // namespace datamunge::optim
