#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct CuckooSearchOptions {
    /// @brief Number of nests (candidate solutions) in the population.
    std::size_t population_size{25};
    /// @brief pa: fraction of the worst nests discovered/abandoned each iteration. Yang & Deb's
    ///        original paper headlines 0.25, but this implementation's "replace a randomly
    ///        chosen comparison nest" Levy-flight rule (faithful to the paper) homogenizes the
    ///        population fast enough at pa=0.25 that convergence quality becomes sharply
    ///        seed-dependent on smooth objectives (verified: only 6/30 seeds converged well at
    ///        pa=0.25, population_size=25 on a simple 3D sphere). 0.4 -- still within the
    ///        commonly-used literature range -- injects enough fresh diversity each iteration to
    ///        fix this at the default population size (verified: 30/30 seeds).
    double discovery_rate{0.4};
    /// @brief Levy flight stability index, in (0, 2]; 1.5 is the standard choice in the
    ///        original paper.
    double levy_beta{1.5};
    /// @brief alpha: overall Levy step-size multiplier.
    double step_scale{0.01};
    std::size_t max_iterations{500};
    /// @brief Stops after the best-ever value has improved by less than this for 20
    ///        consecutive iterations.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief Cuckoo Search (Yang & Deb, 2009) on an ArbitraryFunction within a box constraint: a
///        population of nests is perturbed by heavy-tailed Levy flights (via Mantegna's
///        algorithm) and a fraction of the worst nests is abandoned and rebuilt each
///        iteration, mimicking brood parasitism. Derivative-free.
class CuckooSearch {
public:
    explicit CuckooSearch(CuckooSearchOptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed one nest of the population.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    CuckooSearchOptions options_;
};

} // namespace datamunge::optim
