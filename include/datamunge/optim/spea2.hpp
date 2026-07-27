#pragma once

#include <datamunge/optim/multi_objective.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::optim {

struct SPEA2Options {
    std::size_t population_size{100};
    /// @brief Size of the external archive of best solutions; if 0, defaults to population_size.
    std::size_t archive_size{0};
    std::size_t max_generations{250};
    double crossover_rate{0.9};
    double eta_crossover{15.0};
    /// @brief Per-variable polynomial-mutation probability; if < 0, defaults to 1 / n_variables.
    double mutation_rate{-1.0};
    double eta_mutation{20.0};
    std::uint64_t seed{42};
};

/// @brief SPEA2 -- the Strength Pareto Evolutionary Algorithm 2 (Zitzler, Laumanns & Thiele,
///        2001). A multi-objective EA maintaining an external archive of the best solutions
///        found. Each solution's fitness combines a *raw* term -- the summed strengths of the
///        solutions that dominate it (0 for a non-dominated solution) -- with a *density* term
///        based on the distance to its k-th nearest neighbor in objective space, which spreads
///        the archive evenly. Environmental selection copies the non-dominated solutions into
///        the next archive, then either fills it with the best dominated ones or truncates it
///        by repeatedly removing the most crowded solution. Derivative-free.
class SPEA2 {
public:
    explicit SPEA2(SPEA2Options options = {});

    /// @brief Approximates the Pareto front of @p function within [@p lower_bound,
    ///        @p upper_bound], returning the non-dominated set of the final archive.
    [[nodiscard]] std::vector<ParetoPoint> optimize(MultiObjectiveFunction& function,
                                                     const std::vector<double>& lower_bound,
                                                     const std::vector<double>& upper_bound) const;

private:
    SPEA2Options options_;
};

} // namespace datamunge::optim
