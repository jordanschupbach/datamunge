#pragma once

#include <datamunge/optim/multi_objective.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::optim {

struct NSGA2Options {
    /// @brief Number of solutions carried each generation (rounded up to an even number so the
    ///        crossover pairing is exact).
    std::size_t population_size{100};
    std::size_t max_generations{250};
    /// @brief Probability of applying SBX to a parent pair (else the parents pass through).
    double crossover_rate{0.9};
    /// @brief SBX distribution index (larger => children nearer parents).
    double eta_crossover{15.0};
    /// @brief Per-variable polynomial-mutation probability; if < 0, defaults to 1 / n_variables.
    double mutation_rate{-1.0};
    /// @brief Polynomial-mutation distribution index (larger => smaller perturbations).
    double eta_mutation{20.0};
    std::uint64_t seed{42};
};

/// @brief NSGA-II -- the Non-dominated Sorting Genetic Algorithm II (Deb, Pratap, Agarwal &
///        Meyarivan, 2002). An elitist multi-objective evolutionary algorithm: each generation
///        it merges parents and offspring, ranks them into Pareto fronts by *fast
///        non-dominated sorting*, and fills the next generation front by front, breaking the
///        final over-full front by *crowding distance* to preserve diversity along the front.
///        Selection, crossover (SBX) and mutation (polynomial) act on real-valued decision
///        vectors within a box constraint. Derivative-free.
class NSGA2 {
public:
    explicit NSGA2(NSGA2Options options = {});

    /// @brief Approximates the Pareto front of @p function within [@p lower_bound,
    ///        @p upper_bound], returning the non-dominated set of the final population.
    [[nodiscard]] std::vector<ParetoPoint> optimize(MultiObjectiveFunction& function,
                                                     const std::vector<double>& lower_bound,
                                                     const std::vector<double>& upper_bound) const;

private:
    NSGA2Options options_;
};

} // namespace datamunge::optim
