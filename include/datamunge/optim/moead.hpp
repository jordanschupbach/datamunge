#pragma once

#include <datamunge/optim/multi_objective.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::optim {

struct MOEADOptions {
    /// @brief Requested number of scalar subproblems / weight vectors. The actual count is the
    ///        largest Das-Dennis simplex-lattice size not exceeding this (exactly this for two
    ///        objectives).
    std::size_t population_size{100};
    std::size_t max_generations{250};
    /// @brief Neighborhood size T: each subproblem mates with, and updates, its T closest
    ///        weight-vector neighbors.
    std::size_t neighborhood_size{20};
    /// @brief Scalarization: "tchebycheff" (default) or "weighted_sum".
    std::string decomposition{"tchebycheff"};
    double crossover_rate{0.9};
    double eta_crossover{20.0};
    /// @brief Per-variable polynomial-mutation probability; if < 0, defaults to 1 / n_variables.
    double mutation_rate{-1.0};
    double eta_mutation{20.0};
    std::uint64_t seed{42};
};

/// @brief MOEA/D -- Multi-Objective Evolutionary Algorithm based on Decomposition (Zhang & Li,
///        2007). Rather than sorting by Pareto dominance, MOEA/D decomposes the problem into a
///        set of single-objective scalar subproblems -- one per uniformly spread weight vector
///        -- and solves them simultaneously. Each subproblem is optimized using information
///        only from its neighbors (the subproblems with nearby weight vectors), so a good
///        solution to one subproblem helps its neighbors. An ideal-point-anchored Tchebycheff
///        (or weighted-sum) scalarization drives each subproblem toward a different part of the
///        Pareto front. Derivative-free.
class MOEAD {
public:
    explicit MOEAD(MOEADOptions options = {});

    /// @brief Approximates the Pareto front of @p function within [@p lower_bound,
    ///        @p upper_bound], returning the non-dominated set of the final population.
    [[nodiscard]] std::vector<ParetoPoint> optimize(MultiObjectiveFunction& function,
                                                     const std::vector<double>& lower_bound,
                                                     const std::vector<double>& upper_bound) const;

private:
    MOEADOptions options_;
};

} // namespace datamunge::optim
