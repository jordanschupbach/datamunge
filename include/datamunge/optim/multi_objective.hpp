#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace datamunge::optim {

/// @brief A vector-valued objective for multi-objective optimization: evaluate() returns one
///        value per objective, all of which are minimized simultaneously. Unlike the
///        single-objective ArbitraryFunction, there is generally no single best point -- the
///        solution is a *set* of mutually non-dominated trade-offs (the Pareto front).
///
///        This is a director-enabled extension point: subclass it directly (in C++, or, via
///        SWIG directors, in any supported language) to define a custom multi-objective
///        problem. Every objective is minimized; negate an objective to maximize it.
class MultiObjectiveFunction {
public:
    virtual ~MultiObjectiveFunction() = default;

    /// @brief Evaluates every objective at @p coordinates, returning a vector whose length is
    ///        the (fixed) number of objectives.
    virtual std::vector<double> evaluate(const std::vector<double>& coordinates) = 0;
};

/// @brief A decision vector together with its objective vector -- one member of an
///        approximated Pareto set (@c coordinates) / Pareto front (@c objectives).
struct ParetoPoint {
    std::vector<double> coordinates;
    std::vector<double> objectives;
};

/// @brief Pareto dominance for minimization: returns true iff @p a is no worse than @p b in
///        every objective and strictly better in at least one. A point is on the Pareto front
///        iff no other feasible point dominates it.
[[nodiscard]] bool dominates(const std::vector<double>& a, const std::vector<double>& b);

/// @brief The non-dominated subset of @p points, compared by their objective vectors. This is
///        the Pareto-optimal approximation extracted from a final population/archive.
[[nodiscard]] std::vector<ParetoPoint> non_dominated_front(const std::vector<ParetoPoint>& points);

namespace detail {

/// @brief Simulated Binary Crossover (SBX; Deb & Agrawal 1995). Recombines real-valued
///        parents @p p1, @p p2 into children @p c1, @p c2, clamped to [@p lower, @p upper].
///        Per-variable it mimics the spread of single-point binary crossover; @p eta_c is the
///        distribution index (larger = children nearer parents). Applied with prob.
///        @p crossover_rate, else children copy the parents.
void sbx_crossover(const std::vector<double>& p1, const std::vector<double>& p2,
                   std::vector<double>& c1, std::vector<double>& c2, const std::vector<double>& lower,
                   const std::vector<double>& upper, double crossover_rate, double eta_c, std::mt19937_64& rng);

/// @brief Polynomial mutation (Deb & Goyal 1996), in place, clamped to [@p lower, @p upper].
///        Each variable is perturbed with probability @p mutation_rate by a polynomial
///        distribution with index @p eta_m (larger = smaller perturbations).
void polynomial_mutation(std::vector<double>& x, const std::vector<double>& lower,
                         const std::vector<double>& upper, double mutation_rate, double eta_m,
                         std::mt19937_64& rng);

} // namespace detail

} // namespace datamunge::optim
