#pragma once

/// @file
/// @brief Environmental selection for reference-direction and indicator Pareto optimizers.

#include <datamunge/optim/pareto_search.hpp>

namespace datamunge::optim::detail {

/// @brief Return the validated unit directions for NSGA-III niching.
/// @param dimensions Fixed objective dimension.
/// @param options Explicit reference directions or default lattice controls.
/// @return Direction vectors with Euclidean norm one.
/// @throws std::invalid_argument Invalid explicit directions or lattice size.
/// @throws std::length_error More than 4096 default lattice directions.
std::vector<std::vector<double>> reference_directions(std::size_t dimensions,
                                                      const ParetoSearchOptions& options);

/// @brief Fitness for IBEA or HypE mating selection.
/// @param points Scored population, with objectives scaled into the unit box.
/// @param options Algorithm choice, indicator scale and sample count.
/// @param rng Borrowed local engine for HypE samples.
/// @return Fitness ordered so larger is better, one value per population member.
std::vector<double> pareto_mating_fitness(const std::vector<ParetoPoint>& points,
                                         const ParetoSearchOptions& options, std::mt19937_64& rng);

/// @brief Select survivor indices from a merged parent and offspring pool.
/// @param points Scored population, with objectives scaled into the unit box.
/// @param keep Positive number of survivors, less than the population size.
/// @param options Search method and selection-specific controls.
/// @param directions Valid unit directions for NSGA-III; unused for other methods.
/// @param rng Borrowed engine for selection ties and Monte Carlo draws.
/// @return Survivor indices in their next-population order.
std::vector<std::size_t> pareto_survivors(const std::vector<ParetoPoint>& points,
    std::size_t keep, const ParetoSearchOptions& options,
    const std::vector<std::vector<double>>& directions, std::mt19937_64& rng);

} // namespace datamunge::optim::detail
