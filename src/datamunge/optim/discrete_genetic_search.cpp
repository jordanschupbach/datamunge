/// @file
/// @brief Budgeted binary, permutation and island genetic search.

#include <datamunge/optim/discrete_genetic_search.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::optim {
/// @brief Typed genetic operations and evaluated populations.
namespace {

/// @brief Evaluated chromosome.
struct Genotype {
  CategoricalPoint genes; ///< Feasible bit vector or permutation.
  double value; ///< Observed loss for these genes.
};

/// @brief Compare losses for minimization.
/// @param a First chromosome.
/// @param b Second chromosome.
/// @return Whether a strictly improves on b.
bool better(const Genotype& a, const Genotype& b) {
  return a.value < b.value;
}

/// @brief Draw the better of independent tournament candidates.
/// @param population Initialized island.
/// @param size Positive tournament size.
/// @param rng Borrowed local engine.
/// @return Reference to the lowest-loss sampled parent.
const Genotype& parent(const std::vector<Genotype>& population, std::size_t size, std::mt19937_64& rng) {
  std::uniform_int_distribution<std::size_t> draw(0, population.size() - 1);
  auto winner = draw(rng);
  for (std::size_t i = 1; i < size; ++i) {
    const auto candidate = draw(rng);
    if (better(population[candidate], population[winner])) winner = candidate;
  }
  return population[winner];
}

/// @brief Select two ordered distinct loci.
/// @param n Chromosome length, at least two.
/// @param rng Borrowed local engine.
/// @return Lower and higher selected indices.
std::pair<std::size_t, std::size_t> loci(std::size_t n, std::mt19937_64& rng) {
  const auto a = std::uniform_int_distribution<std::size_t>(0, n - 1)(rng);
  auto b = std::uniform_int_distribution<std::size_t>(0, n - 2)(rng);
  if (b >= a) ++b;
  return std::minmax(a, b);
}

/// @brief Retain a segment and fill remaining loci cyclically in the other parent's order.
/// @param genes First parent copied for filling, with a retained segment.
/// @param other Second parent with the same set of genes.
/// @param rng Borrowed local engine for cut positions.
void order_crossover(CategoricalPoint& genes, const CategoricalPoint& other, std::mt19937_64& rng) {
  const auto n = genes.size();
  const auto [lo, hi] = loci(n, rng);
  std::vector<bool> present(n);
  for (std::size_t i = lo; i <= hi; ++i) present[genes[i]] = true;
  auto insert = (hi + 1) % n;
  auto read = insert;
  for (std::size_t i = 0; i < n; ++i) {
    const auto value = other[read];
    if (!present[value]) {
      genes[insert] = value;
      insert = (insert + 1) % n;
    }
    read = (read + 1) % n;
  }
}

/// @brief Choose parents and produce a feasible child.
/// @param island Initialized active population.
/// @param options Representation, tournament and variation controls.
/// @param rng Borrowed local engine.
/// @return Valid chromosome before evaluation.
CategoricalPoint offspring(const std::vector<Genotype>& island, const DiscreteGeneticOptions& options,
                            std::mt19937_64& rng) {
  auto genes = parent(island, options.tournament, rng).genes;
  const auto& other = parent(island, options.tournament, rng).genes;
  std::uniform_real_distribution<double> uniform;
  if (options.encoding == GeneticEncoding::binary) {
    const bool cross = uniform(rng) < options.crossover;
    for (std::size_t j = 0; j < genes.size(); ++j) {
      if (cross && uniform(rng) < 0.5) genes[j] = other[j];
      if (uniform(rng) < options.mutation) genes[j] = 1 - genes[j];
    }
  } else if (genes.size() >= 2) {
    if (uniform(rng) < options.crossover) order_crossover(genes, other, rng);
    if (uniform(rng) < options.mutation) {
      const auto [lo, hi] = loci(genes.size(), rng);
      if (uniform(rng) < options.inversion) {
        std::reverse(genes.begin() + lo, genes.begin() + hi + 1);
      } else {
        std::swap(genes[lo], genes[hi]);
      }
    }
  }
  return genes;
}

/// @brief Copy evaluated island champions around a ring without extra work.
/// @param islands Populations of evaluated chromosomes, updated synchronously.
void migrate(std::vector<std::vector<Genotype>>& islands) {
  std::vector<Genotype> elites;
  elites.reserve(islands.size());
  for (const auto& island : islands) elites.push_back(*std::ranges::min_element(island, better));
  for (std::size_t i = 0; i < islands.size(); ++i) {
    auto& destination = islands[(i + 1) % islands.size()];
    const auto worst = std::ranges::max_element(destination, better);
    if (better(elites[i], *worst)) *worst = std::move(elites[i]);
  }
}

} // namespace

DiscreteDistributionResult discrete_genetic_search(const CategoricalObjective& objective,
                                                    std::size_t dimensions,
                                                    const DiscreteGeneticOptions& options) {
  if (!objective || dimensions == 0 || options.evaluations == 0 || options.population < 2 ||
      options.tournament == 0 || options.islands == 0 ||
      (options.encoding != GeneticEncoding::binary && options.encoding != GeneticEncoding::permutation) ||
      !(options.crossover >= 0 && options.crossover <= 1 && options.mutation >= 0 &&
        options.mutation <= 1 && options.inversion >= 0 && options.inversion <= 1)) {
    throw std::invalid_argument("Invalid discrete genetic search domain or controls");
  }
  DiscreteDistributionResult result{{}, std::numeric_limits<double>::infinity(), 0};
  std::mt19937_64 rng(options.seed);
  std::bernoulli_distribution bit;
  /// Score a feasible chromosome, advance the work count and retain the incumbent.
  auto evaluate = [&](CategoricalPoint genes) {
    const auto value = objective(genes);
    if (!std::isfinite(value)) throw std::domain_error("Discrete genetic loss must be finite");
    ++result.evaluations;
    if (value < result.value) {
      result.value = value;
      result.point = genes;
    }
    return Genotype{std::move(genes), value};
  };

  std::vector<std::vector<Genotype>> islands(std::min(options.islands, options.evaluations));
  for (std::size_t individual = 0;
       individual < options.population && result.evaluations < options.evaluations; ++individual) {
    for (auto& island : islands) {
      if (result.evaluations == options.evaluations) break;
      CategoricalPoint genes(dimensions);
      if (options.encoding == GeneticEncoding::binary) {
        for (auto& value : genes) value = bit(rng);
      } else {
        std::iota(genes.begin(), genes.end(), std::size_t{0});
        std::shuffle(genes.begin(), genes.end(), rng);
      }
      island.push_back(evaluate(std::move(genes)));
    }
  }

  std::size_t sweep = 0;
  while (result.evaluations < options.evaluations) {
    for (auto& island : islands) {
      if (result.evaluations == options.evaluations) break;
      if (options.steady_state) {
        auto child = evaluate(offspring(island, options, rng));
        const auto worst = std::ranges::max_element(island, better);
        if (better(child, *worst)) *worst = std::move(child);
      } else {
        std::vector<Genotype> next;
        next.reserve(island.size());
        next.push_back(*std::ranges::min_element(island, better));
        while (next.size() < island.size() && result.evaluations < options.evaluations) {
          next.push_back(evaluate(offspring(island, options, rng)));
        }
        island = std::move(next);
      }
    }
    ++sweep;
    if (result.evaluations < options.evaluations && islands.size() > 1 &&
        options.migration_interval != 0 && sweep % options.migration_interval == 0) {
      migrate(islands);
    }
  }
  return result;
}

} // namespace datamunge::optim
