/// @file
/// @brief Categorical probability updates and discrete dependency-chain fitting.

#include <datamunge/optim/discrete_distribution_search.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace datamunge::optim {
/// @brief Internal sample, model and update logic.
namespace {

/// @brief Association between a sampled assignment and its evaluated loss.
struct Sample {
  CategoricalPoint point; ///< Sampled categories.
  double loss; ///< Evaluated objective.
};

/// @brief Per-coordinate categorical probability vectors.
using Marginals = std::vector<std::vector<double>>;

/// @brief A binary chain node with a parameter for each parent value.
struct Conditional {
  std::size_t coordinate; ///< Coordinate to draw at this chain position.
  std::array<double, 2> one; ///< Conditional probabilities of one; root entries equal.
};

/// @brief Validate Cartesian product and controls for the active update rule.
/// @param objective Borrowed scoring callback.
/// @param sizes Number of categories for each axis.
/// @param options Search controls.
/// @throws std::invalid_argument Invalid inputs.
void validate(const CategoricalObjective& objective, const std::vector<std::size_t>& sizes,
              const DiscreteDistributionOptions& options) {
  if (!objective || sizes.empty() || options.evaluations == 0 ||
      !(options.exploration >= 0 && options.exploration <= 1) ||
      !std::ranges::all_of(sizes, [&](std::size_t size) {
        return options.method == DiscreteDistributionMethod::cross_entropy ? size != 0 : size == 2;
      })) {
    throw std::invalid_argument("Invalid discrete distribution search domain or budget");
  }
  const bool blend_valid = options.learning_rate > 0 && options.learning_rate <= 1;
  const bool elite_valid = options.population > 0 && options.elite_fraction > 0 && options.elite_fraction <= 1;
  bool valid = false;
  switch (options.method) {
    case DiscreteDistributionMethod::cross_entropy:
      valid = blend_valid && elite_valid;
      break;
    case DiscreteDistributionMethod::umda:
    case DiscreteDistributionMethod::mimic:
      valid = elite_valid;
      break;
    case DiscreteDistributionMethod::boa:
      valid = elite_valid && options.maximum_parents > 0 && options.maximum_parents <= 3;
      break;
    case DiscreteDistributionMethod::compact_ga:
      valid = options.virtual_population >= 2;
      break;
    case DiscreteDistributionMethod::pbil:
      valid = options.population > 0 && blend_valid &&
              options.mutation_probability >= 0 && options.mutation_probability <= 1 &&
              options.mutation_strength >= 0 && options.mutation_strength <= 1;
      break;
  }
  if (!valid) throw std::invalid_argument("Invalid discrete distribution update controls");
}

/// @brief Bernoulli entropy in nats.
/// @param fraction Probability of one.
/// @return Binary entropy, zero for deterministic components.
double entropy(double fraction) {
  if (fraction == 0 || fraction == 1) return 0;
  return -fraction * std::log(fraction) - (1 - fraction) * std::log1p(-fraction);
}

/// @brief Find the MIMIC chain and its conditional Bernoulli probabilities.
/// @param samples Sorted samples with the selected prefix.
/// @param count Positive count of selected samples.
/// @param dimension Number of binary coordinates.
/// @return Chain in ancestral order.
std::vector<Conditional> fit_chain(const std::vector<Sample>& samples, std::size_t count,
                                   std::size_t dimension) {
  std::vector<double> one(dimension);
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = 0; j < dimension; ++j) one[j] += samples[i].point[j];
  }
  for (auto& probability : one) probability /= count;
  std::size_t root = 0;
  for (std::size_t j = 1; j < dimension; ++j) {
    if (entropy(one[j]) < entropy(one[root])) root = j;
  }
  std::vector<bool> used(dimension);
  used[root] = true;
  std::vector<Conditional> chain{{root, {one[root], one[root]}}};
  std::vector<std::array<std::size_t, 2>> ones_by_parent(dimension);
  while (chain.size() < dimension) {
    const auto parent = chain.back().coordinate;
    std::array<std::size_t, 2> totals{};
    std::ranges::fill(ones_by_parent, std::array<std::size_t, 2>{});
    for (std::size_t i = 0; i < count; ++i) {
      const auto& point = samples[i].point;
      const auto bit = point[parent];
      ++totals[bit];
      for (std::size_t j = 0; j < dimension; ++j) {
        if (!used[j]) ones_by_parent[j][bit] += point[j];
      }
    }
    double best_entropy = std::numeric_limits<double>::infinity();
    Conditional chosen{};
    for (std::size_t j = 0; j < dimension; ++j) {
      if (used[j]) continue;
      std::array<double, 2> conditional{};
      double expected_entropy = 0;
      for (std::size_t bit = 0; bit < 2; ++bit) {
        conditional[bit] = totals[bit] == 0 ? one[j]
            : static_cast<double>(ones_by_parent[j][bit]) / totals[bit];
        expected_entropy += totals[bit] * entropy(conditional[bit]);
      }
      if (expected_entropy < best_entropy) {
        chosen = {j, conditional};
        best_entropy = expected_entropy;
      }
    }
    chain.push_back(chosen);
    used[chosen.coordinate] = true;
  }
  return chain;
}

/// @brief Binary network node with a probability for each assignment of its parents.
struct BinaryNode {
  std::size_t coordinate; ///< Coordinate to generate.
  std::vector<std::size_t> parents; ///< Earlier coordinates furnishing the conditioning assignment.
  std::vector<double> one; ///< Smoothed probability of one in every parent configuration.
};

/// @brief Encode a parent assignment as a zero-based binary table row.
/// @param point Complete or partially generated point with parents already assigned.
/// @param parents Parent coordinate list in table order.
/// @return Parent-state index.
std::size_t row_index(const CategoricalPoint& point, const std::vector<std::size_t>& parents) {
  std::size_t index = 0;
  for (const auto parent : parents) index = 2 * index + point[parent];
  return index;
}

/// @brief Count conditional binary observations in the elite sample.
/// @param samples Population sorted by loss.
/// @param elites Positive length of the selected prefix.
/// @param child Conditioned coordinate.
/// @param parents Candidate parent list.
/// @return Value-zero and value-one frequencies at each parent assignment.
auto conditional_counts(const std::vector<Sample>& samples, std::size_t elites,
                        std::size_t child, const std::vector<std::size_t>& parents) {
  std::vector<std::array<std::size_t, 2>> counts(std::size_t{1} << parents.size());
  for (std::size_t i = 0; i < elites; ++i) {
    const auto& point = samples[i].point;
    ++counts[row_index(point, parents)][point[child]];
  }
  return counts;
}

/// @brief Score one conditional table using Bayesian information criterion.
/// @param counts Frequencies for binary outcomes by parent state.
/// @param n Number of elite observations.
/// @return Log likelihood less half the parameter count times log sample size.
double bic(const std::vector<std::array<std::size_t, 2>>& counts, std::size_t n) {
  double value = -0.5 * counts.size() * std::log(static_cast<double>(n));
  for (const auto& row : counts) {
    const auto total = static_cast<double>(row[0] + row[1]);
    for (const auto observations : row) {
      if (observations != 0) value += observations * std::log(observations / total);
    }
  }
  return value;
}

/// @brief Fit an acyclic binary model by greedy local-score improvements.
/// @param samples Scored samples sorted by loss.
/// @param elites Positive selected prefix size.
/// @param dimensions Number of binary axes.
/// @param parent_limit Maximum in-degree.
/// @return Learned conditional tables in topological sampling order.
std::vector<BinaryNode> fit_network(const std::vector<Sample>& samples, std::size_t elites,
                                    std::size_t dimensions, std::size_t parent_limit) {
  std::vector<std::vector<std::size_t>> parents(dimensions);
  std::vector<double> current(dimensions);
  for (std::size_t child = 0; child < dimensions; ++child) {
    current[child] = bic(conditional_counts(samples, elites, child, parents[child]), elites);
  }
  for (;;) {
    double improvement = 1e-12;
    std::size_t chosen_child = dimensions, chosen_parent = 0;
    double chosen_score = 0;
    for (std::size_t child = 0; child < dimensions; ++child) {
      auto candidates = parents[child];
      if (candidates.size() == parent_limit) continue;
      // A candidate parent may be neither the child nor a descendant of the child.
      std::vector<bool> forbidden(dimensions);
      forbidden[child] = true;
      for (const auto parent : candidates) forbidden[parent] = true;
      std::vector<std::size_t> stack{child};
      while (!stack.empty()) {
        const auto ancestor = stack.back();
        stack.pop_back();
        for (std::size_t node = 0; node < dimensions; ++node) {
          if (!forbidden[node] && std::ranges::find(parents[node], ancestor) != parents[node].end()) {
            forbidden[node] = true;
            stack.push_back(node);
          }
        }
      }
      candidates.push_back(0);
      for (std::size_t parent = 0; parent < dimensions; ++parent) {
        if (forbidden[parent]) continue;
        candidates.back() = parent;
        const auto value = bic(conditional_counts(samples, elites, child, candidates), elites);
        const auto gain = value - current[child];
        if (gain > improvement) {
          improvement = gain;
          chosen_score = value;
          chosen_child = child;
          chosen_parent = parent;
        }
      }
    }
    if (chosen_child == dimensions) break;
    parents[chosen_child].push_back(chosen_parent);
    current[chosen_child] = chosen_score;
  }

  std::vector<bool> visited(dimensions);
  std::vector<BinaryNode> order;
  while (order.size() < dimensions) {
    for (std::size_t child = 0; child < dimensions; ++child) {
      if (visited[child] || !std::ranges::all_of(parents[child], [&](auto parent) { return visited[parent]; })) {
        continue;
      }
      const auto counts = conditional_counts(samples, elites, child, parents[child]);
      std::vector<double> estimates;
      for (const auto& row : counts) estimates.push_back((row[1] + 0.5) / (row[0] + row[1] + 1.0));
      order.push_back({child, parents[child], std::move(estimates)});
      visited[child] = true;
    }
  }
  return order;
}

/// @brief Fit product marginals on the selected sample prefix.
/// @param marginals Current distribution, blended in place.
/// @param samples Ranked samples.
/// @param count Positive selected sample count.
/// @param learning_rate Weight of the new empirical distribution.
void fit_marginals(Marginals& marginals, const std::vector<Sample>& samples, std::size_t count,
                   double learning_rate) {
  for (std::size_t j = 0; j < marginals.size(); ++j) {
    auto& prior = marginals[j];
    std::vector<std::size_t> counts(prior.size());
    for (std::size_t i = 0; i < count; ++i) ++counts[samples[i].point[j]];
    for (std::size_t k = 0; k < prior.size(); ++k) {
      prior[k] = std::lerp(prior[k], static_cast<double>(counts[k]) / count, learning_rate);
    }
  }
}

} // namespace

DiscreteDistributionResult discrete_distribution_search(
    const CategoricalObjective& objective, const std::vector<std::size_t>& cardinalities,
    const DiscreteDistributionOptions& options) {
  validate(objective, cardinalities, options);
  const auto dimension = cardinalities.size();
  const auto method = options.method;
  const auto limit = method == DiscreteDistributionMethod::compact_ga ? 2 : options.population;
  Marginals marginals;
  marginals.reserve(dimension);
  for (const auto size : cardinalities) marginals.emplace_back(size, 1.0 / size);
  std::vector<Conditional> chain;
  std::vector<BinaryNode> network;
  DiscreteDistributionResult result{{}, std::numeric_limits<double>::infinity(), 0};
  std::mt19937_64 rng(options.seed);
  std::uniform_real_distribution<double> uniform;
  std::vector<Sample> samples;
  samples.reserve(std::min(limit, options.evaluations));
  while (result.evaluations < options.evaluations) {
    const auto count = std::min(limit, options.evaluations - result.evaluations);
    samples.clear();
    std::vector<std::discrete_distribution<std::size_t>> product;
    if (chain.empty() && network.empty()) {
      product.reserve(dimension);
      for (const auto& prior : marginals) product.emplace_back(prior.begin(), prior.end());
    }
    for (std::size_t i = 0; i < count; ++i) {
      CategoricalPoint point(dimension);
      if (chain.empty() && network.empty()) {
        for (std::size_t j = 0; j < dimension; ++j) {
          if (uniform(rng) < options.exploration) {
            point[j] = std::uniform_int_distribution<std::size_t>(0, cardinalities[j] - 1)(rng);
          } else {
            point[j] = product[j](rng);
          }
        }
      } else if (!network.empty()) {
        for (const auto& node : network) {
          const auto probability = std::lerp(node.one[row_index(point, node.parents)], 0.5, options.exploration);
          point[node.coordinate] = uniform(rng) < probability;
        }
      } else {
        std::size_t parent = 0;
        for (const auto& conditional : chain) {
          const double probability = std::lerp(conditional.one[parent], 0.5, options.exploration);
          parent = uniform(rng) < probability;
          point[conditional.coordinate] = parent;
        }
      }
      const auto loss = objective(point);
      if (!std::isfinite(loss)) throw std::domain_error("Discrete distribution search requires finite loss");
      ++result.evaluations;
      if (loss < result.value) {
        result.point = point;
        result.value = loss;
      }
      samples.push_back({std::move(point), loss});
    }
    if (result.evaluations == options.evaluations) break;
    std::stable_sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b) {
      return a.loss < b.loss;
    });
    if (method == DiscreteDistributionMethod::compact_ga) {
      if (samples[0].loss == samples[1].loss) continue;
      const double step = 1.0 / options.virtual_population;
      for (std::size_t j = 0; j < dimension; ++j) {
        const double direction = static_cast<double>(samples[0].point[j]) - samples[1].point[j];
        const double probability = std::clamp(marginals[j][1] + step * direction, 0.0, 1.0);
        marginals[j] = {1 - probability, probability};
      }
    } else if (method == DiscreteDistributionMethod::pbil) {
      fit_marginals(marginals, samples, 1, options.learning_rate);
      for (auto& axis : marginals) {
        if (uniform(rng) < options.mutation_probability) {
          const double probability = std::lerp(axis[1], uniform(rng) < 0.5 ? 1.0 : 0.0,
                                               options.mutation_strength);
          axis = {1 - probability, probability};
        }
      }
    } else {
      const auto elites = std::min(count, static_cast<std::size_t>(std::ceil(options.elite_fraction * count)));
      if (method == DiscreteDistributionMethod::mimic) {
        chain = fit_chain(samples, elites, dimension);
      } else if (method == DiscreteDistributionMethod::boa) {
        network = fit_network(samples, elites, dimension, options.maximum_parents);
      } else {
        const double learning_rate = method == DiscreteDistributionMethod::umda ? 1 : options.learning_rate;
        fit_marginals(marginals, samples, elites, learning_rate);
      }
    }
  }
  return result;
}

} // namespace datamunge::optim
