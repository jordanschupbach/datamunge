/// @file
/// @brief Mixed-domain Parzen estimators, randomized regression forests and resource composition.

#include <datamunge/optim/mixed_search.hpp>
#include <datamunge/optim/rbf_gaussian_process_surrogate.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::optim {
/// @brief Encoded conditional locations and modeling internals.
namespace {

/// @brief Inactive sentinel, normalized numerical feature or categorical index per dimension.
using Encoded = std::vector<double>;

/// @brief Training input and its real objective value.
struct Row {
  Encoded point; ///< Internally feasible encoded features.
  double loss; ///< Observed raw loss.
};

/// @brief Conditional-space operations and local entropy source.
struct Space {
  const std::vector<MixedParameter>& parameters; ///< Valid ordered domain.
  std::mt19937_64 rng; ///< Borrowed-workflow local engine.
  std::uniform_real_distribution<double> uniform; ///< Unit uniform samples.
  std::normal_distribution<double> gaussian; ///< Standard normal samples.

  /// @brief Determine whether earlier parameters activate a feature.
  /// @param point Partially assembled configuration.
  /// @param axis Feature index.
  /// @return True if the feature must be assigned.
  bool active(const Encoded& point, std::size_t axis) const {
    const auto& condition = parameters[axis].condition;
    return !condition || point[condition->parent] == static_cast<double>(condition->category);
  }

  /// @brief Draw one active value from the domain prior.
  /// @param axis Feature index.
  /// @return Normalized numeric value or encoded category.
  double sample(std::size_t axis) {
    const auto& parameter = parameters[axis];
    if (parameter.kind == ParameterKind::categorical) {
      return static_cast<double>(std::uniform_int_distribution<std::size_t>(0, parameter.categories - 1)(rng));
    }
    if (parameter.lower == parameter.upper) return 0;
    if (parameter.kind == ParameterKind::integer) {
      const auto n = std::uniform_int_distribution<std::int64_t>(static_cast<std::int64_t>(parameter.lower),
                                                                   static_cast<std::int64_t>(parameter.upper))(rng);
      return (static_cast<double>(n) - parameter.lower) / (parameter.upper - parameter.lower);
    }
    return uniform(rng);
  }

  /// @brief Materialize normalized features or perturb an existing configuration.
  /// @param base Optional existing point; null draws pure prior.
  /// @param scale Width of local numeric perturbation, when a base is supplied.
  /// @return Feasible configuration in internal representation.
  Encoded propose(const Encoded* base = nullptr, double scale = 0.2) {
    Encoded result(parameters.size(), -1);
    const auto chosen = base ? std::uniform_int_distribution<std::size_t>(0, parameters.size() - 1)(rng) : 0;
    for (std::size_t j = 0; j < parameters.size(); ++j) {
      if (!active(result, j)) continue;
      if (!base || (*base)[j] < 0) {
        result[j] = sample(j);
      } else if (j == chosen || uniform(rng) < 1.0 / parameters.size()) {
        result[j] = parameters[j].kind == ParameterKind::categorical ? sample(j)
            : std::clamp((*base)[j] + scale * gaussian(rng), 0.0, 1.0);
      } else {
        result[j] = (*base)[j];
      }
    }
    return result;
  }

  /// @brief Convert to typed features and canonicalize integer coordinates for modeling.
  /// @param point Mutable encoded location whose numeric discrete axes are canonicalized.
  /// @return Feasible typed feature values, including monostate for missing axes.
  MixedConfiguration decode(Encoded& point) const {
    MixedConfiguration result(parameters.size());
    for (std::size_t j = 0; j < parameters.size(); ++j) {
      if (point[j] < 0) continue;
      const auto& p = parameters[j];
      if (p.kind == ParameterKind::categorical) {
        result[j] = static_cast<std::size_t>(point[j]);
      } else if (p.kind == ParameterKind::integer) {
        const auto value = std::round(std::lerp(p.lower, p.upper, point[j]));
        result[j] = static_cast<std::int64_t>(value);
        point[j] = p.lower == p.upper ? 0 : (value - p.lower) / (p.upper - p.lower);
      } else {
        result[j] = std::lerp(p.lower, p.upper, point[j]);
      }
    }
    return result;
  }
};

/// @brief Validate shapes, parent order and finite typed supports.
/// @param parameters Ordered candidate-space definition.
/// @throws std::invalid_argument Invalid or empty domain.
void validate(const std::vector<MixedParameter>& parameters) {
  if (parameters.empty()) throw std::invalid_argument("Mixed space requires at least one parameter");
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    const auto& p = parameters[i];
    if (p.kind == ParameterKind::continuous || p.kind == ParameterKind::integer) {
      if (!std::isfinite(p.lower) || !std::isfinite(p.upper) || p.lower > p.upper ||
          (p.kind == ParameterKind::integer &&
           (p.lower != std::trunc(p.lower) || p.upper != std::trunc(p.upper) ||
            std::max(std::abs(p.lower), std::abs(p.upper)) > 0x1p52))) {
        throw std::invalid_argument("Invalid numeric mixed-space bounds");
      }
    } else if (p.kind != ParameterKind::categorical || p.categories == 0 ||
               p.categories > (std::uint64_t{1} << 53)) {
      throw std::invalid_argument("Invalid categorical mixed-space support");
    }
    if (p.condition) {
      const auto [parent, category] = *p.condition;
      if (parent >= i || parameters[parent].kind != ParameterKind::categorical ||
          category >= parameters[parent].categories) {
        throw std::invalid_argument("Conditional parameters require an earlier categorical parent");
      }
    }
  }
}

/// @brief Validate the budget-independent model and acquisition controls.
/// @param options Model settings.
/// @param method Active mixed optimizer.
/// @throws std::invalid_argument Invalid active options.
void validate_model(const MixedSearchOptions& options, MixedSearchMethod method) {
  if (options.startup < 2 || options.candidates == 0 ||
      !(options.random_fraction >= 0 && options.random_fraction <= 1) ||
      (method == MixedSearchMethod::tpe && !(options.elite_fraction > 0 && options.elite_fraction < 1)) ||
      (method != MixedSearchMethod::tpe && options.trees < 2) ||
      (method == MixedSearchMethod::surrogate_evolution &&
       (options.candidates < 2 || options.evolution_steps == 0)) ||
      (method != MixedSearchMethod::tpe && method != MixedSearchMethod::smac &&
       method != MixedSearchMethod::surrogate_evolution)) {
    throw std::invalid_argument("Invalid mixed model settings");
  }
}

/// @brief Count one acquisition or surrogate query.
/// @param total Work counter to increment.
/// @throws std::overflow_error Counter is already saturated.
void count_query(std::size_t& total) {
  if (total == std::numeric_limits<std::size_t>::max()) throw std::overflow_error("Model work counter exhausted");
  ++total;
}

/// @brief Per-feature samples conditioned on activation.
using Marginals = std::vector<std::vector<double>>;

/// @brief Estimate smoothing bandwidth for bounded numeric Parzen kernels.
/// @param count Number of active samples at this coordinate.
/// @return Positive normalized bandwidth.
double bandwidth(std::size_t count) {
  return std::clamp(0.3 * std::pow(static_cast<double>(std::max(std::size_t{1}, count)), -0.2), 0.04, 0.3);
}

/// @brief Compute one conditional marginal density with a uniform prior component.
/// @param p Parameter definition.
/// @param samples Encoded active observations.
/// @param value Candidate active encoded feature.
/// @return Positive density or categorical mass.
double density(const MixedParameter& p, const std::vector<double>& samples, double value) {
  if (p.kind == ParameterKind::categorical) {
    return (1.0 / p.categories + std::ranges::count(samples, value)) / (samples.size() + 1.0);
  }
  const auto scale = bandwidth(samples.size());
  double total = 1;
  for (const auto center : samples) {
    const auto z = (value - center) / scale;
    const auto retained = 0.5 * (std::erf((1 - center) / (scale * std::numbers::sqrt2)) +
                                 std::erf(center / (scale * std::numbers::sqrt2)));
    total += std::exp(-0.5 * z * z) / (std::sqrt(2 * std::numbers::pi) * scale * retained);
  }
  return total / (samples.size() + 1.0);
}

/// @brief Construct a feasible proposal maximizing the log density ratio.
/// @param space Valid mixed domain and local engine.
/// @param history All real observations at a common fidelity.
/// @param options Parzen configuration.
/// @param queries Counter to increment for each candidate ranking.
/// @return Selected modeled proposal.
Encoded parzen(Space& space, const std::vector<Row>& history, const MixedSearchOptions& options,
               std::size_t& queries) {
  std::vector<std::size_t> order(history.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return history[a].loss < history[b].loss; });
  const auto count = std::clamp(static_cast<std::size_t>(std::ceil(options.elite_fraction * order.size())),
                                std::size_t{1}, order.size() - 1);
  Marginals good(space.parameters.size()), bad(space.parameters.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    const auto& point = history[order[i]].point;
    auto& group = i < count ? good : bad;
    for (std::size_t j = 0; j < point.size(); ++j) {
      if (point[j] >= 0) group[j].push_back(point[j]);
    }
  }

  Encoded best;
  double highest = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < options.candidates; ++i) {
    Encoded point(good.size(), -1);
    double value = 0;
    for (std::size_t j = 0; j < point.size(); ++j) {
      if (!space.active(point, j)) continue;
      const auto& samples = good[j];
      const auto index = std::uniform_int_distribution<std::size_t>(0, samples.size())(space.rng);
      const auto& parameter = space.parameters[j];
      if (index == samples.size()) point[j] = space.sample(j);
      else if (parameter.kind == ParameterKind::categorical) point[j] = samples[index];
      else {
        do {
          point[j] = samples[index] + bandwidth(samples.size()) * space.gaussian(space.rng);
        } while (point[j] < 0 || point[j] > 1);
      }
      if (parameter.kind == ParameterKind::integer) {
        const auto integral = std::round(std::lerp(parameter.lower, parameter.upper, point[j]));
        point[j] = parameter.lower == parameter.upper ? 0 :
          (integral - parameter.lower) / (parameter.upper - parameter.lower);
      }
      value += std::log(density(parameter, good[j], point[j])) - std::log(density(parameter, bad[j], point[j]));
    }
    count_query(queries);
    if (value > highest) {
      highest = value;
      best = std::move(point);
    }
  }
  return best;
}

/// @brief Binary split of a bootstrapped regression tree.
struct TreeNode {
  double mean; ///< Mean normalized target of the node.
  std::size_t axis{}; ///< Split feature.
  double cut{}; ///< Threshold or category, including inactive sentinel.
  bool exact{}; ///< Use equality rather than a threshold.
  std::size_t left{}; ///< Left child index, zero for a leaf.
  std::size_t right{}; ///< Right child index.
};

/// @brief Test a feature in a learned split.
/// @param value Encoded feature.
/// @param cut Level or threshold.
/// @param exact Whether to test equality.
/// @return True for the left partition.
bool goes_left(double value, double cut, bool exact) {
  return exact ? value == cut : value <= cut;
}

/// @brief Create a randomized regression node, minimizing total child squared error.
/// @param space Domain and local engine.
/// @param rows Active real observations.
/// @param targets Normalized losses for those observations.
/// @param sample Bootstrap draw indices at this node.
/// @param tree Existing tree to extend.
/// @param depth Remaining recursion depth.
/// @return Index of the newly fitted node.
std::size_t tree_node(Space& space, const std::vector<Row>& rows, const std::vector<double>& targets,
                      const std::vector<std::size_t>& sample, std::vector<TreeNode>& tree, unsigned depth) {
  double sum = 0;
  for (const auto i : sample) sum += targets[i];
  const auto index = tree.size();
  tree.push_back({sum / sample.size()});
  if (sample.size() < 4 || depth == 0) return index;

  std::uniform_int_distribution<std::size_t> axis(0, space.parameters.size() - 1);
  std::uniform_int_distribution<std::size_t> draw(0, sample.size() - 1);
  double gain = 1e-12;
  auto best = tree[index];
  for (std::size_t candidate = 0; candidate < 2 + std::sqrt(space.parameters.size()); ++candidate) {
    const auto feature = axis(space.rng);
    const auto a = rows[sample[draw(space.rng)]].point[feature];
    const auto b = rows[sample[draw(space.rng)]].point[feature];
    const bool exact = space.parameters[feature].kind == ParameterKind::categorical;
    if (a == b) continue;
    const auto cut = exact ? a : std::midpoint(a, b);
    std::size_t count = 0;
    double left = 0;
    for (const auto i : sample) {
      if (goes_left(rows[i].point[feature], cut, exact)) {
        ++count;
        left += targets[i];
      }
    }
    if (count < 2 || sample.size() - count < 2) continue;
    const auto right = sum - left;
    const auto improvement = left*left/count + right*right/(sample.size() - count) - sum*sum/sample.size();
    if (improvement > gain) {
      gain = improvement;
      best.axis = feature;
      best.cut = cut;
      best.exact = exact;
    }
  }
  if (gain <= 1e-12) return index;
  std::vector<std::size_t> left, right;
  for (const auto i : sample) {
    (goes_left(rows[i].point[best.axis], best.cut, best.exact) ? left : right).push_back(i);
  }
  best.left = tree_node(space, rows, targets, left, tree, depth - 1);
  best.right = tree_node(space, rows, targets, right, tree, depth - 1);
  tree[index] = best;
  return index;
}

/// @brief Owned bootstrap forest for a fixed model fit.
using Forest = std::vector<std::vector<TreeNode>>;

/// @brief Refit a forest on all currently available true observations.
/// @param space Domain and local engine.
/// @param rows Real observed trials.
/// @param count Positive number of bootstrapped trees.
/// @return Fitted predictive forest over standardized losses.
Forest fit_forest(Space& space, const std::vector<Row>& rows, std::size_t count) {
  const auto [low, high] = std::ranges::minmax_element(rows, {}, &Row::loss);
  const auto lo = low->loss / 2, hi = high->loss / 2;
  std::vector<double> targets;
  for (const auto& row : rows) targets.push_back(hi > lo ? (row.loss / 2 - lo) / (hi - lo) : 0);
  std::uniform_int_distribution<std::size_t> draw(0, rows.size() - 1);
  Forest forest(count);
  for (auto& tree : forest) {
    std::vector<std::size_t> sample(rows.size());
    for (auto& i : sample) i = draw(space.rng);
    tree_node(space, rows, targets, sample, tree, 7);
  }
  return forest;
}

/// @brief Predict ensemble mean and standard deviation for a feature vector.
/// @param forest Fitted forest.
/// @param point Feasible internal point.
/// @return Latent ensemble mean and spread.
std::pair<double, double> predict(const Forest& forest, const Encoded& point) {
  double mean = 0, square = 0;
  for (const auto& tree : forest) {
    auto node = tree.front();
    while (node.left != 0) {
      node = tree[goes_left(point[node.axis], node.cut, node.exact) ? node.left : node.right];
    }
    mean += node.mean / forest.size();
    square += node.mean * node.mean / forest.size();
  }
  return {mean, std::sqrt(std::max(0.0, square - mean * mean))};
}

/// @brief Choose the next true evaluation through a fitted forest.
/// @param space Valid mixed domain and local engine.
/// @param past Real scored rows.
/// @param options Forest and candidate-pool settings.
/// @param queries Count of model calls, incremented on each prediction.
/// @return Highest-acquisition or best-surrogate configuration.
Encoded forest_proposal(Space& space, const std::vector<Row>& past, const MixedSearchOptions& options,
                        std::size_t& queries) {
  const auto forest = fit_forest(space, past, options.trees);
  const auto& elite = std::ranges::min_element(past, {}, &Row::loss)->point;
  const bool evolve = options.method == MixedSearchMethod::surrogate_evolution;
  std::vector<Row> pool;
  /// Score one point on the surrogate, using minimization ordering.
  const auto score = [&](Encoded point) {
    count_query(queries);
    const auto [mean, sd] = predict(forest, point);
    const auto value = evolve ? mean : -expected_improvement(mean, std::max(sd, 1e-8), 0);
    return Row{std::move(point), value};
  };
  for (std::size_t i = 0; i < options.candidates; ++i) {
    pool.push_back(score(space.propose(i % 2 == 0 ? nullptr : &elite)));
  }
  for (std::size_t generation = 0; evolve && generation < options.evolution_steps; ++generation) {
    std::stable_sort(pool.begin(), pool.end(), [](const Row& a, const Row& b) { return a.loss < b.loss; });
    const auto survivors = std::max(std::size_t{1}, pool.size() / 2);
    pool.resize(survivors);
    std::uniform_int_distribution<std::size_t> draw(0, survivors - 1);
    while (pool.size() < options.candidates) {
      auto point = space.propose(&pool[draw(space.rng)].point, 0.2 / (1 + generation));
      pool.push_back(score(std::move(point)));
    }
  }
  return std::move(std::ranges::min_element(pool, {}, &Row::loss)->point);
}

} // namespace

MixedSearchResult mixed_search(const MixedObjective& objective, const std::vector<MixedParameter>& parameters,
                               const MixedSearchOptions& options) {
  validate(parameters);
  validate_model(options, options.method);
  if (!objective || options.evaluations == 0) throw std::invalid_argument("Mixed search requires a callback and budget");
  Space space{parameters, std::mt19937_64(options.seed), {}, {}};
  MixedSearchResult result{{}, std::numeric_limits<double>::infinity(), 0, 0};
  std::vector<Row> history;
  while (result.evaluations < options.evaluations) {
    Encoded point;
    if (history.size() < options.startup || space.uniform(space.rng) < options.random_fraction) {
      point = space.propose();
    } else if (options.method == MixedSearchMethod::tpe) {
      point = parzen(space, history, options, result.model_evaluations);
    } else {
      point = forest_proposal(space, history, options, result.model_evaluations);
    }
    auto decoded = space.decode(point);
    const auto loss = objective(decoded);
    if (!std::isfinite(loss)) throw std::domain_error("Mixed search loss must be finite");
    ++result.evaluations;
    if (loss < result.value) {
      result.point = std::move(decoded);
      result.value = loss;
    }
    history.push_back({std::move(point), loss});
  }
  return result;
}

BohbResult bohb_search(const StartMixedTrial& start, const AdvanceTrial& advance,
                      const std::vector<MixedParameter>& parameters, ResourceSearchOptions resource,
                      const MixedSearchOptions& model) {
  validate(parameters);
  validate_model(model, MixedSearchMethod::tpe);
  if (!start || !advance) throw std::invalid_argument("BOHB requires start and advance callbacks");
  BohbResult result;
  std::vector<Encoded> encoded;
  std::map<std::size_t, std::vector<Row>, std::greater<>> history;
  std::size_t queries = 0;
  resource.method = ResourceSearchMethod::hyperband;
  /// Start from the highest fidelity with enough same-rung observations.
  const auto configure = [&](std::size_t id, std::uint64_t seed) {
    Space space{parameters, std::mt19937_64(seed), {}, {}};
    const auto found = std::ranges::find_if(history, [&](const auto& rung) {
      return rung.second.size() >= model.startup;
    });
    const bool use_model = found != history.end() && space.uniform(space.rng) >= model.random_fraction;
    auto point = use_model ? parzen(space, found->second, model, queries) : space.propose();
    result.configurations.push_back(space.decode(point));
    encoded.push_back(std::move(point));
    result.model_starts += use_model;
    start(id, result.configurations.back(), seed);
  };
  /// Record the previously configured trial in the fidelity-specific training data.
  const auto train = [&](std::size_t id, std::size_t previous, std::size_t target) {
    const auto loss = advance(id, previous, target);
    history[target].push_back({encoded[id], loss});
    return loss;
  };
  result.schedule = resource_search(configure, train, resource);
  return result;
}

MixedSearchResult iterated_racing(const MixedSimulation& simulation, const std::vector<MixedParameter>& parameters,
                                  const IteratedRacingOptions& options) {
  validate(parameters);
  if (!simulation || options.evaluations == 0 || options.stage_budget == 0 || options.candidates < 2 ||
      options.pilot < 2 || !(options.alpha > 0 && options.alpha < 0.5)) {
    throw std::invalid_argument("Invalid iterated racing controls");
  }
  Space space{parameters, std::mt19937_64(options.seed), {}, {}};
  MixedSearchResult result;
  Encoded winner;
  std::size_t stage = 0;
  while (result.evaluations < options.evaluations) {
    const auto cap = std::min(options.stage_budget, options.evaluations - result.evaluations);
    const auto count = std::min(options.candidates, cap);
    std::vector<Encoded> tested;
    std::vector<MixedConfiguration> decoded;
    if (!winner.empty()) tested.push_back(winner);
    while (tested.size() < count) {
      const auto* base = winner.empty() || space.uniform(space.rng) < 0.2 ? nullptr : &winner;
      tested.push_back(space.propose(base, 0.3 / std::sqrt(1.0 + stage)));
    }
    for (auto& point : tested) decoded.push_back(space.decode(point));
    SimulationSearchOptions controls{cap, space.rng(), SimulationSearchMethod::friedman_race,
                                      options.pilot, options.alpha};
    const auto race = simulation_search([&](std::size_t i, std::uint64_t seed) {
      return simulation(decoded[i], seed);
    }, count, controls);
    winner = std::move(tested[*race.selected]);
    result.point = std::move(decoded[*race.selected]);
    result.value = race.estimates[*race.selected].mean;
    result.evaluations += race.evaluations;
    ++stage;
  }
  return result;
}

} // namespace datamunge::optim
