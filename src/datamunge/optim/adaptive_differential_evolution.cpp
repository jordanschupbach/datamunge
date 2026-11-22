/// @file
/// @brief Adaptive differential evolution mutation and control histories.

#include <datamunge/optim/adaptive_differential_evolution.hpp>

#include "boxed_search_detail.hpp"
#include "differential_evolution_detail.hpp"

#include <deque>
#include <numeric>
#include <utility>

namespace datamunge::optim {
/// @brief Private population and adaptation state.
namespace {

/// @brief Mutation scale and crossover probability.
struct Control {
  double scale{0.5}; ///< Differential multiplier.
  double crossover{0.5}; ///< Per-axis donor probability.
};

/// @brief One evaluated population member.
struct Member {
  std::vector<double> point; ///< Position in the unit box.
  double value; ///< Evaluated objective.
  Control control; ///< Inherited jDE parameters.
};

/// @brief Response observed from one strictly improving offspring.
struct Success {
  Control control; ///< Used control values.
  long double improvement; ///< Parent loss minus trial loss.
};

/// @brief One generation of SaDE strategy outcomes.
struct StrategyOutcomes {
  std::array<std::size_t, 4> attempts{}; ///< Constructed proposals by strategy.
  std::array<std::vector<double>, 4> crossovers; ///< Crossover rates of successful proposals.
};

/// @brief Compute current SaDE strategy weights and crossover medians.
/// @param window Recent generation outcomes.
/// @param weights Output strategy weights.
/// @param centers Output crossover-rate centers.
void strategy_parameters(const std::deque<StrategyOutcomes>& window,
                         std::array<double, 4>& weights, std::array<double, 4>& centers) {
  for (std::size_t strategy = 0; strategy < weights.size(); ++strategy) {
    std::size_t attempts = 0;
    std::vector<double> rates;
    for (const auto& outcomes : window) {
      attempts += outcomes.attempts[strategy];
      const auto& accepted = outcomes.crossovers[strategy];
      rates.insert(rates.end(), accepted.begin(), accepted.end());
    }
    weights[strategy] = window.empty() ? 1 : 0.01 + (attempts == 0 ? 0 : static_cast<double>(rates.size()) / attempts);
    if (!rates.empty()) {
      const auto middle = rates.begin() + rates.size() / 2;
      std::nth_element(rates.begin(), middle, rates.end());
      centers[strategy] = *middle;
      if (rates.size() % 2 == 0) {
        centers[strategy] = std::midpoint(centers[strategy], *std::max_element(rates.begin(), middle));
      }
    }
  }
}

/// @brief Draw a positive scale from a distribution and truncate it at the upper limit.
/// @tparam Distribution Continuous sampling distribution.
/// @param distribution Distribution used until a positive scale is drawn.
/// @param rng Borrowed local engine.
/// @param maximum Positive upper limit.
/// @return Positive capped scale.
template <class Distribution>
double draw_scale(Distribution distribution, std::mt19937_64& rng, double maximum) {
  double value = 0;
  while (value <= 0) value = distribution(rng);
  return std::min(value, maximum);
}

/// @brief Draw one bounded crossover rate from a normal packet.
/// @param center Rate around which new values are sampled.
/// @param rng Borrowed local engine.
/// @return Gaussian draw clipped to [0, 1].
double draw_crossover(double center, std::mt19937_64& rng) {
  return std::clamp(std::normal_distribution<double>(center, 0.1)(rng), 0.0, 1.0);
}

/// @brief Fit scale and crossover statistics from successful updates.
/// @param successes Nonempty set of strictly improving trials.
/// @param method Variant determining uniform or weighted statistics and crossover rule.
/// @return Lehmer-mean scale and fitted crossover mean.
Control fit_controls(const std::vector<Success>& successes, AdaptiveDEMethod method) {
  long double largest = 1;
  for (const auto& success : successes) largest = std::max(largest, success.improvement);
  long double weight_sum = 0, scale_sum = 0, scale_squared = 0, crossover_sum = 0, crossover_squared = 0;
  for (const auto& success : successes) {
    const auto weight = method == AdaptiveDEMethod::jade ? 1 : success.improvement / largest;
    const auto f = success.control.scale;
    const auto cr = success.control.crossover;
    weight_sum += weight;
    scale_sum += weight * f;
    scale_squared += weight * f * f;
    crossover_sum += weight * cr;
    crossover_squared += weight * cr * cr;
  }
  const double crossover = method == AdaptiveDEMethod::lshade
      ? (crossover_sum > 0 ? static_cast<double>(crossover_squared / crossover_sum) : -1)
      : static_cast<double>(crossover_sum / weight_sum);
  return {static_cast<double>(scale_squared / scale_sum), crossover};
}

/// @brief Build stable population ranks from smallest loss to largest.
/// @param population Evaluated members.
/// @return Member indices ranked by loss.
std::vector<std::size_t> rank(const std::vector<Member>& population) {
  std::vector<std::size_t> order(population.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return population[a].value < population[b].value;
  });
  return order;
}

} // namespace

AdaptiveDifferentialEvolution::AdaptiveDifferentialEvolution(AdaptiveDEOptions options) : options_(options) {}

double AdaptiveDifferentialEvolution::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                                const std::vector<double>& lower,
                                                const std::vector<double>& upper) const {
  const auto dimension = coordinates.size();
  detail::validate_box(dimension, lower, upper, options_.evaluations);
  const auto method = options_.method;
  const bool jade = method == AdaptiveDEMethod::jade;
  const bool shade = method == AdaptiveDEMethod::shade;
  const bool linear = method == AdaptiveDEMethod::lshade;
  const bool history = shade || linear;
  const bool sade = method == AdaptiveDEMethod::sade;
  const bool jde = method == AdaptiveDEMethod::jde;
  const bool pbest = jade || history;
  if (options_.population < 6 || !(pbest || sade || jde) ||
      (history && options_.history_size == 0) || (sade && options_.learning_period == 0) ||
      ((jade || linear) && !(options_.pbest_fraction > 0 && options_.pbest_fraction <= 1)) ||
      (jade && !(options_.jade_learning > 0 && options_.jade_learning <= 1)) ||
      (linear && (options_.minimum_population < 4 || options_.minimum_population > options_.population))) {
    throw std::invalid_argument("Invalid adaptive differential evolution controls");
  }

  auto point = coordinates;
  auto initial = detail::normalize_start(point, lower, upper);
  detail::CountingObjective observe(function);
  std::mt19937_64 rng(options_.seed);
  std::uniform_real_distribution<double> uniform;
  /// Evaluate a normalized proposal through the shared objective accounting.
  auto evaluate = [&](const std::vector<double>& x) {
    for (std::size_t j = 0; j < dimension; ++j) point[j] = std::lerp(lower[j], upper[j], x[j]);
    return observe.evaluate(point);
  };

  std::vector<Member> population;
  population.reserve(std::min(options_.population, options_.evaluations));
  population.push_back({std::move(initial), observe.evaluate(point), {0.5, jde ? 0.9 : 0.5}});
  while (observe.calls < options_.evaluations && population.size() < options_.population) {
    std::vector<double> individual(dimension);
    for (auto& x : individual) x = uniform(rng);
    const auto value = evaluate(individual);
    population.push_back({std::move(individual), value, {0.5, jde ? 0.9 : 0.5}});
  }

  std::vector<Control> memories(history ? options_.history_size : 1);
  std::size_t next_memory = 0;
  std::vector<std::vector<double>> archive;
  std::deque<StrategyOutcomes> recent;
  std::array<double, 4> centers{0.5, 0.5, 0.5, 0.5};
  while (observe.calls < options_.evaluations) {
    const auto size = population.size();
    const auto to_score = std::min(options_.evaluations - observe.calls, size);
    const auto order = rank(population);
    std::vector<Member> next = population;
    std::vector<std::vector<double>> displaced;
    std::vector<Success> successes;
    std::array<double, 4> weights{1, 1, 1, 1};
    if (sade) strategy_parameters(recent, weights, centers);
    std::discrete_distribution<std::size_t> select_strategy(weights.begin(), weights.end());
    StrategyOutcomes outcomes;

    for (std::size_t target = 0; target < to_score; ++target) {
      const auto& parent = population[target];
      Control control = parent.control;
      std::size_t strategy = 0;
      if (jde) {
        if (uniform(rng) < 0.1) control.scale = 0.1 + 0.9 * uniform(rng);
        if (uniform(rng) < 0.1) control.crossover = uniform(rng);
      } else if (sade) {
        strategy = select_strategy(rng);
        ++outcomes.attempts[strategy];
        control = {draw_scale(std::normal_distribution<double>(0.5, 0.3), rng, 2),
                   draw_crossover(centers[strategy], rng)};
      } else {
        const auto index = std::uniform_int_distribution<std::size_t>(0, memories.size() - 1)(rng);
        const auto& center = memories[index];
        control = {draw_scale(std::cauchy_distribution<double>(center.scale, 0.1), rng, 1),
                   center.crossover < 0 ? 0 : draw_crossover(center.crossover, rng)};
      }

      std::size_t elite = order.front();
      std::array<std::size_t, 5> donors{};
      const std::vector<double>* second = nullptr;
      if (pbest) {
        const auto fraction = shade ? std::lerp(2.0 / size, 0.2, uniform(rng)) : options_.pbest_fraction;
        const auto top = std::clamp(static_cast<std::size_t>(std::ceil(fraction * size)), std::size_t{2}, size);
        elite = order[std::uniform_int_distribution<std::size_t>(0, top - 1)(rng)];
        donors = detail::random_distinct_indices(rng, size, target, 1);
        std::uniform_int_distribution<std::size_t> draw(0, size + archive.size() - 1);
        auto index = draw(rng);
        while (index == target || index == donors[0]) index = draw(rng);
        second = index < size ? &population[index].point : &archive[index - size];
      } else {
        donors = detail::random_distinct_indices(rng, size, target, strategy == 1 || strategy == 2 ? 5 : 3);
      }

      const auto axis = std::uniform_int_distribution<std::size_t>(0, dimension - 1)(rng);
      const auto interpolation = strategy == 3 ? uniform(rng) : 0;
      auto trial = parent.point;
      for (std::size_t j = 0; j < dimension; ++j) {
        if (lower[j] == upper[j] || (strategy != 3 && j != axis && uniform(rng) >= control.crossover)) continue;
        const auto f = control.scale;
        const auto a = population[donors[0]].point[j];
        const auto x = parent.point[j];
        double mutant;
        if (pbest) {
          mutant = x + f * (population[elite].point[j] - x) + f * (a - (*second)[j]);
          if (mutant < 0) mutant = 0.5 * x;
          if (mutant > 1) mutant = std::midpoint(x, 1.0);
        } else {
          const auto b = population[donors[1]].point[j];
          const auto c = population[donors[2]].point[j];
          if (strategy == 3) {
            mutant = x + interpolation * (a - x) + f * (b - c);
          } else {
            mutant = a + f * (b - c);
            if (strategy == 1 || strategy == 2) {
              mutant += f * (population[donors[3]].point[j] - population[donors[4]].point[j]);
            }
            if (strategy == 1) mutant += f * (population[elite].point[j] - a);
          }
        }
        trial[j] = std::clamp(mutant, 0.0, 1.0);
      }

      const auto value = evaluate(trial);
      if (value < parent.value) {
        if (pbest) {
          successes.push_back({control, static_cast<long double>(parent.value) - value});
          if (options_.archive) displaced.push_back(parent.point);
        }
        if (sade) outcomes.crossovers[strategy].push_back(control.crossover);
      }
      if (value <= parent.value) next[target] = {std::move(trial), value, control};
    }

    if (sade) {
      recent.push_back(std::move(outcomes));
      if (recent.size() > options_.learning_period) recent.pop_front();
    }
    if (!successes.empty()) {
      const auto fit = fit_controls(successes, method);
      if (jade) {
        memories[0].scale = std::lerp(memories[0].scale, fit.scale, options_.jade_learning);
        memories[0].crossover = std::lerp(memories[0].crossover, fit.crossover, options_.jade_learning);
      } else {
        memories[next_memory] = fit;
        next_memory = (next_memory + 1) % memories.size();
      }
    }
    if (linear) {
      const double fraction = static_cast<double>(observe.calls) / options_.evaluations;
      const auto desired = static_cast<std::size_t>(std::round(std::lerp(
          static_cast<double>(options_.population), static_cast<double>(options_.minimum_population), fraction)));
      if (desired < next.size()) {
        std::stable_sort(next.begin(), next.end(), [](const Member& a, const Member& b) { return a.value < b.value; });
        next.resize(desired);
      }
    }
    population = std::move(next);
    for (auto& member : displaced) archive.push_back(std::move(member));
    while (archive.size() > population.size()) {
      const auto drop = std::uniform_int_distribution<std::size_t>(0, archive.size() - 1)(rng);
      std::swap(archive[drop], archive.back());
      archive.pop_back();
    }
  }

  coordinates = std::move(observe.best);
  return observe.best_value;
}

} // namespace datamunge::optim
