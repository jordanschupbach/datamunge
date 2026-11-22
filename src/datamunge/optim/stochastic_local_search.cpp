/// @file
/// @brief Adaptive local search and hybrid search using nested local minimization.

#include <datamunge/optim/stochastic_local_search.hpp>

#include "boxed_search_detail.hpp"
#include <datamunge/optim/random_search.hpp>

#include <random>
#include <utility>

namespace datamunge::optim {

StochasticLocalSearch::StochasticLocalSearch(StochasticLocalOptions options) : options_(options) {}

double StochasticLocalSearch::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                       const std::vector<double>& lower,
                                       const std::vector<double>& upper) const {
  const auto dimension = coordinates.size();
  detail::validate_box(dimension, lower, upper, options_.evaluations);
  const auto method = options_.method;
  if (!(options_.initial_scale > 0 && options_.initial_scale <= 1) ||
      (method != StochasticLocalMethod::hill_climbing && method != StochasticLocalMethod::luus_jaakola &&
       method != StochasticLocalMethod::one_plus_one) ||
      (method != StochasticLocalMethod::hill_climbing &&
       !(options_.update_interval > 0 && options_.scale_factor > 0 && options_.scale_factor < 1))) {
    throw std::invalid_argument("Invalid stochastic local search controls");
  }

  std::vector<double> best = coordinates;
  auto unit = detail::normalize_start(best, lower, upper);
  double best_value = detail::score(function, best);
  std::mt19937_64 rng(options_.seed);
  std::normal_distribution<double> normal;
  std::uniform_real_distribution<double> uniform;
  std::vector<double> point(dimension), trial(dimension);
  double scale = options_.initial_scale;
  std::size_t successes = 0;
  for (std::size_t call = 1; call < options_.evaluations; ++call) {
    for (std::size_t axis = 0; axis < dimension; ++axis) {
      if (method == StochasticLocalMethod::luus_jaakola) {
        const double lo = std::max(0.0, unit[axis] - scale);
        const double hi = std::min(1.0, unit[axis] + scale);
        trial[axis] = std::lerp(lo, hi, uniform(rng));
      } else {
        trial[axis] = std::clamp(unit[axis] + scale * normal(rng), 0.0, 1.0);
      }
      point[axis] = std::lerp(lower[axis], upper[axis], trial[axis]);
    }
    const auto value = detail::score(function, point);
    if (value < best_value) {
      best_value = value;
      best = point;
      unit.swap(trial);
      ++successes;
    }
    if (method != StochasticLocalMethod::hill_climbing && call % options_.update_interval == 0) {
      if (method == StochasticLocalMethod::luus_jaakola ||
          static_cast<long double>(successes) * 5 < options_.update_interval) {
        scale *= options_.scale_factor;
      } else if (static_cast<long double>(successes) * 5 > options_.update_interval) {
        scale = std::min(1.0, scale / options_.scale_factor);
      }
      successes = 0;
    }
  }
  coordinates = std::move(best);
  return best_value;
}

/// @brief Normalized proposals and single-linkage clustering.
namespace {

/// @brief Draw independent normalized coordinates and map them to the box.
/// @param lower Lower bounds.
/// @param upper Upper bounds.
/// @param rng Borrowed random engine.
/// @return Feasible uniformly sampled point.
std::vector<double> uniform_point(const std::vector<double>& lower, const std::vector<double>& upper,
                                  std::mt19937_64& rng) {
  std::vector<double> point(lower.size());
  std::uniform_real_distribution<double> uniform;
  for (std::size_t j = 0; j < point.size(); ++j) point[j] = std::lerp(lower[j], upper[j], uniform(rng));
  return point;
}

/// @brief Compute whether normalized positions fall within a critical radius.
/// @param a First unit-box point.
/// @param b Second unit-box point.
/// @param radius_squared Squared linkage radius.
/// @return Whether Euclidean distance is at most the radius.
bool linked(const std::vector<double>& a, const std::vector<double>& b, double radius_squared) {
  double squared_distance = 0;
  for (std::size_t axis = 0; axis < a.size(); ++axis) {
    const auto delta = a[axis] - b[axis];
    squared_distance += delta * delta;
    if (squared_distance > radius_squared) return false;
  }
  return true;
}

/// @brief One global sample and its clustering information.
struct LinkageSample {
  std::vector<double> point; ///< Point in original coordinates.
  std::vector<double> unit; ///< Point in normalized coordinates.
  double value; ///< Scored loss.
  bool launched{}; ///< Whether the sample started a local solve.
};

/// @brief Run the multilevel uniform-sampling and single-linkage filter.
/// @tparam Local Callable minimizing from a given point using remaining budget.
/// @param initial Clamped initial point.
/// @param lower Valid box lower bounds.
/// @param upper Valid box upper bounds.
/// @param options Valid MLSL controls.
/// @param observe Counting objective and incumbent recorder.
/// @param local Local search callable borrowing observe.
/// @param rng Borrowed random engine.
template <class Local>
void run_mlsl(std::vector<double> initial, const std::vector<double>& lower,
              const std::vector<double>& upper, const HybridSearchOptions& options,
              detail::CountingObjective& observe, Local& local, std::mt19937_64& rng) {
  std::size_t dimension = 0;
  for (std::size_t j = 0; j < lower.size(); ++j) dimension += lower[j] != upper[j];
  dimension = std::max(std::size_t{1}, dimension);
  auto unit = detail::normalize_start(initial, lower, upper);
  const auto first_value = observe.evaluate(initial);
  std::vector<LinkageSample> samples{{std::move(initial), std::move(unit), first_value}};
  std::vector<std::vector<double>> minima;
  while (observe.calls < options.evaluations) {
    const auto count = std::min(options.sample_batch, options.evaluations - observe.calls);
    for (std::size_t i = 0; i < count; ++i) {
      auto point = uniform_point(lower, upper, rng);
      auto normalized = detail::normalize_start(point, lower, upper);
      const auto value = observe.evaluate(point);
      samples.push_back({std::move(point), std::move(normalized), value});
    }
    const double n = static_cast<double>(samples.size());
    const double radius = options.linkage_scale * std::pow(std::log(n) / n, 1.0 / dimension);
    const auto eligible = std::min(samples.size(),
        static_cast<std::size_t>(std::ceil(options.sample_fraction * n)));
    std::stable_sort(samples.begin(), samples.end(), [](const LinkageSample& a, const LinkageSample& b) {
      return a.value < b.value;
    });
    for (std::size_t i = 0; i < eligible && observe.calls < options.evaluations; ++i) {
      auto& sample = samples[i];
      if (sample.launched) continue;
      const auto nearby = [&](const std::vector<double>& other) {
        return linked(sample.unit, other, radius * radius);
      };
      if (std::ranges::any_of(minima, nearby) ||
          std::any_of(samples.begin(), samples.begin() + i, [&](const LinkageSample& better) {
            return nearby(better.unit);
          })) continue;
      auto point = sample.point;
      local(point);
      sample.launched = true;
      minima.push_back(detail::normalize_start(point, lower, upper));
    }
  }
}

} // namespace

HybridRandomSearch::HybridRandomSearch(HybridSearchOptions options) : options_(options) {}

double HybridRandomSearch::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                     const std::vector<double>& lower,
                                     const std::vector<double>& upper) const {
  detail::validate_box(coordinates.size(), lower, upper, options_.evaluations);
  const auto method = options_.method;
  if (options_.local_evaluations == 0 || !(options_.local_step > 0 && options_.local_step <= 1) ||
      (method != HybridSearchMethod::multistart && method != HybridSearchMethod::basin_hopping &&
       method != HybridSearchMethod::mlsl) ||
      (method == HybridSearchMethod::basin_hopping &&
       !(std::isfinite(options_.temperature) && options_.temperature > 0 &&
         options_.hop_scale > 0 && options_.hop_scale <= 1)) ||
      (method == HybridSearchMethod::mlsl &&
       !(options_.sample_batch > 0 && options_.sample_fraction > 0 && options_.sample_fraction <= 1 &&
         std::isfinite(options_.linkage_scale) && options_.linkage_scale > 0))) {
    throw std::invalid_argument("Invalid hybrid random search controls");
  }

  auto current = coordinates;
  auto center = detail::normalize_start(current, lower, upper);
  detail::CountingObjective observe(function);
  std::mt19937_64 rng(options_.seed);
  /// Spend the lesser of the local cap and the global remaining budget.
  auto local = [&](std::vector<double>& point) {
    const auto remaining = options_.evaluations - observe.calls;
    return SolisWets({std::min(remaining, options_.local_evaluations), rng(), options_.local_step})
        .optimize(observe, point, lower, upper);
  };

  if (method == HybridSearchMethod::mlsl) {
    run_mlsl(std::move(current), lower, upper, options_, observe, local, rng);
  } else {
    auto current_value = local(current);
    center = detail::normalize_start(current, lower, upper);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> uniform;
    while (observe.calls < options_.evaluations) {
      if (method == HybridSearchMethod::multistart) {
        auto start = uniform_point(lower, upper, rng);
        local(start);
      } else {
        std::vector<double> candidate(current.size());
        for (std::size_t j = 0; j < candidate.size(); ++j) {
          const double unit = std::clamp(center[j] + options_.hop_scale * normal(rng), 0.0, 1.0);
          candidate[j] = std::lerp(lower[j], upper[j], unit);
        }
        const auto value = local(candidate);
        if (value <= current_value || uniform(rng) < std::exp((current_value - value) / options_.temperature)) {
          center = detail::normalize_start(candidate, lower, upper);
          current_value = value;
        }
      }
    }
  }
  coordinates = std::move(observe.best);
  return observe.best_value;
}

} // namespace datamunge::optim
