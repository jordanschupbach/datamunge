/// @file
/// @brief Normalized annealing proposals, q-acceptance and coordinate reannealing.

#include <datamunge/optim/annealing_search.hpp>

#include "boxed_search_detail.hpp"

#include <random>
#include <utility>

namespace datamunge::optim {
/// @brief Numerical helpers for the annealing schedules and transitions.
namespace {

/// @brief Validate controls needed by the selected annealing variant.
/// @param options Configured annealing controls.
/// @throws std::invalid_argument Invalid active controls or unknown variant.
void validate_options(const AnnealingSearchOptions& options) {
  if (!(std::isfinite(options.temperature) && options.temperature > 0 &&
        options.proposal_scale > 0 && options.proposal_scale <= 1)) {
    throw std::invalid_argument("Invalid annealing temperature or proposal scale");
  }
  const bool geometric = options.cooling > 0 && options.cooling <= 1;
  bool valid = false;
  switch (options.method) {
    case AnnealingMethod::reheating:
      valid = geometric && options.patience > 0;
      break;
    case AnnealingMethod::tunneling:
      valid = geometric && std::isfinite(options.tunneling_strength) && options.tunneling_strength > 0;
      break;
    case AnnealingMethod::tsallis:
      valid = options.visiting_q > 1 && options.visiting_q < 3 &&
              std::isfinite(options.acceptance_q) && options.acceptance_q < 1;
      break;
    case AnnealingMethod::vfsa:
      valid = std::isfinite(options.vfsa_cooling) && options.vfsa_cooling > 0 &&
              options.probe_distance > 0 && options.probe_distance <= 1;
      break;
  }
  if (!valid) throw std::invalid_argument("Invalid annealing method or controls");
}

/// @brief Transform one objective value into the tunneling landscape.
/// @param value Raw finite loss.
/// @param record Best raw loss already observed and no greater than value.
/// @param strength Positive exponential coefficient.
/// @return Monotonic transformed loss in [0, 1].
long double tunnel(double value, double record, double strength) {
  return -std::expm1(-strength * (static_cast<long double>(value) - record));
}

/// @brief Convert a normalized point from a cooled landscape to a probability.
/// @param delta Nonnegative finite or saturated positive increase.
/// @param temperature Positive temperature or zero when underflowed.
/// @param q Generalized acceptance exponent; one selects Boltzmann acceptance.
/// @return Acceptance probability in [0,1].
long double probability(long double delta, long double temperature, long double q) {
  if (delta <= 0) return 1;
  if (temperature <= 0) return 0;
  if (q == 1) return std::exp(-delta / temperature);
  const auto scaled = (1 - q) * delta / temperature;
  if (scaled >= 1) return 0;
  return std::exp(std::log1p(-scaled) / (1 - q));
}

/// @brief Compute Ingber's normalized proposal from a uniform draw.
/// @param uniform Unit uniform sample in [0,1).
/// @param temperature Positive coordinate temperature or zero when underflowed.
/// @return Signed displacement in [-1,1].
double vfsa_step(double uniform, double temperature) {
  if (temperature == 0) return 0;
  const auto sign = 2 * uniform - 1;
  const long double t = temperature;
  const auto magnitude = t * std::expm1(std::abs(sign) * (std::log1p(t) - std::log(t)));
  return std::copysign(static_cast<double>(std::min(1.0L, magnitude)), sign);
}

/// @brief Compute independent finite differences around the record and adjust proposal clocks.
/// @tparam Evaluate Callable converting and scoring a normalized point.
/// @param best Record coordinates in the normalized box, copied for the complete probe sweep.
/// @param reference Loss at the sweep's original center.
/// @param free_axes Axis indices not fixed by bounds.
/// @param clocks Effective annealing time for each coordinate; updated after probes.
/// @param options Annealing schedule and probe controls.
/// @param calls Evaluation count incremented by evaluate.
/// @param evaluate Objective wrapper counting actual calls and updating the global record.
template <class Evaluate>
void reanneal(std::vector<double> best, double reference, const std::vector<std::size_t>& free_axes,
              std::vector<double>& clocks, const AnnealingSearchOptions& options,
              const std::size_t& calls, Evaluate& evaluate) {
  const auto exponent = 1.0 / free_axes.size();
  std::vector<long double> sensitivities(clocks.size());
  long double maximum = 0;
  for (const auto axis : free_axes) {
    if (calls == options.evaluations) return;
    auto probe = best;
    const double step = best[axis] <= 0.5 ? options.probe_distance : -options.probe_distance;
    probe[axis] = std::clamp(probe[axis] + step, 0.0, 1.0);
    if (probe[axis] == best[axis]) continue;
    const long double value = evaluate(probe);
    sensitivities[axis] = std::abs((value - reference) / (probe[axis] - best[axis]));
    maximum = std::max(maximum, sensitivities[axis]);
  }
  if (!(maximum > 0 && std::isfinite(maximum))) return;
  for (const auto axis : free_axes) {
    const auto log_ratio = sensitivities[axis] == 0 ? std::numeric_limits<long double>::infinity()
        : std::log(maximum) - std::log(sensitivities[axis]);
    const auto cooled = std::pow(clocks[axis], exponent);
    const auto effective = std::max(0.0L, cooled - log_ratio / options.vfsa_cooling);
    clocks[axis] = static_cast<double>(std::pow(effective, 1 / exponent));
  }
}

} // namespace

AnnealingSearch::AnnealingSearch(AnnealingSearchOptions options) : options_(options) {}

double AnnealingSearch::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                 const std::vector<double>& lower,
                                 const std::vector<double>& upper) const {
  const auto dimension = coordinates.size();
  detail::validate_box(dimension, lower, upper, options_.evaluations);
  validate_options(options_);
  auto point = coordinates;
  auto current = detail::normalize_start(point, lower, upper);
  auto best_unit = current;
  detail::CountingObjective observe(function);
  double current_value = observe.evaluate(point);
  /// Map normalized position, score it and retain any new record.
  auto evaluate = [&](const std::vector<double>& unit) {
    for (std::size_t j = 0; j < dimension; ++j) point[j] = std::lerp(lower[j], upper[j], unit[j]);
    const auto record = observe.best_value;
    const auto value = observe.evaluate(point);
    if (value < record) best_unit = unit;
    return value;
  };

  const auto method = options_.method;
  const bool vfsa = method == AnnealingMethod::vfsa;
  const bool tsallis = method == AnnealingMethod::tsallis;
  const bool tunneling = method == AnnealingMethod::tunneling;
  std::mt19937_64 rng(options_.seed);
  std::uniform_real_distribution<double> uniform;
  std::normal_distribution<double> normal;
  std::vector<std::size_t> free_axes;
  for (std::size_t j = 0; j < dimension; ++j) {
    if (lower[j] != upper[j]) free_axes.push_back(j);
  }
  const double inverse_dimension = 1.0 / std::max(std::size_t{1}, free_axes.size());
  const double df = tsallis ? (3 - options_.visiting_q) / (options_.visiting_q - 1) : 1;
  std::chi_squared_distribution<double> chi_square(df);
  std::vector<double> clocks(dimension), trial(dimension);
  double temperature = options_.temperature;
  std::size_t stale = 0, reheats = 0, proposals = 0;
  bool restart = false;
  while (observe.calls < options_.evaluations) {
    trial = current;
    if (restart) {
      for (const auto j : free_axes) trial[j] = uniform(rng);
      current_value = evaluate(trial);
      current.swap(trial);
      restart = false;
      continue;
    }

    double scale = options_.proposal_scale;
    if (tsallis) {
      const double q_minus_one = options_.visiting_q - 1;
      const double relative = std::expm1(q_minus_one * std::log(2.0)) /
          std::expm1(q_minus_one * std::log(2.0 + proposals));
      temperature = options_.temperature * relative;
      const auto draw = chi_square(rng);
      const auto spread = std::pow(relative, 1 / (3 - options_.visiting_q));
      scale *= spread > 0 ? spread / std::sqrt(std::max(std::numeric_limits<double>::min(), draw) * q_minus_one) : 0;
    } else if (vfsa) {
      temperature = options_.temperature *
          std::exp(-options_.vfsa_cooling * std::pow(static_cast<double>(proposals), inverse_dimension));
    }
    for (const auto j : free_axes) {
      double displacement;
      if (vfsa) {
        const double axis_temperature = options_.proposal_scale *
            std::exp(-options_.vfsa_cooling * std::pow(clocks[j], inverse_dimension));
        displacement = vfsa_step(uniform(rng), axis_temperature);
        clocks[j] += 1;
      } else {
        displacement = scale * normal(rng);
      }
      trial[j] = std::clamp(current[j] + displacement, 0.0, 1.0);
    }
    const auto record = observe.best_value;
    const auto value = evaluate(trial);
    ++proposals;
    if (value <= current_value ||
        uniform(rng) < probability(tunneling ? tunnel(value, record, options_.tunneling_strength) -
                                              tunnel(current_value, record, options_.tunneling_strength)
                                            : static_cast<long double>(value) - current_value,
                                  temperature, tsallis ? options_.acceptance_q : 1.0)) {
      current.swap(trial);
      current_value = value;
    }

    if (!tsallis && !vfsa) temperature *= options_.cooling;
    if (method == AnnealingMethod::reheating) {
      stale = value < record ? 0 : stale + 1;
      if (stale >= options_.patience) {
        temperature = options_.temperature;
        stale = 0;
        ++reheats;
        restart = options_.restart_every != 0 && reheats % options_.restart_every == 0;
      }
    } else if (vfsa && options_.reanneal_every != 0 && proposals % options_.reanneal_every == 0 &&
               observe.calls < options_.evaluations && !free_axes.empty()) {
      reanneal(best_unit, observe.best_value, free_axes, clocks, options_, observe.calls, evaluate);
    }
  }
  coordinates = std::move(observe.best);
  return observe.best_value;
}

} // namespace datamunge::optim
