/// @file
/// @brief Paired-probe gradient estimation and normalized boxed descent.

#include <datamunge/optim/stochastic_approximation.hpp>

#include "boxed_search_detail.hpp"

#include <random>
#include <utility>

namespace datamunge::optim {

StochasticApproximation::StochasticApproximation(StochasticApproximationOptions options)
    : options_(options) {}

double StochasticApproximation::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                         const std::vector<double>& lower,
                                         const std::vector<double>& upper) const {
  const auto n = coordinates.size();
  detail::validate_box(n, lower, upper, options_.evaluations);
  if ((options_.method != StochasticApproximationMethod::spsa &&
       options_.method != StochasticApproximationMethod::rdsa) ||
      !(std::isfinite(options_.gain) && options_.gain > 0 &&
        options_.perturbation > 0 && options_.perturbation <= 1 &&
        std::isfinite(options_.stability) && options_.stability >= 0 &&
        std::isfinite(options_.gain_decay) && options_.gain_decay >= 0 &&
        std::isfinite(options_.perturbation_decay) && options_.perturbation_decay >= 0)) {
    throw std::invalid_argument("Invalid stochastic approximation controls");
  }
  auto start = coordinates;
  auto unit = detail::normalize_start(start, lower, upper);
  detail::CountingObjective objective(function);
  objective.evaluate(start);

  std::mt19937_64 rng(options_.seed);
  std::normal_distribution<double> normal;
  std::bernoulli_distribution sign;
  const bool spsa = options_.method == StochasticApproximationMethod::spsa;
  std::vector<double> point(n), direction(n), plus(n), minus(n);
  std::size_t free_axes = 0;
  for (std::size_t j = 0; j < n; ++j) free_axes += lower[j] != upper[j];
  for (std::size_t step = 1; options_.evaluations - objective.calls >= 2; ++step) {
    double norm = 0;
    for (std::size_t j = 0; j < n; ++j) {
      direction[j] = 0;
      if (lower[j] != upper[j]) {
        direction[j] = spsa ? (sign(rng) ? 1 : -1) : normal(rng);
        norm += direction[j] * direction[j];
      }
    }
    const double radius = options_.perturbation / std::pow(static_cast<double>(step),
                                                           options_.perturbation_decay);
    const double spread = spsa || norm == 0 ? radius : radius / std::sqrt(norm);
    for (std::size_t j = 0; j < n; ++j) {
      plus[j] = std::clamp(unit[j] + spread * direction[j], 0.0, 1.0);
      minus[j] = std::clamp(unit[j] - spread * direction[j], 0.0, 1.0);
      point[j] = std::lerp(lower[j], upper[j], plus[j]);
    }
    const long double positive = objective.evaluate(point);
    for (std::size_t j = 0; j < n; ++j) point[j] = std::lerp(lower[j], upper[j], minus[j]);
    const long double negative = objective.evaluate(point);
    const auto gain = options_.gain / std::pow(options_.stability + step, options_.gain_decay);
    if (gain == 0 || positive == negative) continue;

    long double squared_span = 0;
    for (std::size_t j = 0; j < n; ++j) {
      direction[j] = plus[j] - minus[j];
      squared_span += static_cast<long double>(direction[j]) * direction[j];
    }
    if (squared_span == 0) continue;
    const auto scale = static_cast<long double>(gain) * (positive - negative);
    for (std::size_t j = 0; j < n; ++j) {
      if (direction[j] == 0) continue;
      const auto displacement = spsa ? scale / direction[j]
          : scale * (free_axes * static_cast<long double>(direction[j]) / squared_span);
      unit[j] = static_cast<double>(std::clamp(unit[j] - displacement, 0.0L, 1.0L));
    }
  }
  if (objective.calls < options_.evaluations) {
    for (std::size_t j = 0; j < n; ++j) point[j] = std::lerp(lower[j], upper[j], unit[j]);
    objective.evaluate(point);
  }
  coordinates = std::move(objective.best);
  return objective.best_value;
}

} // namespace datamunge::optim
