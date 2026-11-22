/// @file
/// @brief Implementation of budgeted boxed stochastic search methods.

#include <datamunge/optim/random_search.hpp>
#include "boxed_search_detail.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::optim {
/// @brief Validation and unit-box sampling primitives.
namespace {

/// @brief Find ascending prime radices for a Halton design.
/// @param dimension Number of required bases.
/// @return The first dimension primes.
std::vector<std::size_t> halton_bases(std::size_t dimension) {
  std::vector<std::size_t> bases;
  for (std::size_t candidate = 2; bases.size() < dimension; ++candidate) {
    bool prime = true;
    for (const auto divisor : bases) {
      if (divisor > candidate / divisor) break;
      if (candidate % divisor == 0) {
        prime = false;
        break;
      }
    }
    if (prime) bases.push_back(candidate);
  }
  return bases;
}

/// @brief Compute a radical inverse by reversing the digits across the radix point.
/// @param index Zero-based Halton index.
/// @param base Prime radix for this coordinate.
/// @return Coordinate before applying the random shift.
double radical_inverse(std::size_t index, std::size_t base) {
  const double inverse = 1.0 / static_cast<double>(base);
  double place = inverse;
  double value = 0;
  while (index != 0) {
    value += place * static_cast<double>(index % base);
    place *= inverse;
    index /= base;
  }
  return value;
}

/// @brief Number of fixed-point bits in the Sobol generator.
constexpr unsigned sobol_bits = 53;
/// @brief Fixed-point direction numbers, in decreasing order of significance.
using Directions = std::array<std::uint64_t, sobol_bits>;

/// @brief Primitive-polynomial parameters of one Sobol dimension.
struct SobolParameters {
  unsigned degree; ///< Degree of the primitive polynomial.
  unsigned coefficients; ///< Packed interior coefficients.
  std::array<unsigned, 6> initial; ///< Initial odd numerators.
};

/// @brief First fifteen rows of the Joe–Kuo D(6) table, covering axes 2 through 16.
constexpr std::array<SobolParameters, 15> sobol_parameters{{
    {1, 0, {1}}, {2, 1, {1, 3}}, {3, 1, {1, 3, 1}}, {3, 2, {1, 1, 1}},
    {4, 1, {1, 1, 3, 3}}, {4, 4, {1, 3, 5, 13}}, {5, 2, {1, 1, 5, 5, 17}},
    {5, 4, {1, 1, 5, 5, 5}}, {5, 7, {1, 1, 7, 11, 19}}, {5, 11, {1, 1, 5, 1, 1}},
    {5, 13, {1, 1, 1, 3, 11}}, {5, 14, {1, 3, 5, 5, 31}}, {6, 1, {1, 3, 3, 9, 7, 49}},
    {6, 13, {1, 1, 1, 15, 21, 21}}, {6, 16, {1, 3, 1, 13, 27, 49}},
}};

/// @brief Generate the direction numbers used by the Gray-code iteration.
/// @param axis Zero-based coordinate in [0, 15].
/// @return Fixed-point direction numbers for the axis.
Directions sobol_directions(std::size_t axis) {
  Directions directions{};
  if (axis == 0) {
    for (unsigned bit = 0; bit < sobol_bits; ++bit) {
      directions[bit] = std::uint64_t{1} << (sobol_bits - bit - 1);
    }
    return directions;
  }
  const auto& params = sobol_parameters[axis - 1];
  for (unsigned bit = 0; bit < params.degree; ++bit) {
    directions[bit] = std::uint64_t{params.initial[bit]} << (sobol_bits - bit - 1);
  }
  for (unsigned bit = params.degree; bit < sobol_bits; ++bit) {
    const auto first = directions[bit - params.degree];
    directions[bit] = first ^ (first >> params.degree);
    for (unsigned lag = 1; lag < params.degree; ++lag) {
      if ((params.coefficients >> (params.degree - lag - 1)) & 1U) {
        directions[bit] ^= directions[bit - lag];
      }
    }
  }
  return directions;
}

} // namespace

RandomSearch::RandomSearch(RandomSearchOptions options) : options_(options) {}

double RandomSearch::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                              const std::vector<double>& lower,
                              const std::vector<double>& upper) const {
  const auto dimension = coordinates.size();
  detail::validate_box(dimension, lower, upper, options_.evaluations);

  std::mt19937_64 rng(options_.seed);
  std::uniform_real_distribution<double> uniform;
  std::vector<std::vector<std::size_t>> strata;
  std::vector<std::size_t> bases;
  std::vector<double> shifts;
  std::vector<Directions> directions;
  std::vector<std::uint64_t> sobol;

  switch (options_.design) {
    case RandomDesign::uniform:
      break;
    case RandomDesign::latin_hypercube:
      strata.resize(dimension);
      for (auto& axis : strata) {
        axis.resize(options_.evaluations);
        std::iota(axis.begin(), axis.end(), std::size_t{0});
        std::shuffle(axis.begin(), axis.end(), rng);
      }
      break;
    case RandomDesign::halton:
      bases = halton_bases(dimension);
      shifts.resize(dimension);
      for (auto& shift : shifts) shift = uniform(rng);
      break;
    case RandomDesign::sobol:
      if (dimension > 1 + sobol_parameters.size() ||
          options_.evaluations > (std::uint64_t{1} << sobol_bits)) {
        throw std::invalid_argument("Sobol search supports up to 16 dimensions and 2^53 points");
      }
      for (std::size_t axis = 0; axis < dimension; ++axis) {
        directions.push_back(sobol_directions(axis));
        sobol.push_back(rng() >> (64 - sobol_bits));
      }
      break;
    default:
      throw std::invalid_argument("unknown random design");
  }

  std::vector<double> point(dimension), best(dimension);
  double best_value = std::numeric_limits<double>::infinity();
  for (std::size_t call = 0; call < options_.evaluations; ++call) {
    for (std::size_t axis = 0; axis < dimension; ++axis) {
      double unit = 0;
      switch (options_.design) {
        case RandomDesign::uniform:
          unit = uniform(rng);
          break;
        case RandomDesign::latin_hypercube:
          unit = (static_cast<double>(strata[axis][call]) + uniform(rng)) /
                 static_cast<double>(options_.evaluations);
          break;
        case RandomDesign::halton:
          unit = radical_inverse(call, bases[axis]) + shifts[axis];
          unit -= std::floor(unit);
          break;
        case RandomDesign::sobol:
          if (call != 0) sobol[axis] ^= directions[axis][std::countr_zero(call)];
          unit = std::ldexp(static_cast<double>(sobol[axis]), -static_cast<int>(sobol_bits));
          break;
      }
      point[axis] = std::lerp(lower[axis], upper[axis], unit);
    }
    const double value = detail::score(function, point);
    if (value < best_value) {
      best_value = value;
      best = point;
    }
  }
  coordinates = std::move(best);
  return best_value;
}

SolisWets::SolisWets(SolisWetsOptions options) : options_(options) {}

double SolisWets::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                           const std::vector<double>& lower,
                           const std::vector<double>& upper) const {
  const auto dimension = coordinates.size();
  detail::validate_box(dimension, lower, upper, options_.evaluations);
  if (!(options_.initial_step > 0 && options_.initial_step <= 1)) {
    throw std::invalid_argument("SolisWets initial_step must be in (0, 1]");
  }

  std::vector<double> best = coordinates;
  auto unit = detail::normalize_start(best, lower, upper);

  double best_value = detail::score(function, best);
  std::mt19937_64 rng(options_.seed);
  std::normal_distribution<double> normal;
  double step = options_.initial_step;
  std::size_t calls = 1, successes = 0, failures = 0;
  std::vector<double> bias(dimension), delta(dimension), trial(dimension), point(dimension);
  while (calls < options_.evaluations) {
    for (std::size_t axis = 0; axis < dimension; ++axis) {
      delta[axis] = upper[axis] == lower[axis] ? 0 : step * normal(rng) + bias[axis];
    }
    bool improved = false;
    for (const double sign : {1.0, -1.0}) {
      if (calls == options_.evaluations) break;
      for (std::size_t axis = 0; axis < dimension; ++axis) {
        trial[axis] = std::clamp(unit[axis] + sign * delta[axis], 0.0, 1.0);
        point[axis] = std::lerp(lower[axis], upper[axis], trial[axis]);
      }
      const double value = detail::score(function, point);
      ++calls;
      if (value < best_value) {
        best_value = value;
        best = point;
        for (std::size_t axis = 0; axis < dimension; ++axis) {
          bias[axis] = 0.2 * bias[axis] + 0.4 * (trial[axis] - unit[axis]);
        }
        unit.swap(trial);
        improved = true;
        break;
      }
    }
    if (improved) {
      failures = 0;
      if (++successes == 5) {
        step = std::min(1.0, step * 2);
        successes = 0;
      }
    } else {
      successes = 0;
      for (auto& component : bias) component *= 0.5;
      if (++failures == 3) {
        step *= 0.5;
        failures = 0;
      }
    }
  }
  coordinates = std::move(best);
  return best_value;
}

} // namespace datamunge::optim
