/// @file
/// @brief Reference-direction association, epsilon indicators and hypervolume survival.

#include "pareto_selection_detail.hpp"

#include <datamunge/optim/nsga2.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::optim::detail {
/// @brief Numerical primitives for environmental selection.
namespace {

/// @brief Positive margin above unit-normalized objectives for hypervolume scoring.
constexpr double reference = 1.1;

/// @brief Enumerate equally spaced nonnegative lattice directions of a simplex.
/// @param direction Mutable current integer barycentric coordinate represented as doubles.
/// @param axis Next axis to assign.
/// @param remaining Barycentric count remaining for this and following axes.
/// @param result Output accumulator, capped at 4096 points.
/// @throws std::length_error The direction count exceeds 4096.
void lattice(std::vector<double>& direction, std::size_t axis, std::size_t remaining,
             std::vector<std::vector<double>>& result) {
  if (axis + 1 == direction.size()) {
    if (result.size() == 4096) throw std::length_error("NSGA-III lattice exceeds 4096 directions");
    direction[axis] = static_cast<double>(remaining);
    result.push_back(direction);
    return;
  }
  for (std::size_t count = 0; count <= remaining; ++count) {
    direction[axis] = static_cast<double>(count);
    lattice(direction, axis + 1, remaining - count, result);
  }
}

/// @brief Contribution of a competitor to IBEA fitness.
/// @param competitor Competing normalized objective vector.
/// @param subject Objective vector being evaluated.
/// @param scale Positive indicator scaling factor.
/// @return Exponential of negative additive epsilon indicator divided by scale.
double epsilon_contribution(const std::vector<double>& competitor,
                            const std::vector<double>& subject, double scale) {
  double epsilon = -std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < subject.size(); ++j) epsilon = std::max(epsilon, competitor[j] - subject[j]);
  return std::exp(std::min(100.0, -epsilon / scale));
}

/// @brief Sum pairwise IBEA contribution penalties.
/// @param points Population with unit-normalized objectives.
/// @param scale Positive indicator scale.
/// @return Negative aggregate penalty for each point.
std::vector<double> epsilon_fitness(const std::vector<ParetoPoint>& points, double scale) {
  std::vector<double> result(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (std::size_t j = 0; j < points.size(); ++j) {
      if (i != j) result[i] -= epsilon_contribution(points[j].objectives, points[i].objectives, scale);
    }
  }
  return result;
}

/// @brief Estimate Hypervolume-based fitness for removing a specified number of members.
/// @param points Population with unit-normalized objectives.
/// @param indices Active subset for assignment.
/// @param sample_count Positive number of Monte Carlo draws.
/// @param removed Number of removals targeted by the HypE fitness.
/// @param rng Borrowed local engine.
/// @return Relative hypervolume fitness for each supplied index.
std::vector<double> hype_fitness(const std::vector<ParetoPoint>& points,
    const std::vector<std::size_t>& indices, std::size_t sample_count, std::size_t removed,
    std::mt19937_64& rng) {
  const auto n = indices.size();
  const auto dimension = points.front().objectives.size();
  std::vector<double> minimum = points[indices.front()].objectives;
  for (const auto index : indices) {
    for (std::size_t j = 0; j < dimension; ++j) minimum[j] = std::min(minimum[j], points[index].objectives[j]);
  }
  std::vector<double> weights(removed + 1), result(n), sample(dimension);
  double coefficient = 1;
  for (std::size_t coverage = 1; coverage <= removed; ++coverage) {
    weights[coverage] = coefficient / static_cast<double>(coverage);
    if (coverage < removed) coefficient *= static_cast<double>(removed - coverage) / (n - coverage);
  }
  std::uniform_real_distribution<double> uniform;
  std::vector<std::size_t> covered;
  covered.reserve(n);
  for (std::size_t draw = 0; draw < sample_count; ++draw) {
    for (std::size_t j = 0; j < dimension; ++j) sample[j] = std::lerp(minimum[j], reference, uniform(rng));
    covered.clear();
    for (std::size_t i = 0; i < n; ++i) {
      const auto& point = points[indices[i]].objectives;
      if (std::equal(point.begin(), point.end(), sample.begin(), std::less_equal{})) covered.push_back(i);
      if (covered.size() > removed) break;
    }
    if (covered.size() > removed) continue;
    const auto weight = weights[covered.size()];
    for (const auto i : covered) result[i] += weight;
  }
  return result;
}

/// @brief Find least exact two-dimensional exclusive hypervolume contribution in a front.
/// @param points Scored pool with normalized objectives.
/// @param indices Worst nondominated front.
/// @return Position in indices to remove.
std::size_t least_hypervolume(const std::vector<ParetoPoint>& points,
                              const std::vector<std::size_t>& indices) {
  auto sorted = indices;
  std::stable_sort(sorted.begin(), sorted.end(), [&](auto a, auto b) {
    return points[a].objectives[0] < points[b].objectives[0];
  });
  std::size_t chosen = sorted.front();
  double smallest = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    const auto& x = points[sorted[i]].objectives;
    const auto next_x = i + 1 == sorted.size() ? reference : points[sorted[i + 1]].objectives[0];
    const auto last_y = i == 0 ? reference : points[sorted[i - 1]].objectives[1];
    const double contribution = std::max(0.0, next_x - x[0]) * std::max(0.0, last_y - x[1]);
    if (contribution < smallest) {
      smallest = contribution;
      chosen = sorted[i];
    }
  }
  return std::find(indices.begin(), indices.end(), chosen) - indices.begin();
}

/// @brief Determine hyperplane intercept factors from objective-wise extreme points.
/// @param points Scored pool with normalized objectives translated to the ideal point.
/// @param eligible Eligible indices from complete fronts and the splitting front.
/// @param dimensions Number of objective components.
/// @return Rescaling factors or zeros when the hyperplane is ill-conditioned.
std::vector<double> intercept_factors(const std::vector<ParetoPoint>& points,
                                      const std::vector<std::size_t>& eligible, std::size_t dimensions) {
  std::vector<std::vector<double>> equation;
  for (std::size_t axis = 0; axis < dimensions; ++axis) {
    double smallest = std::numeric_limits<double>::infinity();
    std::size_t extreme = eligible.front();
    for (const auto i : eligible) {
      double scalar = 0;
      for (std::size_t j = 0; j < dimensions; ++j) {
        scalar = std::max(scalar, points[i].objectives[j] / (j == axis ? 1 : 1e-6));
      }
      if (scalar < smallest) {
        smallest = scalar;
        extreme = i;
      }
    }
    equation.push_back(points[extreme].objectives);
    equation.back().push_back(1);
  }

  for (std::size_t j = 0; j < dimensions; ++j) {
    auto pivot = j;
    for (std::size_t i = j + 1; i < dimensions; ++i) {
      if (std::abs(equation[i][j]) > std::abs(equation[pivot][j])) pivot = i;
    }
    if (std::abs(equation[pivot][j]) < 1e-12) return std::vector<double>(dimensions);
    std::swap(equation[j], equation[pivot]);
    const double divisor = equation[j][j];
    for (std::size_t col = j; col <= dimensions; ++col) equation[j][col] /= divisor;
    for (std::size_t i = 0; i < dimensions; ++i) {
      if (i == j) continue;
      const auto factor = equation[i][j];
      for (std::size_t col = j; col <= dimensions; ++col) equation[i][col] -= factor * equation[j][col];
    }
  }
  std::vector<double> factors(dimensions);
  for (std::size_t j = 0; j < dimensions; ++j) {
    factors[j] = equation[j].back();
    if (!(std::isfinite(factors[j]) && factors[j] > 1e-12 && factors[j] < 1e12)) {
      return std::vector<double>(dimensions);
    }
  }
  return factors;
}

/// @brief Fill a splitting front by NSGA-III niche occupancy selection.
/// @param points Cohort with normalized objectives.
/// @param partial Last nondominated front reached by survivor selection.
/// @param directions Explicit unit reference directions.
/// @param keep Total survivor count.
/// @param retained Already selected indices, appended to until complete.
/// @param rng Borrowed local engine for association and survivor ties.
void fill_niches(std::vector<ParetoPoint> points, const std::vector<std::size_t>& partial,
                 const std::vector<std::vector<double>>& directions, std::size_t keep,
                 std::vector<std::size_t>& retained, std::mt19937_64& rng) {
  const auto dimensions = directions.front().size();
  auto eligible = retained;
  eligible.insert(eligible.end(), partial.begin(), partial.end());
  auto ideal = points[eligible.front()].objectives;
  for (const auto i : eligible) {
    for (std::size_t axis = 0; axis < dimensions; ++axis) ideal[axis] = std::min(ideal[axis], points[i].objectives[axis]);
  }
  std::vector<double> ranges(dimensions);
  for (const auto i : eligible) {
    for (std::size_t axis = 0; axis < dimensions; ++axis) {
      points[i].objectives[axis] -= ideal[axis];
      ranges[axis] = std::max(ranges[axis], points[i].objectives[axis]);
    }
  }
  const auto factors = intercept_factors(points, eligible, dimensions);
  for (const auto i : eligible) {
    for (std::size_t axis = 0; axis < dimensions; ++axis) {
      const double value = points[i].objectives[axis];
      points[i].objectives[axis] = factors[axis] > 0 ? value * factors[axis]
          : ranges[axis] > 0 ? value / ranges[axis] : 0;
    }
  }

  std::vector<std::size_t> counts(directions.size());
  std::vector<std::vector<std::pair<std::size_t, double>>> choices(directions.size());
  for (std::size_t i = 0; i < eligible.size(); ++i) {
    const auto index = eligible[i];
    const auto& point = points[index].objectives;
    const auto norm = std::inner_product(point.begin(), point.end(), point.begin(), 0.0);
    double distance = std::numeric_limits<double>::infinity();
    std::size_t niche = 0;
    for (std::size_t j = 0; j < directions.size(); ++j) {
      const auto projection = std::inner_product(point.begin(), point.end(), directions[j].begin(), 0.0);
      const auto candidate = std::max(0.0, norm - projection * projection);
      if (candidate < distance) {
        distance = candidate;
        niche = j;
      }
    }
    if (i < retained.size()) ++counts[niche];
    else choices[niche].emplace_back(index, distance);
  }

  std::vector<std::size_t> sparse;
  while (retained.size() < keep) {
    sparse.clear();
    auto density = std::numeric_limits<std::size_t>::max();
    for (std::size_t j = 0; j < directions.size(); ++j) {
      if (choices[j].empty() || counts[j] > density) continue;
      if (counts[j] < density) {
        sparse.clear();
        density = counts[j];
      }
      sparse.push_back(j);
    }
    const auto niche = sparse[std::uniform_int_distribution<std::size_t>(0, sparse.size() - 1)(rng)];
    auto& candidates = choices[niche];
    const auto selected = counts[niche] == 0
        ? std::min_element(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
              return a.second < b.second;
            }) - candidates.begin()
        : static_cast<std::ptrdiff_t>(std::uniform_int_distribution<std::size_t>(0, candidates.size() - 1)(rng));
    retained.push_back(candidates[selected].first);
    candidates.erase(candidates.begin() + selected);
    ++counts[niche];
  }
}

} // namespace

std::vector<std::vector<double>> reference_directions(std::size_t dimensions,
                                                      const ParetoSearchOptions& options) {
  auto directions = options.reference_directions;
  if (directions.empty()) {
    if (options.reference_divisions == 0) throw std::invalid_argument("NSGA-III lattice divisions must be positive");
    if (dimensions > 4096 || options.reference_divisions >= 4096) {
      throw std::length_error("NSGA-III lattice exceeds 4096 directions");
    }
    std::vector<double> direction(dimensions);
    lattice(direction, 0, options.reference_divisions, directions);
  }
  for (auto& vector : directions) {
    if (vector.size() != dimensions || !std::ranges::all_of(vector, [](double x) {
          return x >= 0 && std::isfinite(x);
        })) {
      throw std::invalid_argument("NSGA-III directions must be matching nonnegative finite vectors");
    }
    const double maximum = *std::ranges::max_element(vector);
    if (maximum == 0) throw std::invalid_argument("NSGA-III reference directions must be nonzero");
    for (auto& x : vector) x /= maximum;
    const double norm = std::sqrt(std::inner_product(vector.begin(), vector.end(), vector.begin(), 0.0));
    for (auto& x : vector) x /= norm;
  }
  return directions;
}

std::vector<double> pareto_mating_fitness(const std::vector<ParetoPoint>& points,
                                         const ParetoSearchOptions& options, std::mt19937_64& rng) {
  if (options.method == ParetoSearchMethod::ibea) return epsilon_fitness(points, options.indicator_scale);
  std::vector<std::size_t> indices(points.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  return hype_fitness(points, indices, options.hypervolume_samples, points.size(), rng);
}

std::vector<std::size_t> pareto_survivors(const std::vector<ParetoPoint>& points,
    std::size_t keep, const ParetoSearchOptions& options,
    const std::vector<std::vector<double>>& directions, std::mt19937_64& rng) {
  std::vector<std::size_t> retained;
  if (options.method == ParetoSearchMethod::ibea) {
    retained.resize(points.size());
    std::iota(retained.begin(), retained.end(), std::size_t{0});
    auto fitness = epsilon_fitness(points, options.indicator_scale);
    while (retained.size() > keep) {
      const auto worst = std::min_element(retained.begin(), retained.end(),
                                         [&](auto a, auto b) { return fitness[a] < fitness[b]; });
      const auto removed = *worst;
      retained.erase(worst);
      for (const auto index : retained) {
        fitness[index] += epsilon_contribution(points[removed].objectives, points[index].objectives,
                                              options.indicator_scale);
      }
    }
    return retained;
  }

  auto ranking = nsga2_rank(points);
  retained.reserve(keep);
  for (auto& front : ranking.fronts) {
    const auto count = keep - retained.size();
    if (front.size() <= count) {
      retained.insert(retained.end(), front.begin(), front.end());
      if (retained.size() == keep) break;
      continue;
    }
    if (options.method == ParetoSearchMethod::nsga3) {
      fill_niches(points, front, directions, keep, retained, rng);
    } else if (options.method == ParetoSearchMethod::sms_emoa) {
      front.erase(front.begin() + least_hypervolume(points, front));
      retained.insert(retained.end(), front.begin(), front.end());
    } else if (options.method == ParetoSearchMethod::hype) {
      while (front.size() > count) {
        const auto fitness = hype_fitness(points, front, options.hypervolume_samples, front.size() - count, rng);
        front.erase(front.begin() + (std::ranges::min_element(fitness) - fitness.begin()));
      }
      retained.insert(retained.end(), front.begin(), front.end());
    } else {
      // GDE3 shares the existing NSGA-II crowding order.
      std::stable_sort(front.begin(), front.end(),
                       [&](auto a, auto b) { return ranking.crowding[a] > ranking.crowding[b]; });
      retained.insert(retained.end(), front.begin(), front.begin() + count);
    }
    break;
  }
  return retained;
}

} // namespace datamunge::optim::detail
