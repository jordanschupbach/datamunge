/// @file
/// @brief Chemotactic, visual, gravitational, krill and electrostatic population dynamics.

#include "collective_search_detail.hpp"

#include <numeric>

namespace datamunge::optim::detail {
/// @brief Geometry and fitness transforms for normalized population dynamics.
namespace {

/// @brief Euclidean distance between unit-box positions.
/// @param a First point.
/// @param b Second point with the same dimension.
/// @return Euclidean separation.
double separation(const std::vector<double>& a, const std::vector<double>& b) {
  double squared = 0;
  for (std::size_t j = 0; j < a.size(); ++j) {
    const auto delta = a[j] - b[j];
    squared += delta * delta;
  }
  return std::sqrt(squared);
}

/// @brief Compute direction-normalized motion toward a target.
/// @param from Base position.
/// @param to Desired direction endpoint.
/// @param step Signed motion distance.
/// @return Step from the base in the selected direction, or the base if coincident.
std::vector<double> toward(const std::vector<double>& from, const std::vector<double>& to, double step) {
  auto x = from;
  const auto length = separation(from, to);
  if (length > 0) {
    for (std::size_t j = 0; j < x.size(); ++j) x[j] += (to[j] - x[j]) * (step / length);
  }
  return x;
}

/// @brief Draw an isotropic direction for bounded local motion.
/// @param work Local engine and current problem dimension.
/// @return Unit-length direction vector, or zero if a degenerate draw occurs.
std::vector<double> direction(Work& work) {
  std::vector<double> out(work.lower.size());
  for (auto& v : out) v = work.normal(work.rng);
  const auto norm = std::sqrt(std::inner_product(out.begin(), out.end(), out.begin(), 0.0));
  if (norm > 0) for (auto& v : out) v /= norm;
  return out;
}

/// @brief Additive attraction and short-distance repulsion around a foraging colony.
/// @param x Coordinate being tested.
/// @param population Current evaluated population positions.
/// @param index Position of the bacterium being updated, excluded as self.
/// @return Normalized crowding adjustment.
double swarm_cost(const std::vector<double>& x, const std::vector<Member>& population, std::size_t index) {
  double adjustment = 0;
  for (std::size_t i = 0; i < population.size(); ++i) {
    if (i == index) continue;
    const auto radius = separation(x, population[i].point);
    const auto squared = radius * radius;
    adjustment += (0.15 * std::exp(-30 * squared) - 0.1 * std::exp(-5 * squared)) / population.size();
  }
  return adjustment;
}

/// @brief Convert finite losses to bounded nonnegative fitness values.
/// @param population Evaluated members.
/// @return Fitness one for the best and zero for the worst, or all ones on ties.
std::vector<double> fitness(const std::vector<Member>& population) {
  const auto [best, worst] = std::ranges::minmax_element(population, better);
  const long double span = static_cast<long double>(worst->value) - best->value;
  std::vector<double> scaled(population.size(), 1);
  if (span > 0) {
    for (std::size_t i = 0; i < population.size(); ++i) {
      scaled[i] = static_cast<double>((static_cast<long double>(worst->value) - population[i].value) / span);
    }
  }
  return scaled;
}

} // namespace

void chemotaxis(Work& work, std::vector<Member>& population, bool extended) {
  const auto size = population.size();
  const auto n = work.lower.size();
  std::vector<long double> health(size);
  std::size_t phase = 0;
  while (work.available()) {
    const auto step = work.options.step_scale * (1 - 0.85 * work.progress());
    for (std::size_t i = 0; i < size && work.available(); ++i) {
      const auto move = direction(work);
      auto augmented = [&](const Member& member) {
        return static_cast<long double>(member.value) +
               (extended ? swarm_cost(member.point, population, i) : 0);
      };
      auto value = augmented(population[i]);
      for (std::size_t swim = 0; swim < 4 && work.available(); ++swim) {
        auto trial = population[i].point;
        for (std::size_t j = 0; j < n; ++j) trial[j] += step * move[j];
        auto tested = work.score(std::move(trial));
        const auto adjusted = augmented(tested);
        if (adjusted >= value) break;
        value = adjusted;
        population[i] = std::move(tested);
      }
      health[i] += value / 4;
    }
    if (!extended || ++phase % 4 != 0 || !work.available()) continue;
    std::vector<std::size_t> ranking(size);
    std::iota(ranking.begin(), ranking.end(), std::size_t{0});
    std::stable_sort(ranking.begin(), ranking.end(), [&](auto a, auto b) { return health[a] < health[b]; });
    const auto originals = population;
    for (std::size_t i = 0; i < size; ++i) population[i] = originals[ranking[i / 2]];
    std::ranges::fill(health, 0);
    if (phase % 8 == 0) {
      for (auto& bacterium : population) {
        if (!work.available()) break;
        if (work.uniform(work.rng) < 0.25) bacterium = work.score(work.random_point());
      }
    }
  }
}

void fish(Work& work, std::vector<Member>& population) {
  const auto size = population.size(), n = work.lower.size();
  const auto visual = std::sqrt(static_cast<double>(n)) / 3;
  while (work.available()) {
    const auto step = work.options.step_scale * (1 - 0.8 * work.progress());
    for (std::size_t i = 0; i < size && work.available(); ++i) {
      auto& current = population[i];
      std::size_t count = 0, best = i;
      std::vector<double> center(n);
      for (std::size_t j = 0; j < size; ++j) {
        if (i == j || separation(current.point, population[j].point) > visual) continue;
        ++count;
        if (better(population[j], population[best])) best = j;
        for (std::size_t k = 0; k < n; ++k) center[k] += population[j].point[k];
      }
      if (count != 0 && count <= size / 2) {
        for (auto& coordinate : center) coordinate /= count;
        auto school = work.score(std::move(center));
        if (!work.available()) break;
        const auto& guide = better(school, population[best]) ? school : population[best];
        if (better(guide, current)) {
          current = work.score(toward(current.point, guide.point, work.uniform(work.rng) * step));
          continue;
        }
      }
      bool moved = false;
      for (std::size_t attempts = 0; attempts < 4 && work.available(); ++attempts) {
        const auto ray = direction(work);
        auto probe = current.point;
        const auto radius = visual * work.uniform(work.rng);
        for (std::size_t j = 0; j < n; ++j) probe[j] += radius * ray[j];
        auto target = work.score(std::move(probe));
        if (better(target, current)) {
          if (work.available()) {
            current = work.score(toward(current.point, target.point, std::min(step, radius) * work.uniform(work.rng)));
          }
          moved = true;
          break;
        }
      }
      if (!moved && work.available()) {
        auto probe = current.point;
        const auto ray = direction(work);
        for (std::size_t j = 0; j < n; ++j) probe[j] += step * ray[j];
        current = work.score(std::move(probe));
      }
    }
  }
}

void gravity(Work& work, std::vector<Member>& population) {
  const auto size = population.size(), n = work.lower.size();
  std::vector velocities(size, std::vector<double>(n));
  std::vector<std::size_t> order(size);
  std::iota(order.begin(), order.end(), std::size_t{0});
  while (work.available()) {
    const auto fit = fitness(population);
    const auto norm = std::accumulate(fit.begin(), fit.end(), 0.0);
    const auto progress = work.progress();
    const auto strength = work.options.step_scale * std::exp(-10 * progress);
    const auto count = 1 + static_cast<std::size_t>((size - 1) * (1 - progress));
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return fit[a] > fit[b]; });
    auto next = population;
    for (std::size_t i = 0; i < size && work.available(); ++i) {
      std::vector<double> force(n);
      for (std::size_t rank = 0; rank < count; ++rank) {
        const auto neighbor = order[rank];
        if (neighbor == i) continue;
        const auto r = separation(population[neighbor].point, population[i].point);
        const auto pull = strength * work.uniform(work.rng) * fit[neighbor] / (norm * (r + 1e-12));
        for (std::size_t j = 0; j < n; ++j) force[j] += pull * (population[neighbor].point[j] - population[i].point[j]);
      }
      auto trial = population[i].point;
      for (std::size_t j = 0; j < n; ++j) {
        velocities[i][j] = work.uniform(work.rng) * velocities[i][j] + force[j];
        trial[j] += velocities[i][j];
      }
      next[i] = work.score(std::move(trial));
    }
    population.swap(next);
  }
}

void krill(Work& work, std::vector<Member>& population) {
  const auto size = population.size(), n = work.lower.size();
  auto personal = population;
  std::vector induced(size, std::vector<double>(n)), forage = induced;
  while (work.available()) {
    const auto fit = fitness(population);
    const auto fit_sum = std::accumulate(fit.begin(), fit.end(), 0.0);
    auto food = std::vector<double>(n);
    for (std::size_t i = 0; i < size; ++i) {
      const auto w = fit[i] / fit_sum;
      for (std::size_t j = 0; j < n; ++j) food[j] += w * population[i].point[j];
    }
    const auto tested_food = work.score(std::move(food));
    const auto best = work.leader;
    const auto progress = work.progress();
    const auto scale = work.options.step_scale;
    auto next = population;
    for (std::size_t i = 0; i < size && work.available(); ++i) {
      const auto& current = population[i];
      double sensing = 0;
      for (const auto& other : population) sensing += separation(other.point, current.point) / (5 * size);
      auto neighbor_motion = std::vector<double>(n);
      for (std::size_t k = 0; k < size; ++k) {
        const auto r = separation(population[k].point, current.point);
        if (k == i || r == 0 || r > sensing) continue;
        for (std::size_t j = 0; j < n; ++j) {
          neighbor_motion[j] += (fit[k] - fit[i]) * (population[k].point[j] - current.point[j]) / (r * size);
        }
      }
      const auto global = toward(current.point, best, 1);
      const auto foodward = toward(current.point, tested_food.point, 1);
      const auto memory = toward(current.point, personal[i].point, 1);
      auto trial = current.point;
      for (std::size_t j = 0; j < n; ++j) {
        induced[i][j] = 0.5 * induced[i][j] + 0.5 * (neighbor_motion[j] + (1 - fit[i]) * (global[j] - current.point[j]));
        forage[i][j] = 0.5 * forage[i][j] + 0.4 * (1 - progress) * (foodward[j] - current.point[j]) +
                       0.2 * (memory[j] - current.point[j]);
        const auto diffusion = 0.2 * (1 - progress) * (2 * work.uniform(work.rng) - 1);
        trial[j] += scale * (induced[i][j] + forage[i][j] + diffusion);
      }
      next[i] = work.score(std::move(trial));
      if (better(next[i], personal[i])) personal[i] = next[i];
    }
    population.swap(next);
  }
}

void charged(Work& work, std::vector<Member>& population) {
  const auto size = population.size(), n = work.lower.size();
  std::vector velocities(size, std::vector<double>(n));
  std::vector memory = population;
  while (work.available()) {
    const auto charges = fitness(population);
    const auto best = work.leader;
    const auto progress = work.progress();
    auto next = population;
    for (std::size_t i = 0; i < size && work.available(); ++i) {
      std::vector<double> force(n);
      for (std::size_t k = 0; k < size; ++k) {
        if (i == k) continue;
        const auto relative = charges[k] >= charges[i] ? 1.0 : charges[k] / std::max(1e-12, charges[i]);
        if (work.uniform(work.rng) >= relative) continue;
        std::vector<double> center(n);
        for (std::size_t j = 0; j < n; ++j) center[j] = std::midpoint(population[i].point[j], population[k].point[j]);
        const auto distance = separation(population[i].point, population[k].point);
        const auto ratio = distance / (separation(center, best) + 1e-12);
        constexpr double radius = 0.4;
        const auto field = ratio < radius ? ratio / (radius * radius * radius) : 1 / (ratio * ratio);
        for (std::size_t j = 0; j < n; ++j) {
          force[j] += charges[k] * field * (population[k].point[j] - population[i].point[j]) / size;
        }
      }
      auto trial = population[i].point;
      for (std::size_t j = 0; j < n; ++j) {
        trial[j] += work.uniform(work.rng) * (1 - progress / 2) * velocities[i][j] +
                    work.uniform(work.rng) * (0.5 + progress / 2) * work.options.step_scale * force[j];
        if (trial[j] < 0 || trial[j] > 1) {
          trial[j] = work.uniform(work.rng) < 0.8 ? memory[i].point[j] : work.uniform(work.rng);
        }
        velocities[i][j] = trial[j] - population[i].point[j];
      }
      next[i] = work.score(std::move(trial));
      if (better(next[i], memory[i])) memory[i] = next[i];
    }
    population.swap(next);
  }
}

} // namespace datamunge::optim::detail
