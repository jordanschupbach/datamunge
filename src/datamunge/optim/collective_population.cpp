/// @file
/// @brief Memetic leaping, imperial competition and immune/fractal population updates.

#include "collective_search_detail.hpp"
#include "differential_evolution_detail.hpp"

#include <numeric>

namespace datamunge::optim::detail {
/// @brief Shared variation and population details for multi-phase search.
namespace {

/// @brief Move an inferior individual toward a selected guide.
/// @param current Starting coordinates.
/// @param guide Target coordinates.
/// @param fraction Fraction of the displacement to apply.
/// @return Displaced trial.
std::vector<double> attract(const std::vector<double>& current, const std::vector<double>& guide, double fraction) {
  auto x = current;
  for (std::size_t j = 0; j < x.size(); ++j) x[j] = std::lerp(current[j], guide[j], fraction);
  return x;
}

/// @brief Test if two immune cells are within a separation threshold.
/// @param first First cell coordinate.
/// @param second Second cell coordinate.
/// @param limit Squared normalized threshold.
/// @return Whether the squared separation is less than the threshold.
bool near(const std::vector<double>& first, const std::vector<double>& second, double limit) {
  double distance = 0;
  for (std::size_t j = 0; j < first.size(); ++j) {
    const auto gap = first[j] - second[j];
    distance += gap * gap;
    if (distance >= limit) return false;
  }
  return true;
}

/// @brief An imperialist and its currently owned colonies.
struct Empire {
  Member leader; ///< Best currently installed member.
  std::vector<Member> colonies; ///< Remaining evaluated members.
};

/// @brief Compute imperialist loss plus a small mean colony cost.
/// @param empire Evaluated empire.
/// @return Total cost used for competition.
long double empire_cost(const Empire& empire) {
  long double average = 0;
  for (const auto& colony : empire.colonies) average += static_cast<long double>(colony.value) / empire.colonies.size();
  return static_cast<long double>(empire.leader.value) + 0.1L * average;
}

} // namespace

void frogs(Work& work, std::vector<Member>& population) {
  const auto size = population.size();
  const auto groups = std::max(std::size_t{2}, static_cast<std::size_t>(std::sqrt(static_cast<double>(size))));
  while (work.available()) {
    std::stable_sort(population.begin(), population.end(), better);
    const auto global = work.leader;
    for (std::size_t group = 0; group < groups && work.available(); ++group) {
      for (std::size_t iteration = 0; iteration < 3 && work.available(); ++iteration) {
        auto best = group, worst = group;
        for (std::size_t i = group + groups; i < size; i += groups) {
          if (better(population[i], population[best])) best = i;
          if (better(population[worst], population[i])) worst = i;
        }
        auto& frog = population[worst];
        auto leap = work.score(attract(frog.point, population[best].point, work.uniform(work.rng)));
        if (!better(leap, frog) && work.available()) {
          leap = work.score(attract(frog.point, global, work.uniform(work.rng)));
        }
        if (better(leap, frog)) frog = std::move(leap);
        else if (work.available()) frog = work.score(work.random_point());
      }
    }
  }
}

void empires(Work& work, std::vector<Member>& population) {
  std::stable_sort(population.begin(), population.end(), better);
  const auto count = std::max(std::size_t{2}, population.size() / 4);
  std::vector<Empire> groups;
  groups.reserve(count);
  std::vector<double> powers(count);
  for (std::size_t i = 0; i < count; ++i) {
    groups.push_back({std::move(population[i]), {}});
    powers[i] = static_cast<double>(count - i);
  }
  std::discrete_distribution<std::size_t> assign(powers.begin(), powers.end());
  for (std::size_t i = count; i < population.size(); ++i) {
    groups[assign(work.rng)].colonies.push_back(std::move(population[i]));
  }

  while (work.available()) {
    for (auto& empire : groups) {
      for (auto& colony : empire.colonies) {
        if (!work.available()) break;
        auto trial = attract(colony.point, empire.leader.point, 1.5 * work.uniform(work.rng));
        const auto deviation = 0.1 * work.options.step_scale * (1 - work.progress());
        for (auto& coordinate : trial) coordinate += deviation * work.normal(work.rng);
        if (work.uniform(work.rng) < 0.05) trial = work.random_point();
        colony = work.score(std::move(trial));
        if (better(colony, empire.leader)) std::swap(colony, empire.leader);
      }
    }
    if (groups.size() == 1 || !work.available()) continue;

    std::stable_sort(groups.begin(), groups.end(), [](const Empire& a, const Empire& b) {
      return empire_cost(a) < empire_cost(b);
    });
    auto& weak = groups.back();
    powers.resize(groups.size() - 1);
    const auto worst = empire_cost(weak);
    const auto best = empire_cost(groups.front());
    for (std::size_t i = 0; i < powers.size(); ++i) {
      powers[i] = worst > best ? static_cast<double>((worst - empire_cost(groups[i])) / (worst - best)) : 1;
    }
    auto& strong = groups[std::discrete_distribution<std::size_t>(powers.begin(), powers.end())(work.rng)];
    if (!weak.colonies.empty()) {
      const auto transferred = std::ranges::max_element(weak.colonies, better);
      strong.colonies.push_back(std::move(*transferred));
      weak.colonies.erase(transferred);
    }
    if (weak.colonies.empty()) {
      strong.colonies.push_back(std::move(weak.leader));
      groups.pop_back();
    }
  }
}

void clones(Work& work, std::vector<Member>& population, bool extended) {
  const auto size = population.size();
  while (work.available()) {
    std::stable_sort(population.begin(), population.end(), better);
    const auto count = extended ? size : std::max(std::size_t{1}, size / 3);
    auto matured = population;
    const auto dimension = work.lower.size();
    for (std::size_t i = 0; i < count && work.available(); ++i) {
      const auto copies = extended ? 5 : std::max(std::size_t{1}, size / (i + 1));
      const double rank = static_cast<double>(i) / std::max(std::size_t{1}, count - 1);
      const auto scale = work.options.step_scale * std::exp(-3 * (1 - rank));
      for (std::size_t copy = 0; copy < copies && work.available(); ++copy) {
        auto trial = population[i].point;
        for (auto& x : trial) x += scale * work.normal(work.rng);
        auto candidate = work.score(std::move(trial));
        if (extended) {
          if (better(candidate, matured[i])) matured[i] = std::move(candidate);
        } else {
          matured.push_back(std::move(candidate));
        }
      }
    }
    std::stable_sort(matured.begin(), matured.end(), better);
    if (extended) {
      std::vector<Member> distinct;
      const auto threshold = work.options.step_scale * 0.25;
      const auto threshold_squared = threshold * threshold * dimension;
      for (auto& candidate : matured) {
        if (!std::ranges::any_of(distinct, [&](const Member& retained) {
              return near(candidate.point, retained.point, threshold_squared);
            })) {
          distinct.push_back(std::move(candidate));
        }
      }
      matured = std::move(distinct);
    }
    const auto direct = size - std::max(std::size_t{1}, size / 5);
    if (matured.size() > direct) matured.resize(direct);
    while (matured.size() < size && work.available()) matured.push_back(work.score(work.random_point()));
    population = std::move(matured);
  }
}

void fractals(Work& work, std::vector<Member>& population) {
  const auto size = population.size();
  std::size_t generation = 0;
  while (work.available()) {
    ++generation;
    const auto leader = work.leader;
    const auto g = static_cast<double>(generation) + 1;
    for (auto& member : population) {
      if (!work.available()) break;
      auto retained = member;
      for (std::size_t diffusion = 0; diffusion < 2 && work.available(); ++diffusion) {
        auto trial = member.point;
        for (std::size_t j = 0; j < trial.size(); ++j) {
          const auto spread = work.options.step_scale * std::abs(std::log(g) / g * (trial[j] - leader[j]));
          const auto center = diffusion == 0 ? leader[j] : trial[j];
          trial[j] = center + spread * work.normal(work.rng);
          if (diffusion == 0) {
            trial[j] += work.uniform(work.rng) * leader[j] - work.uniform(work.rng) * member.point[j];
          }
        }
        auto tested = work.score(std::move(trial));
        if (better(tested, retained)) retained = std::move(tested);
      }
      member = std::move(retained);
    }
    for (std::size_t phase = 0; phase < 2 && work.available(); ++phase) {
      std::stable_sort(population.begin(), population.end(), better);
      const auto snapshot = population;
      for (std::size_t i = 0; i < size && work.available(); ++i) {
        const auto threshold = static_cast<double>(i + 1) / size;
        if (phase != 0 && work.uniform(work.rng) >= threshold) continue;
        const auto peer = random_distinct_indices(work.rng, size, i, 3);
        auto trial = snapshot[i].point;
        const auto local = work.uniform(work.rng);
        for (std::size_t j = 0; j < trial.size(); ++j) {
          const auto difference = snapshot[peer[0]].point[j] - snapshot[peer[1]].point[j];
          if (phase == 0) {
            if (work.uniform(work.rng) < threshold) trial[j] = snapshot[peer[2]].point[j] + local * difference;
          } else {
            trial[j] += local * (difference + (local < 0.5
                ? work.leader[j] - snapshot[i].point[j]
                : snapshot[peer[2]].point[j] - snapshot[i].point[j]));
          }
        }
        auto tested = work.score(std::move(trial));
        if (better(tested, population[i])) population[i] = std::move(tested);
      }
    }
  }
}

} // namespace datamunge::optim::detail
