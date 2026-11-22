/// @file
/// @brief Distinct proposal and survival rules under a common objective-call cap.

#include <datamunge/optim/collective_search.hpp>

#include "collective_search_detail.hpp"
#include "differential_evolution_detail.hpp"

#include <numbers>
#include <numeric>
#include <utility>

namespace datamunge::optim {
/// @brief Shared population representation and method-specific updates.
namespace {

using detail::Member;
using detail::Work;
using detail::better;

/// @brief Run teacher and learner phases, evaluating each proposal once.
/// @param work Shared scoring and budget state.
/// @param population Evaluated cohort, updated by improving proposals.
void teaching(Work& work, std::vector<Member>& population) {
  const auto size = population.size(), dimension = work.lower.size();
  while (work.available()) {
    const auto instructor = std::ranges::min_element(population, better)->point;
    std::vector<double> mean(dimension);
    for (const auto& student : population) {
      for (std::size_t j = 0; j < dimension; ++j) mean[j] += student.point[j] / size;
    }
    for (auto& student : population) {
      if (!work.available()) break;
      auto trial = student.point;
      const double factor = work.uniform(work.rng) < 0.5 ? 1 : 2;
      const auto r = work.uniform(work.rng);
      for (std::size_t j = 0; j < dimension; ++j) trial[j] += r * (instructor[j] - factor * mean[j]);
      auto tested = work.score(std::move(trial));
      if (better(tested, student)) student = std::move(tested);
    }
    for (std::size_t i = 0; i < size && work.available(); ++i) {
      const auto peer = detail::random_distinct_indices(work.rng, size, i, 1)[0];
      auto x = population[i].point;
      const auto direction = better(population[i], population[peer]) ? 1 : -1;
      const auto r = work.uniform(work.rng);
      for (std::size_t j = 0; j < dimension; ++j) {
        x[j] += direction * r * (population[i].point[j] - population[peer].point[j]);
      }
      auto tested = work.score(std::move(x));
      if (better(tested, population[i])) population[i] = std::move(tested);
    }
  }
}

/// @brief Move leaders around the best food location and followers toward their predecessors.
/// @param work Shared scoring and budget state.
/// @param population Ordered salp chain.
void salps(Work& work, std::vector<Member>& population) {
  const auto leaders = population.size() / 2;
  while (work.available()) {
    const auto food = work.leader;
    const auto cooling = 2 * std::exp(-std::pow(4 * work.progress(), 2));
    for (std::size_t i = 0; i < population.size() && work.available(); ++i) {
      auto x = population[i].point;
      for (std::size_t j = 0; j < x.size(); ++j) {
        if (i < leaders) {
          x[j] = food[j] + cooling * (work.uniform(work.rng) < 0.5 ? -1 : 1) * work.uniform(work.rng);
        } else {
          x[j] = std::midpoint(x[j], population[i - 1].point[j]);
        }
      }
      population[i] = work.score(std::move(x));
    }
  }
}

/// @brief Alternate Lévy flights toward the incumbent with local differential pollination.
/// @param work Shared scoring and budget state.
/// @param population Evaluated flowering positions.
void pollination(Work& work, std::vector<Member>& population) {
  constexpr double beta = 1.5;
  const auto levy_scale = std::pow(std::tgamma(1 + beta) * std::sin(std::numbers::pi * beta / 2) /
      (std::tgamma((1 + beta) / 2) * beta * std::pow(2.0, (beta - 1) / 2)), 1 / beta);
  while (work.available()) {
    for (std::size_t i = 0; i < population.size() && work.available(); ++i) {
      auto x = population[i].point;
      if (work.uniform(work.rng) < work.options.global_probability) {
        const auto denominator = std::max(1e-12, std::abs(work.normal(work.rng)));
        const auto flight = work.options.step_scale * levy_scale * std::abs(work.normal(work.rng)) /
                            std::pow(denominator, 1 / beta);
        for (std::size_t j = 0; j < x.size(); ++j) x[j] += flight * (work.leader[j] - x[j]);
      } else {
        const auto peers = detail::random_distinct_indices(work.rng, population.size(), i, 2);
        const auto r = work.uniform(work.rng);
        for (std::size_t j = 0; j < x.size(); ++j) {
          x[j] += r * (population[peers[0]].point[j] - population[peers[1]].point[j]);
        }
      }
      auto tested = work.score(std::move(x));
      if (better(tested, population[i])) population[i] = std::move(tested);
    }
  }
}

/// @brief Reproduce seeds and retain the colony's lowest-loss members.
/// @param work Shared scoring and budget state.
/// @param population Current weed colony.
void weeds(Work& work, std::vector<Member>& population) {
  while (work.available()) {
    std::stable_sort(population.begin(), population.end(), better);
    const long double minimum = population.front().value;
    const long double maximum = population.back().value;
    const auto remaining = 1 - work.progress();
    const auto scale = work.options.step_scale * std::lerp(0.025, 1.0, remaining * remaining * remaining);
    std::vector<Member> seeds;
    for (const auto& parent : population) {
      if (!work.available()) break;
      const auto fitness = maximum > minimum ? (maximum - parent.value) / (maximum - minimum) : 0.5L;
      const auto count = 1 + static_cast<std::size_t>(std::floor(4 * fitness));
      for (std::size_t i = 0; i < count && work.available(); ++i) {
        auto x = parent.point;
        for (auto& value : x) value += scale * work.normal(work.rng);
        seeds.push_back(work.score(std::move(x)));
      }
    }
    for (auto& seed : seeds) population.push_back(std::move(seed));
    std::stable_sort(population.begin(), population.end(), better);
    if (population.size() > work.options.population) population.resize(work.options.population);
  }
}

/// @brief Perform frequency motion and loudness-gated bat acceptance.
/// @param work Shared scoring and budget state.
/// @param population Current bat locations.
void bats(Work& work, std::vector<Member>& population) {
  const auto size = population.size();
  std::vector velocities(size, std::vector<double>(work.lower.size()));
  std::vector<double> loudness(size, 0.9);
  std::vector<double> pulse(size, 0.1);
  std::size_t generation = 0;
  while (work.available()) {
    ++generation;
    for (std::size_t i = 0; i < size && work.available(); ++i) {
      const auto frequency = 2 * work.uniform(work.rng);
      const auto local = work.uniform(work.rng) > pulse[i];
      const auto local_step = work.options.step_scale * (2 * work.uniform(work.rng) - 1) *
                             std::accumulate(loudness.begin(), loudness.end(), 0.0) / size;
      auto x = population[i].point;
      for (std::size_t j = 0; j < x.size(); ++j) {
        velocities[i][j] = std::clamp(velocities[i][j] + frequency * (x[j] - work.leader[j]), -1.0, 1.0);
        x[j] = local ? work.leader[j] + local_step : x[j] + velocities[i][j];
      }
      auto tested = work.score(std::move(x));
      if (better(tested, population[i]) && work.uniform(work.rng) < loudness[i]) {
        population[i] = std::move(tested);
        loudness[i] *= 0.9;
        pulse[i] = 0.9 * -std::expm1(-0.2 * static_cast<double>(generation));
      }
    }
  }
}

/// @brief Migrate donor coordinates and mutate low-probability habitats while preserving an elite.
/// @param work Shared scoring and budget state.
/// @param population Current habitats.
void biogeography(Work& work, std::vector<Member>& population) {
  const auto size = population.size();
  std::vector<double> emigration(size), mutation(size);
  const double maximum = static_cast<double>(size - 1);
  const double midpoint = std::floor(maximum / 2);
  const double normalizer = std::lgamma(midpoint + 1) + std::lgamma(maximum - midpoint + 1);
  for (std::size_t i = 0; i < size; ++i) {
    emigration[i] = 1 - i / maximum;
    const double log_probability = normalizer - std::lgamma(i + 1.0) - std::lgamma(size - static_cast<double>(i));
    mutation[i] = work.options.mutation_probability * -std::expm1(std::min(0.0, log_probability));
  }
  std::discrete_distribution<std::size_t> donor(emigration.begin(), emigration.end());
  while (work.available()) {
    std::stable_sort(population.begin(), population.end(), better);
    auto next = population;
    for (std::size_t i = 1; i < size && work.available(); ++i) {
      auto x = population[i].point;
      for (std::size_t j = 0; j < x.size(); ++j) {
        if (work.uniform(work.rng) < 1 - emigration[i]) x[j] = population[donor(work.rng)].point[j];
        if (work.uniform(work.rng) < mutation[i]) x[j] = work.uniform(work.rng);
      }
      next[i] = work.score(std::move(x));
    }
    population = std::move(next);
  }
}

} // namespace

CollectiveSearch::CollectiveSearch(CollectiveSearchOptions options) : options_(options) {}

double CollectiveSearch::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                  const std::vector<double>& lower,
                                  const std::vector<double>& upper) const {
  detail::validate_box(coordinates.size(), lower, upper, options_.evaluations);
  if (options_.population < 4) throw std::invalid_argument("Collective search population must be at least four");
  switch (options_.method) {
    case CollectiveSearchMethod::teaching_learning:
    case CollectiveSearchMethod::salp_swarm:
      break;
    case CollectiveSearchMethod::flower_pollination:
      if (!(options_.global_probability >= 0 && options_.global_probability <= 1)) {
        throw std::invalid_argument("Flower global probability must be in [0, 1]");
      }
      [[fallthrough]];
    case CollectiveSearchMethod::invasive_weed:
    case CollectiveSearchMethod::bat:
    case CollectiveSearchMethod::bacterial_chemotaxis:
    case CollectiveSearchMethod::bacterial_foraging:
    case CollectiveSearchMethod::artificial_fish:
    case CollectiveSearchMethod::gravitational:
    case CollectiveSearchMethod::krill_herd:
    case CollectiveSearchMethod::shuffled_frog:
    case CollectiveSearchMethod::imperialist:
    case CollectiveSearchMethod::clonal_selection:
    case CollectiveSearchMethod::artificial_immune_network:
    case CollectiveSearchMethod::stochastic_fractal:
    case CollectiveSearchMethod::charged_system:
      if (!(options_.step_scale > 0 && options_.step_scale <= 1)) {
        throw std::invalid_argument("Initial proposal scale must be in (0, 1]");
      }
      break;
    case CollectiveSearchMethod::biogeography:
      if (!(options_.mutation_probability >= 0 && options_.mutation_probability <= 1)) {
        throw std::invalid_argument("Biogeography mutation probability must be in [0, 1]");
      }
      break;
    default:
      throw std::invalid_argument("Unknown collective search rule");
  }
  auto start = coordinates;
  auto unit = detail::normalize_start(start, lower, upper);
  Work work{detail::CountingObjective(function), lower, upper, options_, std::mt19937_64(options_.seed),
            {}, {}, std::vector<double>(coordinates.size()), unit};
  std::vector<Member> population;
  population.push_back({std::move(unit), work.observe.evaluate(start)});
  const auto count = options_.method == CollectiveSearchMethod::invasive_weed
      ? options_.population / 2 : options_.population;
  while (population.size() < count && work.available()) {
    population.push_back(work.score(work.random_point()));
  }
  if (work.available()) {
    switch (options_.method) {
      case CollectiveSearchMethod::teaching_learning: teaching(work, population); break;
      case CollectiveSearchMethod::salp_swarm: salps(work, population); break;
      case CollectiveSearchMethod::flower_pollination: pollination(work, population); break;
      case CollectiveSearchMethod::invasive_weed: weeds(work, population); break;
      case CollectiveSearchMethod::bat: bats(work, population); break;
      case CollectiveSearchMethod::biogeography: biogeography(work, population); break;
      case CollectiveSearchMethod::bacterial_chemotaxis: detail::chemotaxis(work, population, false); break;
      case CollectiveSearchMethod::bacterial_foraging: detail::chemotaxis(work, population, true); break;
      case CollectiveSearchMethod::artificial_fish: detail::fish(work, population); break;
      case CollectiveSearchMethod::gravitational: detail::gravity(work, population); break;
      case CollectiveSearchMethod::krill_herd: detail::krill(work, population); break;
      case CollectiveSearchMethod::shuffled_frog: detail::frogs(work, population); break;
      case CollectiveSearchMethod::imperialist: detail::empires(work, population); break;
      case CollectiveSearchMethod::clonal_selection: detail::clones(work, population, false); break;
      case CollectiveSearchMethod::artificial_immune_network: detail::clones(work, population, true); break;
      case CollectiveSearchMethod::stochastic_fractal: detail::fractals(work, population); break;
      case CollectiveSearchMethod::charged_system: detail::charged(work, population); break;
    }
  }
  coordinates = std::move(work.observe.best);
  return work.observe.best_value;
}

} // namespace datamunge::optim
