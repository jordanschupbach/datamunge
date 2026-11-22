/// @file
/// @brief Particle swarm update rules under an explicit evaluation budget.

#include <datamunge/optim/particle_swarm_variants.hpp>

#include "boxed_search_detail.hpp"

#include <numeric>
#include <random>
#include <utility>

namespace datamunge::optim {
/// @brief Swarm state and proposal helpers.
namespace {

/// @brief A particle with its current state and personal record.
/// @tparam Point Vector of normalized coordinates or bit values.
template <class Point>
struct Particle {
  Point point; ///< Current position.
  Point best; ///< Personal best position.
  double value; ///< Loss at the personal best position.
  std::vector<double> velocity; ///< Signed position velocity or bit logits.
  std::vector<std::size_t> exemplars; ///< CLPSO guide particle index for each axis.
  std::size_t stalled{}; ///< Complete updates since the last improvement or exemplar refresh.
};

/// @brief Store personal and swarm incumbents from synchronous sweeps.
/// @tparam Point Position type.
/// @tparam Observe Callable evaluating a point and recording global work.
/// @param particles Mutable particles, with new positions already proposed.
/// @param count Number of particles that can be evaluated in this sweep.
/// @param observe Objective receiving each new position.
template <class Point, class Observe>
void update_personal(std::vector<Particle<Point>>& particles, std::size_t count, Observe& observe) {
  for (std::size_t i = 0; i < count; ++i) {
    auto& particle = particles[i];
    const auto value = observe(particle.point);
    if (value < particle.value) {
      particle.value = value;
      particle.best = particle.point;
      particle.stalled = 0;
    } else {
      ++particle.stalled;
    }
  }
}

/// @brief Tournament selector between two distinct peers other than the current particle.
/// @param particles Initialized continuous particles.
/// @param self Index of the candidate excluding itself.
/// @param rng Borrowed local random engine.
/// @return Lowest-loss peer of two sampled indices.
std::size_t exemplar(const std::vector<Particle<std::vector<double>>>& particles,
                     std::size_t self, std::mt19937_64& rng) {
  auto a = std::uniform_int_distribution<std::size_t>(0, particles.size() - 2)(rng);
  if (a >= self) ++a;
  auto b = std::uniform_int_distribution<std::size_t>(0, particles.size() - 3)(rng);
  if (b >= std::min(self, a)) ++b;
  if (b >= std::max(self, a)) ++b;
  return particles[a].value <= particles[b].value ? a : b;
}

/// @brief Refresh one particle's dimension-specific learning guides.
/// @param particles Initialized particles.
/// @param self Index to refresh.
/// @param rng Borrowed local random engine.
void refresh(std::vector<Particle<std::vector<double>>>& particles, std::size_t self,
              std::mt19937_64& rng) {
  auto& own = particles[self];
  const double fraction = static_cast<double>(self) / (particles.size() - 1);
  const double probability = 0.05 + 0.45 * std::expm1(10 * fraction) / std::expm1(10.0);
  std::bernoulli_distribution learn(probability);
  bool external = false;
  for (auto& guide : own.exemplars) {
    guide = self;
    if (learn(rng)) {
      guide = exemplar(particles, self, rng);
      external = true;
    }
  }
  if (!external) {
    const auto j = std::uniform_int_distribution<std::size_t>(0, own.exemplars.size() - 1)(rng);
    own.exemplars[j] = exemplar(particles, self, rng);
  }
  own.stalled = 0;
}

} // namespace

ParticleSwarmSearch::ParticleSwarmSearch(ParticleSwarmSearchOptions options) : options_(options) {}

double ParticleSwarmSearch::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                    const std::vector<double>& lower,
                                    const std::vector<double>& upper) const {
  const auto n = coordinates.size();
  detail::validate_box(n, lower, upper, options_.evaluations);
  const auto rule = options_.rule;
  if (options_.population < 3 ||
      (rule != ParticleSwarmRule::bare_bones && rule != ParticleSwarmRule::fully_informed &&
       rule != ParticleSwarmRule::comprehensive_learning) ||
      (rule != ParticleSwarmRule::bare_bones &&
       !(options_.inertia >= 0 && options_.inertia <= 1 &&
         std::isfinite(options_.attraction) && options_.attraction > 0 &&
         options_.velocity_limit > 0 && options_.velocity_limit <= 1)) ||
      (rule == ParticleSwarmRule::fully_informed && options_.ring_radius == 0) ||
      (rule == ParticleSwarmRule::comprehensive_learning && options_.refresh_after == 0)) {
    throw std::invalid_argument("Invalid continuous swarm controls");
  }

  std::vector<double> start = coordinates;
  auto global = detail::normalize_start(start, lower, upper);
  double global_value = detail::score(function, start);
  std::size_t calls = 1;
  std::vector<double> best = start, mapped(n);
  std::mt19937_64 rng(options_.seed);
  std::uniform_real_distribution<double> uniform;
  std::normal_distribution<double> normal;
  const auto size = std::min(options_.population, options_.evaluations);
  std::vector<Particle<std::vector<double>>> swarm;
  swarm.reserve(size);
  swarm.push_back({global, global, global_value, std::vector<double>(n), std::vector<std::size_t>(n)});
  /// Map unit-box coordinates, score once and update the global record.
  auto observe = [&](const std::vector<double>& unit) {
    for (std::size_t j = 0; j < n; ++j) mapped[j] = std::lerp(lower[j], upper[j], unit[j]);
    const auto value = detail::score(function, mapped);
    ++calls;
    if (value < global_value) {
      global_value = value;
      global = unit;
      best = mapped;
    }
    return value;
  };
  for (std::size_t i = 1; i < size; ++i) {
    std::vector<double> point(n);
    for (std::size_t j = 0; j < n; ++j) {
      if (lower[j] != upper[j]) point[j] = uniform(rng);
    }
    const auto value = observe(point);
    swarm.push_back({point, std::move(point), value, std::vector<double>(n), std::vector<std::size_t>(n)});
  }
  if (rule == ParticleSwarmRule::comprehensive_learning && calls < options_.evaluations) {
    for (std::size_t i = 0; i < size; ++i) refresh(swarm, i, rng);
  }

  const auto radius = std::min(options_.ring_radius, size / 2);
  const auto neighborhood = std::min(size, 2 * radius + 1);
  while (calls < options_.evaluations) {
    const auto count = std::min(size, options_.evaluations - calls);
    for (std::size_t i = 0; i < count; ++i) {
      auto& p = swarm[i];
      if (rule == ParticleSwarmRule::comprehensive_learning && p.stalled >= options_.refresh_after) {
        refresh(swarm, i, rng);
      }
      for (std::size_t j = 0; j < n; ++j) {
        if (lower[j] == upper[j]) continue;
        if (rule == ParticleSwarmRule::bare_bones) {
          const auto width = std::abs(p.best[j] - global[j]);
          p.point[j] = std::clamp(std::midpoint(p.best[j], global[j]) + width * normal(rng), 0.0, 1.0);
          continue;
        }
        double direction = 0;
        if (rule == ParticleSwarmRule::fully_informed) {
          auto neighbor = i >= radius ? i - radius : size - (radius - i);
          for (std::size_t k = 0; k < neighborhood; ++k) {
            direction += uniform(rng) * (swarm[neighbor].best[j] - p.point[j]) / neighborhood;
            neighbor = (neighbor + 1) % size;
          }
        } else {
          direction = uniform(rng) * (swarm[p.exemplars[j]].best[j] - p.point[j]);
        }
        p.velocity[j] = std::clamp(std::fma(options_.attraction, direction, options_.inertia * p.velocity[j]),
                                   -options_.velocity_limit, options_.velocity_limit);
        p.point[j] = std::clamp(p.point[j] + p.velocity[j], 0.0, 1.0);
      }
    }
    update_personal(swarm, count, observe);
  }
  coordinates = std::move(best);
  return global_value;
}

DiscreteDistributionResult binary_particle_swarm(const CategoricalObjective& objective,
                                                  std::size_t dimension,
                                                  const BinarySwarmOptions& options) {
  if (!objective || dimension == 0 || options.evaluations == 0 || options.population < 2 ||
      !(options.inertia >= 0 && options.inertia <= 1 && options.cognitive >= 0 &&
        std::isfinite(options.cognitive) && options.social >= 0 && std::isfinite(options.social) &&
        options.velocity_limit > 0 && std::isfinite(options.velocity_limit))) {
    throw std::invalid_argument("Invalid binary swarm domain or options");
  }
  const auto size = std::min(options.population, options.evaluations);
  std::mt19937_64 rng(options.seed);
  std::uniform_real_distribution<double> uniform;
  DiscreteDistributionResult result{{}, std::numeric_limits<double>::infinity(), 0};
  /// Score a bit vector and update the global swarm record.
  auto observe = [&](const CategoricalPoint& point) {
    const auto value = objective(point);
    if (!std::isfinite(value)) throw std::domain_error("Binary swarm loss must be finite");
    ++result.evaluations;
    if (value < result.value) {
      result.value = value;
      result.point = point;
    }
    return value;
  };
  std::vector<Particle<CategoricalPoint>> swarm;
  swarm.reserve(size);
  for (std::size_t i = 0; i < size; ++i) {
    CategoricalPoint point(dimension);
    for (auto& bit : point) bit = uniform(rng) < 0.5;
    const auto value = observe(point);
    swarm.push_back({point, std::move(point), value, std::vector<double>(dimension), {}});
  }
  while (result.evaluations < options.evaluations) {
    const auto count = std::min(size, options.evaluations - result.evaluations);
    for (std::size_t i = 0; i < count; ++i) {
      auto& p = swarm[i];
      for (std::size_t j = 0; j < dimension; ++j) {
        const auto bit = static_cast<double>(p.point[j]);
        const auto personal = options.cognitive * uniform(rng) * (static_cast<double>(p.best[j]) - bit);
        const auto social = options.social * uniform(rng) * (static_cast<double>(result.point[j]) - bit);
        const auto capped = std::clamp(static_cast<long double>(options.inertia) * p.velocity[j] +
                                      personal + social, -static_cast<long double>(options.velocity_limit),
                                      static_cast<long double>(options.velocity_limit));
        p.velocity[j] = static_cast<double>(capped);
        const auto positive = std::exp(-std::abs(p.velocity[j]));
        const auto probability = p.velocity[j] >= 0 ? 1 / (1 + positive) : positive / (1 + positive);
        p.point[j] = uniform(rng) < probability;
      }
    }
    update_personal(swarm, count, observe);
  }
  return result;
}

} // namespace datamunge::optim
