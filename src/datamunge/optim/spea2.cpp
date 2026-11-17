/// @file spea2.cpp
/// @brief Single actual SPEA2 scoring, archive selection and seeded producing pipeline.
#include <algorithm>
#include <cmath>
#include <datamunge/optim/spea2.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

/// @brief Canonical checked native producers and owned values.
namespace datamunge::optim {

/// @brief Private invocation-local checked producing helpers.
namespace {

using Individual = SPEA2Candidate; ///< Same actual candidate record exposed in captured pools.

/// @brief Invocation-local actual work and copied-history accounting.
struct Context {
  const SPEA2Options&       options;    ///< Borrowed controls for this call only.
  SPEA2Run&                 run;        ///< Borrowed actual counter/result accumulator.
  std::size_t               retained{}; ///< Actual copied history units retained.
  std::vector<std::size_t>* removed{};  ///< Optional actual victim identities for this selection.

  /// @brief Increment actual work before policy/capacity overflow.
  /// @param value Counter.
  /// @param cap Zero or policy.
  void count(std::size_t& value, std::size_t cap) {
    if (value == std::numeric_limits<std::size_t>::max() || (cap && value >= cap))
      throw std::length_error("SPEA2: work");
    ++value;
  }
  /// @brief Bound actual retained units before copy.
  /// @param n New retained units.
  void copy(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() - retained)
      throw std::length_error("SPEA2: history capacity");
    retained += n;
    if (options.max_history_values && retained > options.max_history_values)
      throw std::length_error("SPEA2: history");
  }
};
/// @brief Validate complete finite objective shape.
/// @param f Actual complete objective vector.
/// @param dimension Zero accepts first positive dimension; otherwise required dimension.
void objective(const std::vector<double>& f, std::size_t dimension) {
  if (f.empty() || (dimension && f.size() != dimension))
    throw std::invalid_argument("SPEA2: objective dimension");
  for (double x : f)
    if (!std::isfinite(x))
      throw std::invalid_argument("SPEA2: finite objectives");
}

/// @brief Validate complete finite strict bound spans.
/// @param lower Complete lower bounds.
/// @param upper Complete upper bounds.
void validate_bounds(const std::vector<double>& lower, const std::vector<double>& upper) {
  const std::size_t n = lower.size();
  if (n == 0)
    throw std::invalid_argument("SPEA2::optimize: bounds must not be empty");
  if (upper.size() != n)
    throw std::invalid_argument("SPEA2::optimize: lower/upper bound sizes must match");
  for (std::size_t j = 0; j < n; ++j)
    if (!std::isfinite(lower[j]) || !std::isfinite(upper[j]) ||
        !std::isfinite(upper[j] - lower[j]) || !(lower[j] < upper[j]))
      throw std::invalid_argument(
          "SPEA2::optimize: lower_bound must be < upper_bound in every dimension");
}

/// @brief Actual finite Euclidean distance used by density and truncation.
/// @param a First actual objective point.
/// @param b Second actual objective point.
/// @param context Actual work accumulator.
/// @return Finite Euclidean objective distance.
double objective_distance(const Individual& a, const Individual& b, Context& context) {
  context.count(context.run.distance_calls, context.options.max_distance_calls);
  long double norm = 0;
  for (std::size_t k = 0; k < a.f.size(); ++k)
    norm = std::hypot(norm, static_cast<long double>(a.f[k]) - b.f[k]);
  if (!std::isfinite(norm) || norm > std::numeric_limits<double>::max())
    throw std::overflow_error("SPEA2: objective distance");
  return static_cast<double>(norm);
}

// SPEA2 fitness: F(i) = raw(i) + density(i). raw(i) is the sum of the strengths (dominated
// counts) of every solution dominating i (0 iff i is non-dominated); density(i) =
// 1 / (dist to the k-th nearest neighbor + 2), k = floor(sqrt(|pool|)).
/// @brief Score the actual complete pool with published one-based neighbor density.
/// @param pool Mutable actual candidate pool.
/// @param context Actual invocation counters/policies.
void assign_fitness(std::vector<Individual>& pool, Context& context) {
  const std::size_t P = pool.size();
  if (P && P > std::numeric_limits<std::size_t>::max() / P)
    throw std::length_error("SPEA2: dominance matrix capacity");
  if (context.options.max_pool_size && P > context.options.max_pool_size)
    throw std::length_error("SPEA2: pool");
  std::vector<std::size_t>       strength(P, 0);
  std::vector<std::vector<bool>> dom(P, std::vector<bool>(P, false));
  for (std::size_t i = 0; i < P; ++i)
    for (std::size_t j = 0; j < P; ++j)
      if (i != j &&
          (context.count(context.run.dominance_tests, context.options.max_dominance_tests),
           dominates(pool[i].f, pool[j].f))) {
        dom[i][j] = true;
        ++strength[i];
      }

  for (std::size_t i = 0; i < P; ++i) {
    double raw = 0.0;
    for (std::size_t j = 0; j < P; ++j)
      if (dom[j][i])
        raw += strength[j];

    std::vector<double> dists;
    dists.reserve(P - 1);
    for (std::size_t j = 0; j < P; ++j)
      if (i != j)
        dists.push_back(objective_distance(pool[i], pool[j], context));
    std::sort(dists.begin(), dists.end());
    const std::size_t k   = static_cast<std::size_t>(std::sqrt(static_cast<double>(P)));
    const double      kth = dists.empty() ? 0.0 : dists[std::min(k ? k - 1 : 0, dists.size() - 1)];
    pool[i].pool_index    = i;
    pool[i].strength      = strength[i];
    pool[i].raw           = raw;
    pool[i].density       = 1.0 / (kth + 2.0);
    pool[i].fitness       = raw + pool[i].density;
    if (!std::isfinite(pool[i].fitness))
      throw std::overflow_error("SPEA2: fitness");
  }
}

// Environmental selection: keep every non-dominated solution (fitness < 1); then fill from the
// best dominated ones, or truncate the most crowded ones, to reach exactly `cap`.
/// @brief Select actual archive candidates, fill or truncate with real identities.
/// @param pool Owned actual scored pool.
/// @param cap Requested positive archive capacity.
/// @param context Actual counters and optional removed-identity sink.
/// @return Actual retained archive in selection order.
std::vector<Individual>
environmental_selection(std::vector<Individual> pool, std::size_t cap, Context& context) {
  std::vector<Individual> archive;
  std::vector<Individual> dominated;
  for (auto& ind : pool) {
    if (ind.fitness < 1.0)
      archive.push_back(std::move(ind));
    else
      dominated.push_back(std::move(ind));
  }

  if (archive.size() < cap) {
    std::sort(dominated.begin(), dominated.end(), [](const Individual& a, const Individual& b) {
      return a.fitness < b.fitness || (a.fitness == b.fitness && a.pool_index < b.pool_index);
    });
    for (std::size_t i = 0; i < dominated.size() && archive.size() < cap; ++i)
      archive.push_back(std::move(dominated[i]));
    return archive;
  }

  // Truncation: repeatedly remove the individual whose distance to its nearest neighbor is
  // smallest (ties broken by the next-nearest, and so on).
  while (archive.size() > cap) {
    const std::size_t                A = archive.size();
    std::vector<std::vector<double>> sorted_d(A);
    for (std::size_t i = 0; i < A; ++i) {
      sorted_d[i].reserve(A - 1);
      for (std::size_t j = 0; j < A; ++j)
        if (i != j)
          sorted_d[i].push_back(objective_distance(archive[i], archive[j], context));
      std::sort(sorted_d[i].begin(), sorted_d[i].end());
    }
    std::size_t victim = 0;
    for (std::size_t i = 1; i < A; ++i) {
      // Lexicographic comparison of the sorted nearest-neighbor distance lists.
      bool i_more_crowded = false, decided = false;
      for (std::size_t t = 0; t < sorted_d[i].size(); ++t) {
        if (sorted_d[i][t] < sorted_d[victim][t]) {
          i_more_crowded = true;
          decided        = true;
          break;
        }
        if (sorted_d[i][t] > sorted_d[victim][t]) {
          decided = true;
          break;
        }
      }
      if (decided && i_more_crowded)
        victim = i;
    }
    context.count(context.run.archive_removals, 0);
    if (context.removed) {
      context.copy(1);
      context.removed->push_back(archive[victim].pool_index);
    }
    archive.erase(archive.begin() + static_cast<std::ptrdiff_t>(victim));
  }
  return archive;
}

} // namespace

SPEA2::SPEA2(SPEA2Options options) : options_(options) {}

/// @brief Private invocation-local checked producing helpers.
namespace {
/// @brief Execute seeded generations with optional real scored-pool capture.
/// @param function Actual mutable objective callback.
/// @param lower_bound Complete lower bounds.
/// @param upper_bound Complete upper bounds.
/// @param options_ Complete controls.
/// @param capture Retain actual pools/selection states.
/// @return Actual owned front and counters.
SPEA2Run produce(
    MultiObjectiveFunction&    function,
    const std::vector<double>& lower_bound,
    const std::vector<double>& upper_bound,
    const SPEA2Options&        options_,
    bool                       capture) {
  SPEA2Run result;
  Context  context{options_, result};
  if (!std::isfinite(options_.crossover_rate) || options_.crossover_rate < 0 ||
      options_.crossover_rate > 1 || !std::isfinite(options_.mutation_rate) ||
      options_.mutation_rate > 1 || !std::isfinite(options_.eta_crossover) ||
      options_.eta_crossover <= 0 || !std::isfinite(options_.eta_mutation) ||
      options_.eta_mutation <= 0)
    throw std::invalid_argument("SPEA2: controls");
  validate_bounds(lower_bound, upper_bound);
  const std::size_t n    = lower_bound.size();
  const std::size_t N    = std::max<std::size_t>(options_.population_size, 2);
  const std::size_t Nbar = options_.archive_size == 0 ? N : options_.archive_size;
  if (N > std::numeric_limits<std::size_t>::max() - Nbar ||
      (options_.max_pool_size && N + Nbar > options_.max_pool_size))
    throw std::length_error("SPEA2: complete pool geometry");
  const double mut_rate =
      options_.mutation_rate < 0.0 ? 1.0 / static_cast<double>(n) : options_.mutation_rate;

  std::mt19937_64                        rng(options_.seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  /// @brief Draw one actual bounded initial point from the invocation-local engine.
  /// @return Owned actual coordinates.
  auto random_point = [&]() {
    std::vector<double> x(n);
    for (std::size_t j = 0; j < n; ++j)
      x[j] = std::lerp(lower_bound[j], upper_bound[j], u(rng));
    return x;
  };
  std::size_t dimension = 0;
  /// @brief Invoke and validate one actual objective callback.
  /// @param x Owned actual candidate coordinates.
  /// @return Owned evaluated candidate.
  auto evaluate = [&](std::vector<double> x) {
    for (std::size_t j = 0; j < x.size(); ++j)
      if (!std::isfinite(x[j]) || x[j] < lower_bound[j] || x[j] > upper_bound[j])
        throw std::overflow_error("SPEA2: offspring bounds");
    context.count(result.evaluations, options_.max_evaluations);
    Individual ind;
    ind.f = function.evaluate(x);
    objective(ind.f, dimension);
    dimension = ind.f.size();
    ind.x     = std::move(x);
    return ind;
  };

  std::vector<Individual> population;
  population.reserve(N);
  for (std::size_t i = 0; i < N; ++i)
    population.push_back(evaluate(random_point()));
  std::vector<Individual> archive;

  for (std::size_t gen = 0; gen < options_.max_generations; ++gen) {
    std::vector<Individual> pool;
    pool.reserve(population.size() + archive.size());
    pool.insert(pool.end(), population.begin(), population.end());
    pool.insert(pool.end(), archive.begin(), archive.end());
    assign_fitness(pool, context);
    if (capture) {
      context.copy(1);
      for (const auto& ind : pool) {
        context.copy(5);
        context.copy(ind.x.size());
        context.copy(ind.f.size());
      }
      result.generations.push_back({gen, result.evaluations, pool, {}, {}});
      context.removed = &result.generations.back().truncated;
    }
    archive = environmental_selection(std::move(pool), Nbar, context);
    if (capture) {
      context.copy(archive.size());
      for (const auto& ind : archive)
        result.generations.back().selected.push_back(ind.pool_index);
      context.removed = nullptr;
    }

    if (gen + 1 == options_.max_generations)
      break;

    // Mating selection: binary tournament on the archive (lower fitness wins), then vary.
    /// @brief Choose an actual mating parent from the current archive.
    /// @return Borrowed actual selected parent, valid until archive mutation.
    auto tournament = [&]() -> const Individual& {
      std::uniform_int_distribution<std::size_t> pick(0, archive.size() - 1);
      const std::size_t                          a = pick(rng), b = pick(rng);
      return archive[a].fitness <= archive[b].fitness ? archive[a] : archive[b];
    };
    std::vector<Individual> next;
    next.reserve(N);
    while (next.size() < N) {
      const Individual&   p1 = tournament();
      const Individual&   p2 = tournament();
      std::vector<double> c1, c2;
      detail::sbx_crossover(
          p1.x,
          p2.x,
          c1,
          c2,
          lower_bound,
          upper_bound,
          options_.crossover_rate,
          options_.eta_crossover,
          rng);
      detail::polynomial_mutation(
          c1, lower_bound, upper_bound, mut_rate, options_.eta_mutation, rng);
      next.push_back(evaluate(std::move(c1)));
      if (next.size() < N) {
        detail::polynomial_mutation(
            c2, lower_bound, upper_bound, mut_rate, options_.eta_mutation, rng);
        next.push_back(evaluate(std::move(c2)));
      }
    }
    population = std::move(next);
  }

  std::vector<ParetoPoint> points;
  points.reserve(archive.size());
  for (auto& ind : archive)
    points.push_back(ParetoPoint{std::move(ind.x), std::move(ind.f)});
  result.front = non_dominated_front(points);
  return result;
}

} // namespace
SPEA2Run SPEA2::optimize_run(
    MultiObjectiveFunction&    f,
    const std::vector<double>& lo,
    const std::vector<double>& hi) const {
  return produce(f, lo, hi, options_, false);
}
SPEA2Run SPEA2::optimize_trace(
    MultiObjectiveFunction&    f,
    const std::vector<double>& lo,
    const std::vector<double>& hi) const {
  return produce(f, lo, hi, options_, true);
}
std::vector<ParetoPoint> SPEA2::optimize(
    MultiObjectiveFunction&    f,
    const std::vector<double>& lo,
    const std::vector<double>& hi) const {
  return optimize_run(f, lo, hi).front;
}
SPEA2Selection spea2_select_trace(
    const std::vector<std::vector<double>>& values,
    std::size_t                             capacity,
    const SPEA2Options&                     options) {
  if (!capacity)
    throw std::invalid_argument("SPEA2: archive capacity");
  if (options.max_pool_size && values.size() > options.max_pool_size)
    throw std::length_error("SPEA2: objective pool");
  SPEA2Run                run;
  Context                 context{options, run};
  std::vector<Individual> pool;
  std::size_t             dimension = 0;
  for (const auto& f : values) {
    objective(f, dimension);
    dimension = f.size();
    context.copy(5);
    context.copy(f.size());
    Individual ind;
    ind.f = f;
    pool.push_back(std::move(ind));
  }
  assign_fitness(pool, context);
  SPEA2Selection out;
  out.pool        = pool;
  context.removed = &out.truncated;
  auto archive    = environmental_selection(std::move(pool), capacity, context);
  context.copy(archive.size());
  for (const auto& ind : archive)
    out.selected.push_back(ind.pool_index);
  out.dominance_tests = run.dominance_tests;
  out.distance_calls  = run.distance_calls;
  return out;
}
} // namespace datamunge::optim
