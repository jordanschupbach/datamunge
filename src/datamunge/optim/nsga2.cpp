/// @file nsga2.cpp
/// @brief Original-order checked NSGA-II with shared actual ranking, crowding and finite-capacity selection.
#include <algorithm>
#include <cmath>
#include <datamunge/optim/nsga2.hpp>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
namespace datamunge::optim {
/// @brief Private checked capture, actual rank/crowding producer and selected survivor ordering.
namespace {
/// @brief Checked addition for complete geometry and actual work/history counts.
/// @param a First count.
/// @param b Second count.
/// @return Checked sum.
/// @throws std::length_error If the sum overflows.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("NSGA-II: count overflow");
  return a + b;
}
/// @brief Checked multiplication for complete projected work/container geometry.
/// @param a First count.
/// @param b Second count.
/// @return Checked product.
/// @throws std::length_error If product overflows.
std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("NSGA-II: count geometry");
  return a * b;
}
/// @brief Exact unordered pair count, divided before multiplication.
/// @param n Complete population count.
/// @return n*(n-1)/2, zero for n<2.
/// @throws std::length_error If count is unrepresentable.
std::size_t pairs(std::size_t n) {
  return n < 2 ? 0 : n % 2 ? mul(n, (n - 1) / 2) : mul(n / 2, n - 1);
}
/// @brief One shared actual logical history accountant for nested producer records.
struct Capture {
  bool        enabled{false}; ///< True only for requested actual capture.
  std::size_t cap{0};         ///< Optional complete/online record-element quota, zero disables.
  std::size_t used{0};        ///< Actual admitted units across nested histories.

  /// @brief Admit actual future owned units before their allocation/copy/append.
  /// @param count Actual logical units being retained.
  /// @throws std::length_error If count geometry or selected policy is exceeded.
  void charge(std::size_t count) {
    if (!enabled)
      return;
    auto next = add(used, count);
    if (cap && next > cap)
      throw std::length_error("NSGA-II: actual capture quota");
    used = next;
  }
};
/// @brief Actual mutable individual, preserving original rank/crowding inheritance between generations.
struct Individual {
  std::vector<double> x;       ///< Owned coordinates.
  std::vector<double> f;       ///< Owned fixed finite objective values.
  std::size_t         rank{0}; ///< Actual zero-based rank from most recent complete sorting.
  double              crowding{
      0}; ///< Actual most recent processed-front crowding, infinity at selected boundaries.
};
/// @brief Add selected ranking/variation work without overflowing counters.
/// @param to Actual aggregate work to update.
/// @param from Actual completed subproducer work.
/// @throws std::length_error If a selected count overflows.
void accumulate(NSGA2Work& to, const NSGA2Work& from) {
  to.evaluations             = add(to.evaluations, from.evaluations);
  to.ranking_passes          = add(to.ranking_passes, from.ranking_passes);
  to.ranking_pairs           = add(to.ranking_pairs, from.ranking_pairs);
  to.ranking_dominance_calls = add(to.ranking_dominance_calls, from.ranking_dominance_calls);
  to.crowding_sorts          = add(to.crowding_sorts, from.crowding_sorts);
  to.crowding_gaps           = add(to.crowding_gaps, from.crowding_gaps);
  to.tournaments             = add(to.tournaments, from.tournaments);
  to.offspring_pairs         = add(to.offspring_pairs, from.offspring_pairs);
}
/// @brief Validate complete finite fixed objective cohorts before quadratic ranking state.
/// @param points Original finite input cohort, coordinates only metadata.
/// @return Fixed objective count, zero for an empty cohort.
/// @throws std::invalid_argument If objective shape/value or coordinates fail.
/// @throws std::length_error If complete cohort/pair geometry exceeds limits.
std::size_t validate_points(const std::vector<ParetoPoint>& points) {
  if (points.size() > 1000000)
    throw std::length_error("NSGA-II ranking: complete cohort cap");
  pairs(points.size());
  const auto m = points.empty() ? 0 : points[0].objectives.size();
  if (!points.empty() && (m < 2 || m > 1000000))
    throw std::invalid_argument("NSGA-II ranking: at least two fixed objectives required");
  for (auto& p : points) {
    if (p.objectives.size() != m)
      throw std::invalid_argument("NSGA-II ranking: mismatched objective count");
    for (double v : p.objectives)
      if (!std::isfinite(v))
        throw std::invalid_argument("NSGA-II ranking: finite objectives required");
    for (double v : p.coordinates)
      if (!std::isfinite(v))
        throw std::invalid_argument("NSGA-II ranking: finite coordinates required");
  }
  return m;
}
/// @brief Partition every individual with the original unordered-pair/decrement discovery schedule.
/// @param pop Validated actual individuals, ranks mutated in place.
/// @param out Optional actual pair-decision sink.
/// @param capture Shared actual history accountant.
/// @param work Actual ranking work counters.
/// @return Complete actual front-ID partition.
/// @throws std::length_error If actual history/count geometry is exceeded.
std::vector<std::vector<std::size_t>> sort_fronts(
    std::vector<Individual>& pop,
    NSGA2RankingTrace*       out,
    Capture&                 capture,
    NSGA2Work&               work) {
  const auto                            n = pop.size();
  std::vector<std::vector<std::size_t>> dominated(n);
  std::vector<std::size_t>              count(n);
  std::vector<std::vector<std::size_t>> fronts(1);
  ++work.ranking_passes;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = p + 1; q < n; ++q) {
      NSGA2DominancePair record;
      record.first  = p;
      record.second = q;
      record.calls  = 1;
      if (dominates(pop[p].f, pop[q].f)) {
        record.relation = -1;
        dominated[p].push_back(q);
        ++count[q];
      } else {
        ++record.calls;
        if (dominates(pop[q].f, pop[p].f)) {
          record.relation = 1;
          dominated[q].push_back(p);
          ++count[p];
        }
      }
      ++work.ranking_pairs;
      work.ranking_dominance_calls = add(work.ranking_dominance_calls, record.calls);
      if (out) {
        capture.charge(1);
        out->pairs.push_back(record);
      }
    }
  for (std::size_t p = 0; p < n; ++p)
    if (!count[p]) {
      pop[p].rank = 0;
      fronts[0].push_back(p);
    }
  std::size_t index = 0;
  while (!fronts[index].empty()) {
    std::vector<std::size_t> next;
    for (auto p : fronts[index])
      for (auto q : dominated[p])
        if (--count[q] == 0) {
          pop[q].rank = index + 1;
          next.push_back(q);
        }
    ++index;
    fronts.push_back(std::move(next));
  }
  fronts.pop_back();
  return fronts;
}
/// @brief Assign actual original-order crowding, retaining sorted IDs and zero-range boundary policy.
/// @param pop Actual rank-assigned individuals, crowding mutated only for this front.
/// @param front Actual front member IDs.
/// @param front_index Actual front ID in partition.
/// @param assigned Actual per-point processed flags to mark.
/// @param out Optional actual sorted-order sink.
/// @param capture Shared actual history accounting.
/// @param work Actual crowding work counters.
/// @throws std::overflow_error If finite objective differences/gaps overflow.
/// @throws std::length_error If actual capture geometry/policy is exceeded.
void crowd(
    std::vector<Individual>&        pop,
    const std::vector<std::size_t>& front,
    std::size_t                     front_index,
    std::vector<char>&              assigned,
    NSGA2RankingTrace*              out,
    Capture&                        capture,
    NSGA2Work&                      work) {
  for (auto i : front) {
    pop[i].crowding = 0;
    assigned[i]     = 1;
  }
  if (front.empty())
    return;
  const auto m = pop[front[0]].f.size();
  for (std::size_t k = 0; k < m; ++k) {
    auto order = front;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return pop[a].f[k] < pop[b].f[k];
    });
    ++work.crowding_sorts;
    pop[order.front()].crowding = std::numeric_limits<double>::infinity();
    pop[order.back()].crowding  = std::numeric_limits<double>::infinity();
    double range                = pop[order.back()].f[k] - pop[order.front()].f[k];
    if (!std::isfinite(range))
      throw std::overflow_error("NSGA-II: nonfinite crowding range");
    if (out) {
      capture.charge(add(1, order.size()));
      out->orders.push_back({front_index, k, order, range});
    }
    if (range <= 0)
      continue;
    for (std::size_t j = 1; j + 1 < order.size(); ++j) {
      auto i = order[j];
      if (pop[i].crowding == std::numeric_limits<double>::infinity())
        continue;
      double gap      = (pop[order[j + 1]].f[k] - pop[order[j - 1]].f[k]) / range;
      double distance = pop[i].crowding + gap;
      if (!std::isfinite(gap) || !std::isfinite(distance))
        throw std::overflow_error("NSGA-II: nonfinite crowding gap");
      pop[i].crowding = distance;
      ++work.crowding_gaps;
    }
  }
}
/// @brief Owned actual partition and selected IDs, with explicit processed-front flags.
struct Partition {
  std::vector<std::vector<std::size_t>> fronts;   ///< Actual complete dominance fronts.
  std::vector<char>                     assigned; ///< Actual per-point crowding-produced flags.
  std::vector<std::size_t>              selected; ///< Actual capacity-selected combined IDs.
  NSGA2Work work; ///< Actual selected work for this partition/survival producer.
};
/// @brief Copy actual point records after admitting their numeric elements.
/// @param pop Actual individuals to snapshot.
/// @param capture Shared optional accountant.
/// @return Owned point snapshots, excluding rank/crowding fields.
/// @throws std::length_error If actual retained numeric geometry/policy is exceeded.
std::vector<ParetoPoint> point_snapshot(const std::vector<Individual>& pop, Capture& capture) {
  std::vector<ParetoPoint> result;
  result.reserve(pop.size());
  for (auto& p : pop) {
    capture.charge(add(1, add(p.x.size(), p.f.size())));
    result.push_back({p.x, p.f});
  }
  return result;
}
/// @brief Own the actual partition values, using explicit placeholders only for unprocessed crowding.
/// @param pop Actual fully ranked individuals.
/// @param partition Actual partition/selected crowding flags.
/// @param capture Shared optional accountant.
/// @return Owned actual public ranking snapshot.
/// @throws std::length_error If actual owned-record/numeric quota is exceeded.
NSGA2Ranking
snapshot(const std::vector<Individual>& pop, const Partition& partition, Capture& capture) {
  NSGA2Ranking r;
  r.points = point_snapshot(pop, capture);
  capture.charge(mul(3, pop.size()));
  for (std::size_t i = 0; i < pop.size(); ++i) {
    r.ranks.push_back(pop[i].rank);
    r.crowding.push_back(partition.assigned[i] ? pop[i].crowding : 0);
    r.crowding_assigned.push_back(partition.assigned[i]);
  }
  for (auto& front : partition.fronts) {
    capture.charge(add(1, front.size()));
    r.fronts.push_back(front);
  }
  r.work = partition.work;
  return r;
}
/// @brief Execute original ranking and either all crowding or exactly the fronts needed for survival.
/// @param pop Actual individuals, ranks/processed crowding mutated.
/// @param keep Optional survivor capacity; absent processes all fronts without selection.
/// @param out Optional actual nested ranking history.
/// @param capture Shared actual history policy/accountant.
/// @return Actual partition, selected IDs, flags and selected work.
/// @throws std::length_error If actual history/count policy is exceeded.
/// @throws std::overflow_error If crowding arithmetic is nonfinite.
Partition partition(
    std::vector<Individual>&   pop,
    std::optional<std::size_t> keep,
    NSGA2RankingTrace*         out,
    Capture&                   capture) {
  const auto begin = capture.used;
  Partition  p;
  p.assigned.assign(pop.size(), 0);
  p.fronts = sort_fronts(pop, out, capture, p.work);
  for (std::size_t index = 0; index < p.fronts.size(); ++index) {
    if (keep && p.selected.size() == *keep)
      break;
    auto& front = p.fronts[index];
    crowd(pop, front, index, p.assigned, out, capture, p.work);
    if (!keep)
      continue;
    if (front.size() <= *keep - p.selected.size())
      p.selected.insert(p.selected.end(), front.begin(), front.end());
    else {
      auto order = front;
      std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return pop[a].crowding > pop[b].crowding;
      });
      for (std::size_t j = 0; p.selected.size() < *keep; ++j)
        p.selected.push_back(order[j]);
      break;
    }
  }
  if (out) {
    out->ranking          = snapshot(pop, p, capture);
    out->captured_entries = capture.used - begin;
  }
  return p;
}
/// @brief Convert validated original points into owned mutable ranking individuals.
/// @param points Original complete finite cohort.
/// @return Owned original-order individuals with fresh rank/crowding0.
std::vector<Individual> individuals(const std::vector<ParetoPoint>& points) {
  std::vector<Individual> pop;
  pop.reserve(points.size());
  for (auto& p : points)
    pop.push_back({p.coordinates, p.objectives, 0, 0});
  return pop;
}
/// @brief Validate complete optimizer controls/bounds and projected callback/pair work before callbacks.
/// @param options Original configured scalar controls.
/// @param lower Complete lower coordinate bounds.
/// @param upper Complete matching upper bounds.
/// @return Actual min2/even-rounded population count.
/// @throws std::invalid_argument If complete rates/indices/bounds fail.
/// @throws std::length_error If projected population/count geometry exceeds limits.
std::size_t validate(
    const NSGA2Options&        options,
    const std::vector<double>& lower,
    const std::vector<double>& upper) {
  if (lower.empty() || lower.size() != upper.size())
    throw std::invalid_argument("NSGA-II: nonempty matching bounds required");
  for (std::size_t j = 0; j < lower.size(); ++j)
    if (!std::isfinite(lower[j]) || !std::isfinite(upper[j]) || lower[j] >= upper[j] ||
        !std::isfinite(upper[j] - lower[j]))
      throw std::invalid_argument("NSGA-II: finite positive bound widths required");
  if (!std::isfinite(options.crossover_rate) || options.crossover_rate < 0 ||
      options.crossover_rate > 1 || !std::isfinite(options.eta_crossover) ||
      options.eta_crossover < 0 || !std::isfinite(options.mutation_rate) ||
      options.mutation_rate > 1 || !std::isfinite(options.eta_mutation) || options.eta_mutation < 0)
    throw std::invalid_argument("NSGA-II: invalid rates/indices");
  if (options.population_size > 1000000)
    throw std::length_error("NSGA-II: effective population cap");
  auto n = std::max<std::size_t>(options.population_size, 2);
  if (n % 2)
    ++n;
  add(n, mul(n, options.max_generations));
  add(1, options.max_generations);
  mul(2, add(pairs(n), mul(pairs(mul(2, n)), options.max_generations)));
  mul(n, lower.size());
  return n;
}
/// @brief Lower rank then strictly larger crowding, preserving second-draw tie behavior.
/// @param a Actual first tournament individual.
/// @param b Actual second tournament individual.
/// @return True iff original crowded comparison favors a over b.
bool crowded_less(const Individual& a, const Individual& b) {
  return a.rank != b.rank ? a.rank < b.rank : a.crowding > b.crowding;
}
/// @brief Admit retained final-result point rows/elements before owning them in the root trace.
/// @param points Actual complete final population/front.
/// @param capture Shared actual policy/accountant.
/// @throws std::length_error If retained geometry/policy is exceeded.
void admit_points(const std::vector<ParetoPoint>& points, Capture& capture) {
  for (auto& p : points)
    capture.charge(add(1, add(p.coordinates.size(), p.objectives.size())));
}
} // namespace
NSGA2Ranking nsga2_rank(const std::vector<ParetoPoint>& points) {
  validate_points(points);
  auto    pop = individuals(points);
  Capture capture;
  auto    p = partition(pop, std::nullopt, nullptr, capture);
  return snapshot(pop, p, capture);
}
NSGA2RankingTrace nsga2_rank_trace(const std::vector<ParetoPoint>& points, std::size_t cap) {
  validate_points(points);
  auto              pop = individuals(points);
  Capture           capture{true, cap, 0};
  NSGA2RankingTrace t;
  partition(pop, std::nullopt, &t, capture);
  return t;
}
NSGA2SurvivalTrace
nsga2_survival_trace(const std::vector<ParetoPoint>& points, std::size_t keep, std::size_t cap) {
  validate_points(points);
  if (keep > points.size())
    throw std::invalid_argument("NSGA-II survival: capacity exceeds input pool");
  auto               pop = individuals(points);
  Capture            capture{true, cap, 0};
  NSGA2SurvivalTrace t;
  auto               p = partition(pop, keep, &t.combined, capture);
  capture.charge(p.selected.size());
  t.selected         = std::move(p.selected);
  t.captured_entries = capture.used;
  return t;
}
NSGA2::NSGA2(NSGA2Options options) : options_(options) {}
NSGA2Run NSGA2::execute(
    MultiObjectiveFunction&    function,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    NSGA2Trace*                history,
    std::size_t                cap) const {
  const auto   N = validate(options_, lower, upper), n = lower.size();
  const double mutation = options_.mutation_rate < 0 ? 1. / double(n) : options_.mutation_rate;
  Capture      capture{history != nullptr, cap, 0};
  if (history) {
    capture.charge(add(7, add(lower.size(), upper.size())));
    history->options = options_;
    history->lower   = lower;
    history->upper   = upper;
  }
  NSGA2Run result;
  result.effective_population = N;
  std::mt19937_64                        rng(options_.seed);
  std::uniform_real_distribution<double> unit(0., 1.);
  /// @brief Draw one actual original-order box point from the shared local stream.
  auto random_point = [&]() {
    std::vector<double> x(n);
    for (std::size_t j = 0; j < n; ++j)
      x[j] = lower[j] + unit(rng) * (upper[j] - lower[j]);
    return x;
  };
  /// @brief Validate actual variation coordinates, evaluate once and admit finite fixed objective dimensions.
  auto evaluate = [&](const std::vector<double>& x) {
    if (x.size() != n)
      throw std::overflow_error("NSGA-II: variation coordinate shape");
    for (std::size_t j = 0; j < n; ++j)
      if (!std::isfinite(x[j]) || x[j] < lower[j] || x[j] > upper[j])
        throw std::overflow_error("NSGA-II: invalid variation coordinate");
    auto values             = function.evaluate(x);
    result.work.evaluations = add(result.work.evaluations, 1);
    if (!result.objectives) {
      if (values.size() < 2 || values.size() > 1000000)
        throw std::domain_error("NSGA-II: objective count must be2..1000000");
      result.objectives = values.size();
      mul(mul(mul(2, N), values.size()), add(1, options_.max_generations));
    }
    if (values.size() != result.objectives)
      throw std::domain_error("NSGA-II: objective count changed");
    for (double v : values)
      if (!std::isfinite(v))
        throw std::domain_error("NSGA-II: finite objectives required");
    return values;
  };
  std::vector<Individual> pop(N);
  for (auto& p : pop) {
    p.x = random_point();
    p.f = evaluate(p.x);
  }
  auto initial = partition(pop, std::nullopt, history ? &history->initial : nullptr, capture);
  accumulate(result.work, initial.work);
  for (std::size_t generation = 0; generation < options_.max_generations; ++generation) {
    NSGA2GenerationTrace* out = nullptr;
    if (history) {
      capture.charge(1);
      history->generations.emplace_back();
      out = &history->generations.back();
    }
    /// @brief Draw actual two slots with replacement and return the original crowded-comparison winner.
    auto tournament = [&]() {
      std::uniform_int_distribution<std::size_t> pick(0, pop.size() - 1);
      auto                                       a = pick(rng), b = pick(rng);
      auto                                       winner = crowded_less(pop[a], pop[b]) ? a : b;
      ++result.work.tournaments;
      if (out) {
        capture.charge(1);
        out->tournaments.push_back(
            {a, b, winner, pop[a].rank, pop[b].rank, pop[a].crowding, pop[b].crowding});
      }
      return winner;
    };
    std::vector<Individual> offspring;
    offspring.reserve(N);
    while (offspring.size() < N) {
      auto parent1 = tournament(), parent2 = tournament();
      ++result.work.offspring_pairs;
      if (out) {
        capture.charge(1);
        out->parents.push_back({parent1, parent2, N + offspring.size(), N + offspring.size() + 1});
      }
      std::vector<double> c1, c2;
      detail::sbx_crossover(
          pop[parent1].x,
          pop[parent2].x,
          c1,
          c2,
          lower,
          upper,
          options_.crossover_rate,
          options_.eta_crossover,
          rng);
      detail::polynomial_mutation(c1, lower, upper, mutation, options_.eta_mutation, rng);
      detail::polynomial_mutation(c2, lower, upper, mutation, options_.eta_mutation, rng);
      Individual first{std::move(c1), {}, 0, 0};
      first.f = evaluate(first.x);
      offspring.push_back(std::move(first));
      Individual second{std::move(c2), {}, 0, 0};
      second.f = evaluate(second.x);
      offspring.push_back(std::move(second));
    }
    std::vector<Individual> combined;
    combined.reserve(2 * N);
    combined.insert(
        combined.end(), std::make_move_iterator(pop.begin()), std::make_move_iterator(pop.end()));
    combined.insert(
        combined.end(),
        std::make_move_iterator(offspring.begin()),
        std::make_move_iterator(offspring.end()));
    auto selection = partition(combined, N, out ? &out->combined : nullptr, capture);
    accumulate(result.work, selection.work);
    if (out) {
      capture.charge(selection.selected.size());
      out->selected = selection.selected;
    }
    std::vector<Individual> next;
    next.reserve(N);
    for (auto i : selection.selected)
      next.push_back(std::move(combined[i]));
    pop = std::move(next);
  }
  result.population.reserve(pop.size());
  for (auto& p : pop)
    result.population.push_back({std::move(p.x), std::move(p.f)});
  result.front = non_dominated_front(result.population);
  if (history) {
    admit_points(result.population, capture);
    admit_points(result.front, capture);
    history->captured_entries = capture.used;
  }
  return result;
}
std::vector<ParetoPoint> NSGA2::optimize(
    MultiObjectiveFunction&    f,
    const std::vector<double>& lower,
    const std::vector<double>& upper) const {
  return execute(f, lower, upper, nullptr, 0).front;
}
NSGA2Run NSGA2::run(
    MultiObjectiveFunction&    f,
    const std::vector<double>& lower,
    const std::vector<double>& upper) const {
  return execute(f, lower, upper, nullptr, 0);
}
NSGA2Trace NSGA2::trace(
    MultiObjectiveFunction&    f,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    std::size_t                cap) const {
  NSGA2Trace t;
  t.run = execute(f, lower, upper, &t, cap);
  return t;
}
std::vector<ParetoPoint> nsga2_demo_points() {
  return {
      {{}, {.1, .9}},
      {{}, {.3, .4}},
      {{}, {.6, .2}},
      {{}, {.9, .1}},
      {{}, {.5, .6}},
      {{}, {.8, .5}}};
}
std::vector<ParetoPoint> nsga2_truncation_points() {
  return {
      {{}, {0, 1}}, {{}, {.2, .8}}, {{}, {.4, .6}}, {{}, {.6, .4}}, {{}, {.8, .2}}, {{}, {1, 0}}};
}
std::vector<ParetoPoint> nsga2_sorted_front(const std::vector<ParetoPoint>& front) {
  for (auto& p : front) {
    if (p.objectives.empty())
      throw std::invalid_argument("NSGA-II presentation: missing objective");
    for (double v : p.objectives)
      if (!std::isfinite(v))
        throw std::invalid_argument("NSGA-II presentation: nonfinite objective");
    for (double v : p.coordinates)
      if (!std::isfinite(v))
        throw std::invalid_argument("NSGA-II presentation: nonfinite coordinates");
  }
  auto result = front;
  std::sort(result.begin(), result.end(), [](auto& a, auto& b) {
    return a.objectives[0] < b.objectives[0];
  });
  return result;
}
} // namespace datamunge::optim
