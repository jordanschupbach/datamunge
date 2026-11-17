/// @file memetic.cpp
/// @brief Canonical memetic API, producing histories and checked native contracts.

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/memetic.hpp>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace datamunge::algorithms {
/// @brief Implementation-only helpers for canonical memetic production and capture.
namespace {
/// @brief Checked logical/operation count addition.
/// @param a Existing count.
/// @param b Additional count.
/// @return Exact sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a, std::size_t b = 1) {
  if (b > SIZE_MAX - a)
    throw std::length_error("memetic count geometry");
  return a + b;
}
/// @brief Checked allocation/capture geometry product.
/// @param a First dimension.
/// @param b Second dimension.
/// @return Exact product.
/// @throws std::length_error On size_t overflow.
std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > SIZE_MAX / a)
    throw std::length_error("memetic allocation geometry");
  return a * b;
}
/// @brief Validate every original weighted field and exact reverse-edge multiplicity.
/// @param n Original nonnegative vertex count.
/// @param graph Original ordered lists.
/// @return Actual directed listed-entry count.
/// @throws std::invalid_argument On shape/id/self/finite/symmetry violations.
/// @throws std::overflow_error If total absolute undirected weight exceeds finite double range.
/// @throws std::length_error On listed/count geometry overflow.
std::size_t validate(int n, const MemeticGraph& graph) {
  if (n < 0 || graph.size() != static_cast<std::size_t>(n))
    throw std::invalid_argument("memetic graph rows");
  std::map<std::tuple<int, int, double>, std::size_t> counts;
  std::size_t                                         listed   = 0;
  long double                                         absolute = 0;
  for (int u = 0; u < n; ++u) {
    listed = add(listed, graph[u].size());
    for (auto [v, w] : graph[u]) {
      if (v < 0 || v >= n || u == v || !std::isfinite(w))
        throw std::invalid_argument("memetic graph edge");
      auto& count = counts[{u, v, w}];
      count       = add(count);
      if (u < v)
        absolute += std::abs(static_cast<long double>(w));
    }
  }
  for (auto& [edge, count] : counts) {
    auto [u, v, w] = edge;
    auto reverse   = counts.find({v, u, w});
    if (reverse == counts.end() || reverse->second != count)
      throw std::invalid_argument("memetic reverse weight multiplicity");
  }
  if (absolute > std::numeric_limits<double>::max())
    throw std::overflow_error("memetic absolute weight range");
  return listed;
}
/// @brief Validate every supplied binary partition bit before objective indexing.
/// @param n Valid vertex count.
/// @param side Original side vector.
/// @throws std::invalid_argument On wrong length or nonbinary values.
void validate_side(int n, const std::vector<int>& side) {
  if (side.size() != static_cast<std::size_t>(n))
    throw std::invalid_argument("memetic side length");
  for (int x : side)
    if (x != 0 && x != 1)
      throw std::invalid_argument("memetic binary side");
}
/// @brief Canonical original-row-order objective, optionally counting actual inspected references.
/// @param graph Fully validated ordered weighted graph.
/// @param side Fully validated or natively produced binary partition.
/// @param work Optional actual counters.
/// @return Actual finite crossing-edge sum, counting each listed u<v entry once.
/// @throws std::overflow_error On nonfinite computed objective.
/// @throws std::length_error On actual reference-count overflow.
double objective(const MemeticGraph& graph, const std::vector<int>& side, MemeticWork* work) {
  double value = 0;
  for (std::size_t u = 0; u < graph.size(); ++u)
    for (auto [v, w] : graph[u]) {
      if (work)
        work->fitness_references = add(work->fitness_references);
      if (u < static_cast<std::size_t>(v) && side[u] != side[v])
        value += w;
    }
  if (!std::isfinite(value))
    throw std::overflow_error("memetic computed objective");
  return value;
}
/// @brief Legacy13/7/17 uint64 xorshift state, with retained seed-zero fallback.
struct Rng {
  std::uint64_t state; ///< Actual current unsigned state/output.

  /// @brief Initialize the original state convention.
  /// @param seed Original seed, zero selects the original nonzero fallback.
  explicit Rng(std::uint64_t seed) : state(seed ? seed : 0x2545F4914F6CDD1DULL) {}
  /// @brief Advance once with defined unsigned shifts/XOR.
  /// @return Actual updated state and raw output.
  std::uint64_t next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
};
/// @brief Admit actual complete logical capture contributions before owned copies.
/// @param trace Optional actual history target.
/// @param limit Optional complete cap, zero off.
/// @param count Actual next entries.
/// @throws std::length_error On overflow or exceeded optional cap.
void admit(MemeticTrace* trace, std::size_t limit, std::size_t count) {
  if (!trace)
    return;
  auto next = add(trace->captured_entries, count);
  if (limit && next > limit)
    throw std::length_error("memetic complete capture");
  trace->captured_entries = next;
}
/// @brief One canonical original-draw-order population/local-search producer with optional bounded history.
/// @param n Original vertex count.
/// @param graph Fully validated original lists.
/// @param listed Actual complete listed-entry count.
/// @param population Positive requested population size.
/// @param generations Nonnegative requested offspring generation count.
/// @param seed Original RNG seed.
/// @param max_checks Optional actual vertex-delta cap, zero unlimited.
/// @param trace Optional owned actual capture.
/// @param limit Optional complete logical capture cap.
/// @return Actual best witness, completion/refinement status and selected work.
/// @throws std::invalid_argument On malformed population/generation request.
/// @throws std::length_error On actual/allocation/capture geometry overflow.
/// @throws std::overflow_error On nonfinite actual delta/objective.
MemeticAnalysis produce(
    int                 n,
    const MemeticGraph& graph,
    std::size_t         listed,
    int                 population,
    int                 generations,
    std::uint64_t       seed,
    std::size_t         max_checks,
    MemeticTrace*       trace,
    std::size_t         limit) {
  if (population <= 0 || generations < 0)
    throw std::invalid_argument("memetic population/generations");
  auto count = static_cast<std::size_t>(population), vertices = static_cast<std::size_t>(n);
  auto bits = multiply(count, vertices);
  if (bits > SIZE_MAX / sizeof(int))
    throw std::length_error("memetic population bytes");
  if (trace) {
    admit(trace, limit, add(add(5, multiply(2, vertices)), multiply(2, listed)));
    trace->n                     = n;
    trace->graph                 = graph;
    trace->population_size       = population;
    trace->requested_generations = generations;
    trace->seed                  = seed;
    trace->max_local_checks      = max_checks;
  }
  MemeticAnalysis     out;
  MemeticMaxCutResult best;
  bool                has_best = false, best_refined = false;
  Rng                 rng(seed);
  /// @brief Advance/capture one actual RNG draw in its real native use context.
  /// @param kind Actual purpose0 initialization,1 tournament,2 crossover,3 mutation.
  /// @param generation Actual zero-based offspring generation or -1 initialization.
  /// @param individual Actual output population slot.
  /// @param slot Actual vertex or tournament draw slot.
  /// @param bound Actual modulo bound, zero for mutation uniform.
  /// @return Actual raw state/output.
  /// @throws std::length_error On actual count/capture overflow.
  auto draw = [&](unsigned    kind,
                  int         generation,
                  std::size_t individual,
                  std::size_t slot,
                  std::size_t bound) {
    auto before = rng.state, raw = rng.next();
    out.work.rng_draws = add(out.work.rng_draws);
    if (trace) {
      admit(trace, limit, 1);
      trace->draws.push_back(
          {kind,
           generation,
           individual,
           slot,
           before,
           raw,
           bound,
           bound ? static_cast<std::size_t>(raw % bound) : 0,
           kind == 3 ? (raw >> 11) * (1.0 / 9007199254740992.0) : 0});
    }
    return raw;
  };
  /// @brief Refine one actual candidate by sequential strict-gain vertex flips.
  /// @param side Mutable actual candidate.
  /// @param generation Actual candidate generation label.
  /// @param individual Actual output slot.
  /// @return True on a real completed no-flip sweep, false on the explicit local-check cap.
  /// @throws std::length_error On actual count/capture overflow.
  /// @throws std::overflow_error On nonfinite computed delta.
  auto local = [&](std::vector<int>& side, int generation, std::size_t individual) {
    bool improved = true;
    while (improved) {
      improved              = false;
      auto sweep            = out.work.local_sweeps;
      out.work.local_sweeps = add(out.work.local_sweeps);
      for (int u = 0; u < n; ++u) {
        if (max_checks && out.work.local_checks >= max_checks)
          return false;
        double delta = 0;
        for (auto [v, w] : graph[u]) {
          out.work.delta_references = add(out.work.delta_references);
          delta += side[v] != side[u] ? -w : w;
        }
        if (!std::isfinite(delta))
          throw std::overflow_error("memetic computed delta");
        int  before = side[u];
        bool flip   = delta > 1e-12;
        if (flip) {
          side[u] ^= 1;
          improved       = true;
          out.work.flips = add(out.work.flips);
        }
        out.work.local_checks = add(out.work.local_checks);
        if (trace) {
          admit(trace, limit, 1);
          trace->local_steps.push_back(
              {generation, individual, sweep, u, before, delta, flip, side[u]});
        }
      }
    }
    return true;
  };
  /// @brief Refine/evaluate/capture one real candidate and update strict global best.
  /// @param side Actual candidate before/after local refinement.
  /// @param generation Actual generation label.
  /// @param slot Actual output slot.
  /// @param parent_a Actual first selected parent or -1 initialization.
  /// @param parent_b Actual second selected parent or -1 initialization.
  /// @param first_draw Actual global RNG count at candidate creation start.
  /// @param value Actual candidate objective output.
  /// @return Whether local refinement completed.
  /// @throws std::length_error On actual capture/count overflow.
  /// @throws std::overflow_error On actual nonfinite delta/objective.
  auto individual = [&](std::vector<int>& side,
                        int               generation,
                        std::size_t       slot,
                        int               parent_a,
                        int               parent_b,
                        std::size_t       first_draw,
                        double&           value) {
    MemeticIndividual record;
    if (trace) {
      admit(trace, limit, add(1, multiply(2, vertices)));
      record.generation   = generation;
      record.slot         = slot;
      record.parent_a     = parent_a;
      record.parent_b     = parent_b;
      record.before_local = side;
      record.first_draw   = first_draw;
      record.past_draw    = out.work.rng_draws;
      record.first_local  = out.work.local_checks;
    }
    bool refined         = local(side, generation, slot);
    value                = objective(graph, side, &out.work);
    out.work.individuals = add(out.work.individuals);
    if (!has_best || value > best.value) {
      has_best              = true;
      best                  = {value, side};
      best_refined          = refined;
      out.work.improvements = add(out.work.improvements);
    }
    if (trace) {
      record.after_local = side;
      record.fitness     = value;
      record.refined     = refined;
      record.past_local  = out.work.local_checks;
      trace->individuals.push_back(std::move(record));
    }
    return refined;
  };
  /// @brief Return actual first strictly maximum member of one installed population.
  /// @param fitness Actual member fitness values.
  /// @return Actual best member index.
  /// @throws std::length_error On comparison-count overflow.
  auto best_index = [&](const std::vector<double>& fitness) {
    std::size_t b = 0;
    for (std::size_t p = 1; p < fitness.size(); ++p) {
      out.work.best_comparisons = add(out.work.best_comparisons);
      if (fitness[p] > fitness[b])
        b = p;
    }
    return b;
  };
  /// @brief Capture one real fully installed population and its actual selected best.
  /// @param index Actual generation index, zero initialization.
  /// @param pop Actual complete population.
  /// @param fit Actual corresponding objectives.
  /// @param best_member Actual first strict maximum index.
  /// @throws std::length_error On complete capture geometry/admission overflow.
  auto snapshot = [&](int                                  index,
                      const std::vector<std::vector<int>>& pop,
                      const std::vector<double>&           fit,
                      std::size_t                          best_member) {
    if (trace) {
      admit(trace, limit, add(add(1, bits), count));
      long double                total = 0;
      std::set<std::vector<int>> unique;
      for (std::size_t i = 0; i < pop.size(); ++i) {
        total += fit[i];
        auto side = pop[i];
        if (!side.empty() && side[0])
          for (int& bit : side)
            bit ^= 1;
        unique.insert(std::move(side));
      }
      trace->generations.push_back(
          {index,
           pop,
           fit,
           best_member,
           fit[best_member],
           double(total / pop.size()),
           unique.size()});
    }
  };
  /// @brief Finalize actual best witness/status without substituting uncomputed partitions.
  /// @return Actual current best candidate and selected work/status.
  auto finish = [&]() {
    out.result          = std::move(best);
    out.locally_refined = best_refined;
    return std::move(out);
  };
  std::vector<std::vector<int>> pop(count, std::vector<int>(vertices));
  std::vector<double>           fit(count);
  for (std::size_t p = 0; p < count; ++p) {
    auto first = out.work.rng_draws;
    for (std::size_t u = 0; u < vertices; ++u)
      pop[p][u] = draw(0, -1, p, u, 2) & 1;
    if (!individual(pop[p], -1, p, -1, -1, first, fit[p])) {
      out.complete = false;
      return finish();
    }
  }
  out.initialized = true;
  auto elite      = best_index(fit);
  snapshot(0, pop, fit, elite);
  for (int generation = 0; generation < generations; ++generation) {
    std::vector<std::vector<int>> next;
    std::vector<double>           next_fit;
    next.reserve(count);
    next_fit.reserve(count);
    next.push_back(pop[elite]);
    next_fit.push_back(fit[elite]);
    while (next.size() < count) {
      auto p = next.size(), first = out.work.rng_draws;
      /// @brief Perform the actual native three-draw strict-fitness parent tournament.
      /// @param parent Actual parent number0 or1, used only to label draw slots.
      /// @return Actual selected member index.
      auto tournament = [&](std::size_t parent) {
        auto b = static_cast<std::size_t>(draw(1, generation, p, 3 * parent, count) % count);
        for (std::size_t t = 1; t < 3; ++t) {
          auto c = static_cast<std::size_t>(draw(1, generation, p, 3 * parent + t, count) % count);
          if (fit[c] > fit[b])
            b = c;
        }
        return b;
      };
      auto             a = tournament(0), b = tournament(1);
      std::vector<int> child(vertices);
      for (std::size_t u = 0; u < vertices; ++u)
        child[u] = (draw(2, generation, p, u, 2) & 1) ? pop[a][u] : pop[b][u];
      for (std::size_t u = 0; u < vertices; ++u)
        if ((draw(3, generation, p, u, 0) >> 11) * (1.0 / 9007199254740992.0) < 1.0 / n)
          child[u] ^= 1;
      double value;
      bool   refined =
          individual(child, generation, p, static_cast<int>(a), static_cast<int>(b), first, value);
      if (!refined) {
        out.complete = false;
        return finish();
      }
      next.push_back(std::move(child));
      next_fit.push_back(value);
    }
    pop                       = std::move(next);
    fit                       = std::move(next_fit);
    out.generations_completed = generation + 1;
    elite                     = best_index(fit);
    snapshot(generation + 1, pop, fit, elite);
  }
  return finish();
}
} // namespace
MemeticMaxCutResult memetic_max_cut(
    int                 n,
    const MemeticGraph& graph,
    int                 pop_size,
    int                 generations,
    std::uint64_t       seed) {
  return memetic_max_cut_analyze(n, graph, pop_size, generations, seed).result;
}
MemeticAnalysis memetic_max_cut_analyze(
    int                 n,
    const MemeticGraph& graph,
    int                 pop_size,
    int                 generations,
    std::uint64_t       seed,
    std::size_t         max_checks) {
  auto listed = validate(n, graph);
  return produce(n, graph, listed, pop_size, generations, seed, max_checks, nullptr, 0);
}
MemeticTrace memetic_max_cut_trace(
    int                 n,
    const MemeticGraph& graph,
    int                 pop_size,
    int                 generations,
    std::uint64_t       seed,
    std::size_t         max_checks,
    std::size_t         limit) {
  auto         listed = validate(n, graph);
  MemeticTrace out;
  out.analysis = produce(n, graph, listed, pop_size, generations, seed, max_checks, &out, limit);
  return out;
}
double memetic_cut_value(int n, const MemeticGraph& graph, const std::vector<int>& side) {
  validate(n, graph);
  validate_side(n, side);
  return objective(graph, side, nullptr);
}
MemeticMaxCutResult
memetic_max_cut_exact(int n, const MemeticGraph& graph, std::size_t max_vertices) {
  validate(n, graph);
  if (n > 63 || (max_vertices && static_cast<std::size_t>(n) > max_vertices))
    throw std::length_error("memetic exact enumeration vertex cap");
  MemeticMaxCutResult out;
  out.value = -std::numeric_limits<double>::infinity();
  std::vector<int> side(n);
  auto             assignments = std::uint64_t{1} << (n ? n - 1 : 0);
  for (std::uint64_t mask = 0; mask < assignments; ++mask) {
    for (int i = 1; i < n; ++i)
      side[i] = (mask >> (i - 1)) & 1;
    double value = objective(graph, side, nullptr);
    if (value > out.value)
      out = {value, side};
  }
  return out;
}
MemeticGraph memetic_demo_graph() {
  MemeticGraph graph(16);
  Rng          rng(999);
  for (int u = 0; u < 16; ++u)
    for (int v = u + 1; v < 16; ++v)
      if (rng.next() % 2) {
        double w = 1 + rng.next() % 9;
        graph[u].push_back({v, w});
        graph[v].push_back({u, w});
      }
  return graph;
}
} // namespace datamunge::algorithms
