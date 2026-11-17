/// @file grasp.cpp
/// @brief Shared validated GRASP execution, checked work and actual online histories.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/grasp_analysis.hpp>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

namespace datamunge::algorithms {
/// @brief Private input validation, quota admission and producing GRASP phases.
namespace grasp_detail {
/// @brief Checked logical counter addition.
/// @param a Current value.
/// @param b Increment.
/// @return Nonwrapping sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("GRASP count overflow");
  return a + b;
}
/// @brief Reject nonfinite arithmetic before it enters a candidate or result.
/// @param x Computed double.
/// @return The unchanged finite value.
/// @throws std::overflow_error If x is nonfinite.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("GRASP arithmetic became nonfinite");
  return x;
}
/// @brief Validate basic graph/parameters before caps, then reciprocal weights before search.
/// @param n Nonnegative matching row count.
/// @param adj Borrowed weighted adjacency.
/// @param iterations Positive restart count.
/// @param alpha Finite RCL fraction in [0,1].
/// @param o Complete input limits.
/// @throws std::invalid_argument On graph/parameters or reciprocal mismatch.
/// @throws std::length_error On input cap, count or container exhaustion.
/// @throws std::bad_alloc On reciprocal-map allocation failure.
void validate(
    int                   n,
    const GraspAdjacency& adj,
    int                   iterations,
    double                alpha,
    const GraspOptions&   o) {
  if (n < 0 || adj.size() != static_cast<std::size_t>(n) || iterations <= 0 ||
      !std::isfinite(alpha) || alpha < 0 || alpha > 1)
    throw std::invalid_argument("GRASP shape/iterations/alpha invalid");
  std::size_t entries = 0;
  for (int v = 0; v < n; ++v) {
    entries = add(entries, adj[v].size());
    for (auto [u, w] : adj[v])
      if (u < 0 || u >= n || u == v || !std::isfinite(w))
        throw std::invalid_argument("GRASP requires finite loopless endpoints");
  }
  if ((o.max_vertices && static_cast<std::size_t>(n) > o.max_vertices) ||
      (o.max_adjacency_entries && entries > o.max_adjacency_entries))
    throw std::length_error("GRASP complete input cap");
  std::map<std::tuple<int, int, double>, std::size_t> multiplicity;
  for (int v = 0; v < n; ++v)
    for (auto [u, w] : adj[v]) {
      auto& c = multiplicity[{v, u, w}];
      c       = add(c, 1);
    }
  for (const auto& [key, count] : multiplicity) {
    auto [v, u, w] = key;
    auto reverse   = multiplicity.find({u, v, w});
    if (reverse == multiplicity.end() || reverse->second != count)
      throw std::invalid_argument("GRASP reciprocal weighted multiplicities differ");
  }
}
/// @brief Admit complete logical history entries before producing work or snapshots.
/// @param t Optional borrowed history under construction; null disables capture.
/// @param count Complete next logical entry requirement.
/// @param cap Optional cap, zero disables it.
/// @throws std::length_error On cap or count overflow.
void admit(GraspTrace* t, std::size_t count, std::size_t cap) {
  if (!t)
    return;
  auto total = add(t->captured_entries, count);
  if (cap && total > cap)
    throw std::length_error("GRASP trace-entry quota");
  t->captured_entries = total;
}
/// @brief Compute the original final cut in adjacency order and count actual inspections.
/// @param adj Valid borrowed graph.
/// @param side Actual complete partition.
/// @param w Producing selected work.
/// @return Sequential u<v crossing-weight sum.
/// @throws std::overflow_error On nonfinite sum.
/// @throws std::length_error On count overflow.
double value(const GraspAdjacency& adj, const std::vector<int>& side, GraspWork& w) {
  w.objective_calls = add(w.objective_calls, 1);
  double val        = 0;
  for (std::size_t u = 0; u < adj.size(); ++u)
    for (auto [v, weight] : adj[u]) {
      w.objective_neighbors = add(w.objective_neighbors, 1);
      if (u < static_cast<std::size_t>(v) && side[u] != side[v])
        val += weight;
    }
  return finite(val);
}
/// @brief Compute one original sequential flip gain and count actual inspected entries.
/// @param adj Valid borrowed adjacency.
/// @param side Complete current partition.
/// @param v Valid tested vertex ID.
/// @param w Selected producing work.
/// @return Actual finite delta from incident weights.
/// @throws std::overflow_error On nonfinite sum.
/// @throws std::length_error On checked work exhaustion.
double delta(const GraspAdjacency& adj, const std::vector<int>& side, int v, GraspWork& w) {
  w.flip_tests = add(w.flip_tests, 1);
  double gain  = 0;
  for (auto [u, weight] : adj[v]) {
    w.local_neighbors = add(w.local_neighbors, 1);
    gain += (side[u] != side[v]) ? -weight : weight;
  }
  return finite(gain);
}
/// @brief Execute original construction/strict-improvement arithmetic once with optional actual
/// capture.
/// @param n Validated by this kernel.
/// @param adj Borrowed weighted graph, never retained or mutated.
/// @param iterations Positive restart count.
/// @param alpha Finite RCL fraction.
/// @param seed Legacy RNG seed.
/// @param o Resource options.
/// @param trace Optional borrowed actual history sink, initially empty.
/// @return Owned actual best output and selected work.
/// @throws std::invalid_argument On graph/parameters.
/// @throws std::length_error On input/pass/capture/counter/container limits.
/// @throws std::overflow_error On nonfinite arithmetic.
/// @throws std::bad_alloc On allocation failure.
GraspRun execute(
    int                   n,
    const GraspAdjacency& adj,
    int                   iterations,
    double                alpha,
    std::uint64_t         seed,
    GraspOptions          o,
    GraspTrace*           trace) {
  validate(n, adj, iterations, alpha, o);
  detail::GraspRng rng(seed);
  GraspRun         run;
  auto&            w = run.work;
  run.result.value   = -std::numeric_limits<double>::infinity();
  for (int iter = 0; iter < iterations; ++iter) {
    if (trace)
      admit(
          trace,
          add(1, add(static_cast<std::size_t>(n), static_cast<std::size_t>(n))),
          o.max_trace_entries);
    GraspRestart restart;
    restart.restart = iter;
    std::vector<int>  side(n, -1);
    std::vector<char> assigned(n, 0);
    int               placed = 0;
    while (placed < n) {
      if (trace)
        admit(trace, add(1, static_cast<std::size_t>(n - placed)), o.max_trace_entries);
      GraspConstruction step;
      step.restart      = iter;
      step.step         = placed;
      double best_gain  = -std::numeric_limits<double>::infinity(),
             worst_gain = std::numeric_limits<double>::infinity();
      std::vector<std::pair<int, int>> cand;
      std::vector<double>              gains;
      for (int v = 0; v < n; ++v) {
        if (assigned[v])
          continue;
        w.candidate_scores = add(w.candidate_scores, 1);
        double g0 = 0, g1 = 0;
        for (auto [u, weight] : adj[v]) {
          w.construction_neighbors = add(w.construction_neighbors, 1);
          if (assigned[u]) {
            if (side[u] == 1)
              g0 += weight;
            else
              g1 += weight;
          }
        }
        finite(g0);
        finite(g1);
        const int    s = g0 >= g1 ? 0 : 1;
        const double g = g0 >= g1 ? g0 : g1;
        cand.emplace_back(v, s);
        gains.push_back(g);
        best_gain  = std::max(best_gain, g);
        worst_gain = std::min(worst_gain, g);
        if (trace)
          step.candidates.push_back({v, s, g0, g1, g, false});
      }
      const double             range = finite(best_gain - worst_gain);
      const double             thr   = finite(best_gain - alpha * range);
      std::vector<std::size_t> rcl;
      for (std::size_t i = 0; i < cand.size(); ++i) {
        w.rcl_tests       = add(w.rcl_tests, 1);
        const bool member = gains[i] >= thr - 1e-12;
        if (member)
          rcl.push_back(i);
        if (trace)
          step.candidates[i].in_rcl = member;
      }
      if (trace) {
        step.best_gain  = best_gain;
        step.worst_gain = worst_gain;
        step.threshold  = thr;
        step.rng_before = rng.s;
      }
      w.rng_words            = add(w.rng_words, 1);
      const std::size_t pick = rcl[rng.below(rcl.size())];
      if (trace) {
        step.rng_after = rng.s;
        step.pick      = pick;
        trace->construction.push_back(std::move(step));
      }
      side[cand[pick].first]     = cand[pick].second;
      assigned[cand[pick].first] = 1;
      ++placed;
    }
    if (trace)
      restart.constructed_side = side;
    bool        improved = true;
    std::size_t pass     = 0;
    while (improved) {
      if (o.max_local_passes && pass >= o.max_local_passes)
        throw std::length_error("GRASP local-pass cap");
      pass           = add(pass, 1);
      w.local_passes = add(w.local_passes, 1);
      improved       = false;
      for (int v = 0; v < n; ++v) {
        admit(trace, 1, o.max_trace_entries);
        const double gain     = delta(adj, side, v, w);
        const bool   accepted = gain > 1e-12;
        if (trace)
          trace->flips.push_back({iter, pass, v, side[v], gain, accepted});
        if (accepted) {
          w.accepted_flips = add(w.accepted_flips, 1);
          side[v] ^= 1;
          improved = true;
        }
      }
    }
    const double val     = value(adj, side, w);
    const bool   changed = val > run.result.value;
    if (changed) {
      run.result.value = val;
      run.result.side  = side;
    }
    w.restarts = add(w.restarts, 1);
    if (trace) {
      restart.final_side    = side;
      restart.final_value   = val;
      restart.best_value    = run.result.value;
      restart.improved_best = changed;
      trace->restarts.push_back(std::move(restart));
    }
  }
  return run;
}
} // namespace grasp_detail
GraspMaxCutResult
grasp_max_cut(int n, const GraspAdjacency& adj, int iterations, double alpha, std::uint64_t seed) {
  GraspOptions o;
  o.max_vertices          = 0;
  o.max_adjacency_entries = 0;
  o.max_trace_entries     = 0;
  return grasp_detail::execute(n, adj, iterations, alpha, seed, o, nullptr).result;
}
GraspRun grasp_max_cut_run(
    int                   n,
    const GraspAdjacency& adj,
    int                   iterations,
    double                alpha,
    std::uint64_t         seed,
    GraspOptions          options) {
  return grasp_detail::execute(n, adj, iterations, alpha, seed, options, nullptr);
}
GraspTrace grasp_max_cut_trace(
    int                   n,
    const GraspAdjacency& adj,
    int                   iterations,
    double                alpha,
    std::uint64_t         seed,
    GraspOptions          options) {
  GraspTrace t;
  t.run = grasp_detail::execute(n, adj, iterations, alpha, seed, options, &t);
  return t;
}
GraspAdjacency grasp_demo_graph() {
  GraspAdjacency   adj(14);
  detail::GraspRng rng(12345);
  for (int i = 0; i < 14; ++i)
    for (int j = i + 1; j < 14; ++j)
      if (rng.next() % 2) {
        const double w = 1 + rng.next() % 9;
        adj[i].push_back({j, w});
        adj[j].push_back({i, w});
      }
  return adj;
}
GraspCutCheck grasp_cut_check(const GraspAdjacency& adj, const std::vector<int>& side) {
  if (adj.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("MAX-CUT diagnostic vertex limit");
  if (side.size() != adj.size())
    throw std::invalid_argument("MAX-CUT side shape");
  for (int s : side)
    if (s != 0 && s != 1)
      throw std::invalid_argument("MAX-CUT side must be zero or one");
  GraspOptions o;
  o.max_vertices          = 0;
  o.max_adjacency_entries = 0;
  grasp_detail::validate(static_cast<int>(adj.size()), adj, 1, .3, o);
  GraspCutCheck result;
  GraspWork     work;
  result.value           = grasp_detail::value(adj, side, work);
  result.tolerance_local = true;
  for (std::size_t v = 0; v < adj.size(); ++v) {
    const double gain = grasp_detail::delta(adj, side, static_cast<int>(v), work);
    result.flip_gains.push_back(gain);
    if (gain > 1e-12)
      result.tolerance_local = false;
  }
  return result;
}
GraspExactCut grasp_exact_max_cut(const GraspAdjacency& adj, std::size_t max_partitions) {
  if (adj.size() > 24)
    throw std::length_error("exact MAX-CUT hard vertex bound");
  GraspOptions o;
  o.max_vertices          = 0;
  o.max_adjacency_entries = 0;
  grasp_detail::validate(static_cast<int>(adj.size()), adj, 1, .3, o);
  const std::size_t count = std::size_t{1} << (adj.empty() ? 0 : adj.size() - 1);
  if (max_partitions && count > max_partitions)
    throw std::length_error("exact MAX-CUT partition cap");
  GraspExactCut result;
  result.result.value = -std::numeric_limits<double>::infinity();
  std::vector<int> side(adj.size(), 0);
  GraspWork        work;
  for (std::size_t mask = 0; mask < count; ++mask) {
    for (std::size_t v = 1; v < adj.size(); ++v)
      side[v] = (mask >> (v - 1)) & 1;
    const double value = grasp_detail::value(adj, side, work);
    ++result.partitions;
    if (value > result.result.value) {
      result.result.value = value;
      result.result.side  = side;
    }
  }
  return result;
}
} // namespace datamunge::algorithms
