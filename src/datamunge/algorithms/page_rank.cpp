/// @file page_rank.cpp
/// @brief One checked original sparse PageRank power loop with actual update and edge-transfer capture.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/page_rank.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked count accounting and same-producer iteration path.
namespace {
/// @brief Add complete/actual counts without size_t wraparound.
/// @param a Existing count.
/// @param b Additional count.
/// @return Checked sum.
/// @throws std::length_error If sum overflows.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("PageRank: count geometry");
  return a + b;
}
/// @brief Execute actual historical uniform-teleport update order with optional trace.
/// @param n Complete original vertex count.
/// @param edges Complete original directed occurrences.
/// @param options Complete numerical controls/history policy.
/// @param history Optional owned actual-history sink.
/// @return Actual finite final scores/work/status.
/// @throws std::invalid_argument If complete endpoint/control validation fails.
/// @throws std::length_error If graph/counter/history geometry or selected quota fails.
/// @throws std::overflow_error If actual arithmetic becomes nonfinite.
PageRankRun run(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    PageRankOptions                                         options,
    PageRankTrace*                                          history) {
  for (auto [u, v] : edges)
    if (u >= n || v >= n)
      throw std::invalid_argument("PageRank: endpoint out of range");
  if (!std::isfinite(options.damping) || options.damping < 0 || options.damping >= 1 ||
      !std::isfinite(options.tolerance) || options.tolerance <= 0)
    throw std::invalid_argument("PageRank: complete finite numerical controls required");
  if (n > std::vector<double>().max_size() ||
      n > std::vector<std::vector<std::size_t>>().max_size() || n > 9007199254740991ULL ||
      edges.size() > 9007199254740991ULL)
    throw std::length_error("PageRank: complete graph geometry");
  /// @brief Admit actual future logical elements before copying/appending.
  /// @param units Complete next retained record contribution.
  /// @throws std::length_error If count geometry or selected policy is exceeded.
  auto charge = [&](std::size_t units) {
    if (!history)
      return;
    auto next = add(history->captured_entries, units);
    if (options.max_trace_entries && next > options.max_trace_entries)
      throw std::length_error("PageRank: actual capture quota");
    history->captured_entries = next;
  };
  if (history) {
    charge(add(4, add(add(edges.size(), edges.size()), add(n, n))));
    history->vertices = n;
    history->edges    = edges;
    history->options  = options;
  }
  PageRankRun r;
  if (!n) {
    r.converged = true;
    return r;
  }
  std::vector<std::vector<std::size_t>> out(n);
  for (auto [u, v] : edges)
    out[u].push_back(v);
  std::vector<double> rank(n, 1.0 / n), next(n);
  if (history)
    history->initial = rank;
  for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
    PageRankIteration step;
    step.ordinal = add(iteration, 1);
    if (history) {
      charge(add(1, add(n, n)));
      step.before         = rank;
      step.transfer_begin = history->transfers.size();
    }
    double dangling = 0;
    for (std::size_t u = 0; u < n; ++u) {
      r.dangling_checks = add(r.dangling_checks, 1);
      if (out[u].empty())
        dangling += rank[u];
    }
    const double base = (1 - options.damping) / n + options.damping * dangling / n;
    std::fill(next.begin(), next.end(), base);
    for (std::size_t u = 0; u < n; ++u)
      if (!out[u].empty())
        for (auto v : out[u]) {
          PageRankTransfer t;
          t.iteration          = step.ordinal;
          t.source             = u;
          t.destination        = v;
          t.out_degree         = out[u].size();
          t.source_rank        = rank[u];
          t.contribution       = options.damping * rank[u] / out[u].size();
          t.destination_before = next[v];
          t.destination_after  = next[v] + t.contribution;
          if (!std::isfinite(t.destination_after))
            throw std::overflow_error("PageRank: nonfinite transfer");
          if (history) {
            charge(1);
            history->transfers.push_back(t);
          }
          next[v]          = t.destination_after;
          r.edge_transfers = add(r.edge_transfers, 1);
        }
    double error = 0;
    for (std::size_t i = 0; i < n; ++i)
      error += std::abs(next[i] - rank[i]);
    if (!std::isfinite(error) || !std::isfinite(base) || !std::isfinite(dangling))
      throw std::overflow_error("PageRank: nonfinite iteration");
    rank.swap(next);
    r.iterations        = step.ordinal;
    r.last_change       = error;
    r.contraction_bound = options.damping / (1 - options.damping) * error;
    if (history) {
      step.after         = rank;
      step.dangling_mass = dangling;
      step.uniform_base  = base;
      step.l1_change     = error;
      step.transfer_end  = history->transfers.size();
      history->iterations.push_back(std::move(step));
    }
    if (error < options.tolerance) {
      r.converged = true;
      break;
    }
  }
  r.scores = std::move(rank);
  return r;
}
} // namespace
std::vector<double> page_rank(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    double                                                  damping,
    double                                                  tolerance,
    std::size_t                                             max_iterations) {
  return run(n, edges, {damping, tolerance, max_iterations, 0}, nullptr).scores;
}
PageRankRun page_rank_run(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    PageRankOptions                                         options) {
  return run(n, edges, options, nullptr);
}
PageRankTrace page_rank_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    PageRankOptions                                         options) {
  PageRankTrace t;
  t.run = run(n, edges, options, &t);
  return t;
}
std::vector<std::pair<std::size_t, std::size_t>> page_rank_demo_edges() {
  return {{0, 2}, {1, 2}, {2, 2}};
}
} // namespace datamunge::algorithms
