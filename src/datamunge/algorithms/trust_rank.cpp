/// @file trust_rank.cpp
/// @brief Canonical producing kernels and native report diagnostics.
#include <datamunge/algorithms/trust_rank.hpp>
namespace datamunge::algorithms {
/// @brief Implement the checked canonical trust_rank_work producer; contracts are declared in the matching header.
TrustRankWork trust_rank_work(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::vector<std::size_t>&                         seeds,
    double                                                  damping,
    double                                                  tolerance,
    std::size_t                                             max_iterations,
    std::size_t                                             max_records) {
  if (!n || n > 1000000 || edges.size() > 10000000 || seeds.empty() || !std::isfinite(damping) ||
      damping < 0 || damping >= 1 || !std::isfinite(tolerance) || tolerance <= 0)
    throw std::invalid_argument("trust_rank: parameters");
  for (auto [u, v] : edges)
    if (u >= n || v >= n)
      throw std::invalid_argument("trust_rank: edge");
  for (auto u : seeds)
    if (u >= n)
      throw std::invalid_argument("trust_rank: seed");
  if (max_records > 1000000 / n || max_iterations > 100000000 / (n + edges.size()))
    throw std::length_error("trust_rank: budget");
  TrustRankWork r;
  r.teleport.resize(n);
  for (auto u : seeds)
    r.teleport[u] += 1.0 / seeds.size();
  r.rank = r.teleport;
  std::vector<std::vector<std::size_t>> out(n);
  for (auto [u, v] : edges)
    out[u].push_back(v);
  std::vector<double> next(n);
  if (max_records)
    r.iterates.push_back(r.rank);
  for (std::size_t it = 0; it < max_iterations; ++it) {
    double dangling = 0;
    for (std::size_t u = 0; u < n; ++u)
      if (out[u].empty())
        dangling += r.rank[u];
    for (std::size_t v = 0; v < n; ++v)
      next[v] = (1 - damping) * r.teleport[v] + damping * dangling * r.teleport[v];
    for (std::size_t u = 0; u < n; ++u)
      if (!out[u].empty())
        for (auto v : out[u]) {
          next[v] += damping * r.rank[u] / out[u].size();
          ++r.edge_visits;
        }
    double error = 0;
    for (std::size_t u = 0; u < n; ++u)
      error += std::abs(next[u] - r.rank[u]);
    r.rank.swap(next);
    ++r.iterations;
    r.deltas.push_back(error);
    if (r.iterates.size() < max_records)
      r.iterates.push_back(r.rank);
    if (error < tolerance) {
      r.converged = true;
      break;
    }
  }
  return r;
}
} // namespace datamunge::algorithms
