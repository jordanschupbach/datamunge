/// @file network_analysis.cpp
/// @brief Canonical producing kernels and native report diagnostics.
#include <datamunge/algorithms/trust_rank.hpp>
/// @file network_analysis.cpp
/// @brief PageRank and TrustRank implementations; HITS has its own shared kernel.

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/network_analysis.hpp>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Helpers for network_analysis.cpp.
namespace {
/// A directed or undirected edge (u, v).
using Edge = std::pair<std::size_t, std::size_t>;
/// Throws if an edge endpoint is not below n.
void validate(std::size_t n, const std::vector<Edge>& edges) {
  for (const auto& [u, v] : edges)
    if (u >= n || v >= n)
      throw std::invalid_argument("network analysis: endpoint out of range");
}
} // namespace

/// Assembles \f$I-\alpha P^\top\f$ and its transpose in CSR form.
PageRankSystem page_rank_system(std::size_t n, const std::vector<Edge>& edges, double damping) {
  validate(n, edges);
  if (!(damping >= 0 && damping < 1))
    throw std::invalid_argument("page_rank_system: damping must be in [0, 1)");
  std::vector<double> out_degree(n, 0.0);
  for (auto [u, v] : edges)
    out_degree[u] += 1.0;
  for (double d : out_degree)
    if (d == 0.0)
      throw std::invalid_argument("page_rank_system: every vertex needs an out-edge");
  linalg::SparseCOO<double> a(n, n), at(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    a.set(i, i, 1.0);
    at.set(i, i, 1.0);
  }
  for (auto [u, v] : edges) {
    a.set(v, u, -damping / out_degree[u]);
    at.set(u, v, -damping / out_degree[u]);
  }
  a.compress();
  at.compress();
  return {
      linalg::to_csr(a),
      linalg::to_csr(at),
      std::vector<double>(n, (1 - damping) / static_cast<double>(n))};
}

/// Power iteration with jumps restricted to the seed distribution.
std::vector<double> trust_rank(
    std::size_t                     n,
    const std::vector<Edge>&        edges,
    const std::vector<std::size_t>& seeds,
    double                          damping,
    double                          tolerance,
    std::size_t                     max_iterations) {
  return trust_rank_work(n, edges, seeds, damping, tolerance, max_iterations).rank;
}

} // namespace datamunge::algorithms
