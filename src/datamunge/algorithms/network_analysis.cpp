/// @file network_analysis.cpp
/// @brief PageRank, TrustRank, HITS and Girvan–Newman implementations.

#include <datamunge/algorithms/network_analysis.hpp>

#include <algorithm>
#include <cmath>
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
  for (const auto& [u,v] : edges)
    if (u >= n || v >= n) throw std::invalid_argument("network analysis: endpoint out of range");
}
/// Scales values to unit Euclidean norm (if non-zero) and returns the old norm.
double normalize(std::vector<double>& values) {
  double norm = 0; for (double x : values) norm += x * x; norm = std::sqrt(norm);
  if (norm > 0) for (double& x : values) x /= norm;
  return norm;
}

} // namespace

/// Power iteration with uniform redistribution of dangling mass.
std::vector<double> page_rank(std::size_t n, const std::vector<Edge>& edges,
                              double damping, double tolerance, std::size_t max_iterations) {
  validate(n, edges);
  if (!(damping >= 0 && damping < 1) || tolerance <= 0)
    throw std::invalid_argument("page_rank: invalid numerical parameter");
  if (n == 0) return {};
  std::vector<std::vector<std::size_t>> out(n);
  for (auto [u,v]:edges) out[u].push_back(v);
  std::vector<double> rank(n,1.0/n), next(n);
  for (std::size_t iter=0;iter<max_iterations;++iter) {
    double dangling=0; for(std::size_t u=0;u<n;++u) if(out[u].empty()) dangling+=rank[u];
    std::fill(next.begin(),next.end(),(1-damping)/n+damping*dangling/n);
    for(std::size_t u=0;u<n;++u) if(!out[u].empty())
      for(auto v:out[u]) next[v]+=damping*rank[u]/out[u].size();
    double error=0; for(std::size_t i=0;i<n;++i) error+=std::abs(next[i]-rank[i]);
    rank.swap(next); if(error<tolerance) break;
  }
  return rank;
}

/// Assembles \f$I-\alpha P^\top\f$ and its transpose in CSR form.
PageRankSystem page_rank_system(std::size_t n, const std::vector<Edge>& edges, double damping) {
  validate(n, edges);
  if (!(damping >= 0 && damping < 1)) throw std::invalid_argument("page_rank_system: damping must be in [0, 1)");
  std::vector<double> out_degree(n, 0.0);
  for (auto [u, v] : edges) out_degree[u] += 1.0;
  for (double d : out_degree)
    if (d == 0.0) throw std::invalid_argument("page_rank_system: every vertex needs an out-edge");
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
  return {linalg::to_csr(a), linalg::to_csr(at), std::vector<double>(n, (1 - damping) / static_cast<double>(n))};
}

/// Power iteration with jumps restricted to the seed distribution.
std::vector<double> trust_rank(std::size_t n, const std::vector<Edge>& edges,
                               const std::vector<std::size_t>& seeds, double damping,
                               double tolerance, std::size_t max_iterations) {
  validate(n, edges);
  if (seeds.empty() || !(damping >= 0 && damping < 1) || tolerance <= 0)
    throw std::invalid_argument("trust_rank: invalid parameter");
  std::vector<double> teleport(n);
  for (auto seed : seeds) {
    if (seed >= n) throw std::invalid_argument("trust_rank: seed out of range");
    teleport[seed] += 1.0 / seeds.size();
  }
  std::vector<std::vector<std::size_t>> out(n);
  for (auto [u,v] : edges) out[u].push_back(v);
  std::vector<double> rank = teleport, next(n);
  for (std::size_t iter=0; iter<max_iterations; ++iter) {
    double dangling=0; for(std::size_t u=0;u<n;++u) if(out[u].empty()) dangling+=rank[u];
    for(std::size_t v=0;v<n;++v) next[v]=(1-damping)*teleport[v]+damping*dangling*teleport[v];
    for(std::size_t u=0;u<n;++u) if(!out[u].empty())
      for(auto v:out[u]) next[v]+=damping*rank[u]/out[u].size();
    double error=0;for(std::size_t i=0;i<n;++i)error+=std::abs(next[i]-rank[i]);
    rank.swap(next);if(error<tolerance)break;
  }
  return rank;
}

/// Alternates authority and hub updates, normalising each.
HITSResult hits(std::size_t n, const std::vector<Edge>& edges,
                double tolerance, std::size_t max_iterations) {
  validate(n,edges); if(tolerance<=0) throw std::invalid_argument("hits: tolerance must be positive");
  HITSResult result; result.hubs.assign(n,n?1/std::sqrt(static_cast<double>(n)):0);
  result.authorities=result.hubs;
  for(;result.iterations<max_iterations;++result.iterations) {
    std::vector<double> a(n),h(n);
    for(auto [u,v]:edges) a[v]+=result.hubs[u];
    normalize(a); for(auto [u,v]:edges) h[u]+=a[v]; normalize(h);
    double error=0; for(std::size_t i=0;i<n;++i)
      error=std::max(error,std::max(std::abs(a[i]-result.authorities[i]),std::abs(h[i]-result.hubs[i])));
    result.authorities.swap(a); result.hubs.swap(h);
    if(error<tolerance) {++result.iterations;break;}
  }
  return result;
}



} // namespace datamunge::algorithms
