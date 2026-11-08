/// @file boruvka.cpp
/// @brief Shared deterministic Boruvka forest kernel with optional phase recording.
#include <datamunge/algorithms/boruvka.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private union-find state and shared traced/untraced traversal.
namespace {
/// @brief Ranked disjoint sets with iterative path compression; owns all vertex state.
struct BoruvkaSets {
  std::vector<std::size_t> parent; ///< Root links, initially the identity.
  std::vector<std::size_t> rank; ///< Union-by-rank levels, initially zero.

  /// @brief Initialise n singleton components.
  /// @param n Number of vertices.
  explicit BoruvkaSets(std::size_t n):parent(n),rank(n) {std::iota(parent.begin(),parent.end(),0);}
  /// @brief Find a valid vertex's root, halving the path.
  /// @param vertex Vertex in range.
  /// @return Current root.
  std::size_t find(std::size_t vertex) {
    while (parent[vertex]!=vertex) {parent[vertex]=parent[parent[vertex]];vertex=parent[vertex];}
    return vertex;
  }
  /// @brief Join distinct sets, retaining the first root when ranks tie.
  /// @param u First valid vertex.
  /// @param v Second valid vertex.
  /// @return True exactly when the number of components decreases.
  bool join(std::size_t u,std::size_t v) {
    u=find(u);v=find(v);if (u==v) return false;
    if (rank[u]<rank[v]) std::swap(u,v);
    parent[v]=u;if (rank[u]==rank[v]) ++rank[u];return true;
  }
  /// @brief Construct canonical minimum-vertex labels without exposing root identities.
  /// @return One owned minimum-vertex component label per original vertex.
  std::vector<std::size_t> labels() {
    const auto n=parent.size();std::vector<std::size_t> minimum(n,n),result(n);
    for (std::size_t v=0;v<n;++v) {const auto root=find(v);minimum[root]=std::min(minimum[root],v);}
    for (std::size_t v=0;v<n;++v) result[v]=minimum[find(v)];
    return result;
  }
};
/// @brief Execute the common forest traversal, allocating phase snapshots only when requested.
/// @tparam Record Whether component/choice/merge records are retained.
/// @param n Number of vertices.
/// @param edges Undirected edges in caller tie order; validated before allocation.
/// @return Forest, counts and optionally populated phases.
/// @throws std::invalid_argument For an invalid endpoint or nonfinite weight.
/// @throws std::overflow_error For an unrepresentable final weight.
template<bool Record>
BoruvkaTrace run_boruvka(std::size_t n,const std::vector<WeightedEdge>& edges) {
  for (const auto& [u,v,w]:edges)
    if (u>=n || v>=n || !std::isfinite(w)) throw std::invalid_argument("boruvka: invalid edge");
  BoruvkaTrace out;out.vertex_count=n;out.component_count=n;BoruvkaSets sets(n);
  long double total=0;
  while (out.component_count>1) {
    BoruvkaPhase phase;
    if constexpr (Record) {
      phase.component_labels=sets.labels();phase.components_before=out.component_count;
      phase.edges_scanned=edges.size();
    }
    std::vector<std::size_t> best(n,edges.size());
    for (std::size_t i=0;i<edges.size();++i) {
      const auto& [u,v,w]=edges[i];const auto a=sets.find(u),b=sets.find(v);if (a==b) continue;
      if (best[a]==edges.size() || w<std::get<2>(edges[best[a]])) best[a]=i;
      if (best[b]==edges.size() || w<std::get<2>(edges[best[b]])) best[b]=i;
    }
    out.edge_scans+=edges.size();const auto previous=out.component_count;
    for (std::size_t root=0;root<n;++root) if (best[root]<edges.size()) {
      const auto i=best[root];const auto& [u,v,w]=edges[i];
      if constexpr (Record) phase.choices.push_back({phase.component_labels[root],i});
      if (sets.join(u,v)) {
        out.forest.edges.push_back(edges[i]);total+=static_cast<long double>(w);--out.component_count;
        if constexpr (Record) phase.accepted_edge_indices.push_back(i);
      }
    }
    if constexpr (Record) {phase.components_after=out.component_count;out.phases.push_back(std::move(phase));}
    if (out.component_count==previous) break;
  }
  if (!std::isfinite(total) || std::fabs(total)>std::numeric_limits<double>::max())
    throw std::overflow_error("boruvka: total weight exceeds finite double range");
  out.forest.total_weight=static_cast<double>(total);out.forest.is_connected=n<=1 || out.component_count==1;
  return out;
}
} // namespace
MinimumSpanningTree boruvka(std::size_t n,const std::vector<WeightedEdge>& edges) {
  return run_boruvka<false>(n,edges).forest;
}
BoruvkaTrace boruvka_trace(std::size_t n,const std::vector<WeightedEdge>& edges) {
  return run_boruvka<true>(n,edges);
}
std::vector<WeightedEdge> dyadic_path_edges(std::size_t n) {
  std::vector<WeightedEdge> edges;if (n>0) edges.reserve(n-1);
  for (std::size_t i=1;i<n;++i) edges.emplace_back(i-1,i,1+std::countr_zero(i));
  return edges;
}
} // namespace datamunge::algorithms
