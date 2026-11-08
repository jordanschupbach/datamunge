/// @file christofides.cpp
/// @brief Shared checked Christofides kernel, metric diagnostics and planar distance preparation.
#include <datamunge/algorithms/christofides.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private validation, checked costs, iterative disjoint sets and shared executing stages.
namespace christofides_detail {
/// @brief Validate every shape/entry before any early metric comparison or construction.
/// @param d Borrowed complete distance matrix.
/// @throws std::invalid_argument If not square, finite and nonnegative.
void validate(const std::vector<std::vector<double>>& d) {
  for(const auto& row:d)if(row.size()!=d.size())throw std::invalid_argument("TSP: distance matrix must be square");
  for(const auto& row:d)for(const auto value:row)if(!std::isfinite(value) || value<0)throw std::invalid_argument("TSP: invalid distance");
}
/// @brief Add one nonnegative finite cost in widened precision, rejecting nonfinite accumulation.
/// @param a Prior finite nonnegative cost.
/// @param b Next finite nonnegative cost.
/// @return Finite widened sum.
/// @throws std::overflow_error If widened accumulation overflows.
long double add(long double a,long double b) {
  const auto sum=a+b;if(!std::isfinite(sum))throw std::overflow_error("Christofides: accumulated cost overflow");return sum;
}
/// @brief Check conversion of a nonnegative widened distance/cost to the public double type.
/// @param value Nonnegative widened cost.
/// @return Finite double cost.
/// @throws std::overflow_error If not finite or above the largest double.
double narrow(long double value) {
  if(!std::isfinite(value) || value>std::numeric_limits<double>::max())throw std::overflow_error("TSP: cost exceeds double range");
  return static_cast<double>(value);
}
/// @brief Increment one named actual count without wrapping.
/// @param value Mutable counter.
/// @throws std::overflow_error If size_t is exhausted.
void count(std::size_t& value) {
  if(value==std::numeric_limits<std::size_t>::max())throw std::overflow_error("Christofides: counter overflow");++value;
}
/// @brief Iterative path-halving disjoint sets with union by rank; no recursive graph traversal.
struct Sets {
  std::vector<std::size_t> parent; ///< Current roots/links.
  std::vector<std::size_t> rank; ///< Upper bounds on rooted height.

  /// @brief Initialise n singleton components.
  /// @param n Vertex count.
  explicit Sets(std::size_t n):parent(n),rank(n){std::iota(parent.begin(),parent.end(),0);}
  /// @brief Find an in-range vertex's root with iterative path halving.
  /// @param v Valid vertex.
  /// @return Current root.
  std::size_t find(std::size_t v){while(parent[v]!=v){parent[v]=parent[parent[v]];v=parent[v];}return v;}
  /// @brief Join two components, keeping the first root on rank ties.
  /// @param u Valid first vertex.
  /// @param v Valid second vertex.
  /// @return True iff previously distinct roots were joined.
  bool join(std::size_t u,std::size_t v){u=find(u);v=find(v);if(u==v)return false;if(rank[u]<rank[v])std::swap(u,v);parent[v]=u;if(rank[u]==rank[v])++rank[u];return true;}
};
/// @brief Actual shared stage execution with optional retention, never reconstructing a second run.
/// @tparam Record Whether to retain stage/identity records in the returned trace.
/// @param d Borrowed finite nonnegative square costs, validated here.
/// @param start First/last vertex, ignored for empty input.
/// @param options Odd-set and allocated-table caps.
/// @return Actual result, stage costs/work and optionally owned stage records.
/// @throws std::invalid_argument If input/start/options or the odd-set size is invalid.
/// @throws std::length_error If storage arithmetic, table cap or container capacity is exceeded.
/// @throws std::overflow_error If a cost/count cannot be represented.
/// @throws std::bad_alloc If allocation fails.
template<bool Record>
ChristofidesTrace run(const std::vector<std::vector<double>>& d,std::size_t start,ChristofidesOptions options) {
  validate(d);if(options.max_odd_vertices>22)throw std::invalid_argument("Christofides: odd cap must be at most 22");
  ChristofidesTrace out;const auto n=d.size();if(!n)return out;
  if(start>=n)throw std::invalid_argument("Christofides: start outside matrix");
  if(n>std::numeric_limits<std::size_t>::max()/n)throw std::length_error("Christofides: input count overflow");
  out.work.input_entries=n*n;
  std::size_t a=n,b=n-1;if(a%2==0)a/=2;else b/=2;
  if(b && a>std::numeric_limits<std::size_t>::max()/b)throw std::length_error("Christofides: edge count overflow");
  std::vector<WeightedEdge> edges;edges.reserve(a*b);
  for(std::size_t u=0;u<n;++u)for(std::size_t v=u+1;v<n;++v)edges.emplace_back(u,v,d[u][v]);
  out.work.complete_edges=edges.size();
  /// @brief Sort by actual weight, then ascending endpoints for explicit tied-MST behavior.
  /// @param lhs First validated upper-triangle edge.
  /// @param rhs Second validated upper-triangle edge.
  /// @return True iff lhs precedes rhs by (weight,u,v).
  const auto by_weight=[](const WeightedEdge& lhs,const WeightedEdge& rhs) {
    return std::tuple{std::get<2>(lhs),std::get<0>(lhs),std::get<1>(lhs)}<std::tuple{std::get<2>(rhs),std::get<0>(rhs),std::get<1>(rhs)};
  };
  std::sort(edges.begin(),edges.end(),by_weight);
  Sets sets(n);std::vector<WeightedEdge> tree;tree.reserve(n-1);std::vector<std::size_t> degrees(n);
  for(const auto& edge:edges) {
    if(tree.size()==n-1)break;
    count(out.work.kruskal_candidates);const auto [u,v,w]=edge;
    if(sets.join(u,v)){tree.push_back(edge);++degrees[u];++degrees[v];out.tree_cost=add(out.tree_cost,w);}
  }
  std::vector<std::size_t> odd;
  for(std::size_t v=0;v<n;++v)if(degrees[v]%2)odd.push_back(v);
  if(odd.size()>options.max_odd_vertices)throw std::invalid_argument("Christofides: odd vertices exceed exact matching cap");
  const std::size_t entries=std::size_t{1}<<odd.size(),full=entries-1;
  if(options.max_matching_states && entries>options.max_matching_states)throw std::length_error("Christofides: matching table cap exceeded");
  out.work.matching_table_entries=entries;
  std::vector<long double> dp(entries,std::numeric_limits<long double>::infinity());
  std::vector<std::uint8_t> choice(entries,0);dp[0]=0;
  for(std::size_t mask=1;mask<=full;++mask) {
    if(std::popcount(mask)%2)continue;
    count(out.work.matching_masks);const auto i=std::countr_zero(mask);
    for(std::size_t j=i+1;j<odd.size();++j)if(mask&(std::size_t{1}<<j)) {
      count(out.work.matching_candidates);const auto rest=mask^(std::size_t{1}<<i)^(std::size_t{1}<<j);
      const auto candidate=add(dp[rest],d[odd[i]][odd[j]]);
      if(candidate<dp[mask]){dp[mask]=candidate;choice[mask]=static_cast<std::uint8_t>(j);}
    }
  }
  std::vector<WeightedEdge> matching;matching.reserve(odd.size()/2);
  for(auto mask=full;mask;) {
    const auto i=std::countr_zero(mask);const auto j=choice[mask];
    matching.emplace_back(odd[i],odd[j],d[odd[i]][odd[j]]);out.matching_cost=add(out.matching_cost,d[odd[i]][odd[j]]);
    mask^=(std::size_t{1}<<i)|(std::size_t{1}<<j);
  }
  auto multi=tree;multi.insert(multi.end(),matching.begin(),matching.end());
  std::vector<std::vector<std::pair<std::size_t,std::size_t>>> adjacency(n);
  for(std::size_t id=0;id<multi.size();++id) {
    const auto [u,v,w]=multi[id];adjacency[u].push_back({v,id});adjacency[v].push_back({u,id});
  }
  constexpr auto no_edge=std::numeric_limits<std::size_t>::max();
  std::vector<bool> used(multi.size());std::vector<std::pair<std::size_t,std::size_t>> stack{{start,no_edge}};
  std::vector<std::size_t> circuit,edge_ids;
  while(!stack.empty()) {
    const auto u=stack.back().first;
    while(!adjacency[u].empty() && used[adjacency[u].back().second])adjacency[u].pop_back();
    if(adjacency[u].empty()) {
      circuit.push_back(u);if constexpr(Record)if(stack.back().second!=no_edge)edge_ids.push_back(stack.back().second);stack.pop_back();
    } else {
      const auto [v,id]=adjacency[u].back();adjacency[u].pop_back();used[id]=true;count(out.work.euler_edge_visits);stack.push_back({v,id});
    }
  }
  std::reverse(circuit.begin(),circuit.end());if constexpr(Record)std::reverse(edge_ids.begin(),edge_ids.end());
  std::vector<bool> seen(n);long double cost=0;
  for(std::size_t position=0;position<circuit.size();++position) {
    count(out.work.shortcut_vertices);const auto v=circuit[position];
    if(position)out.euler_cost=add(out.euler_cost,d[circuit[position-1]][v]);
    if(!seen[v]) {
      seen[v]=true;if constexpr(Record)out.kept_positions.push_back(position);
      if(!out.result.tour.empty())cost=add(cost,d[out.result.tour.back()][v]);
      out.result.tour.push_back(v);
    }
  }
  cost=add(cost,d[out.result.tour.back()][start]);out.result.tour.push_back(start);out.result.cost=narrow(cost);
  if constexpr(Record) {out.tree=std::move(tree);out.matching=std::move(matching);out.odd_vertices=std::move(odd);out.euler_vertices=std::move(circuit);out.euler_edge_ids=std::move(edge_ids);}
  return out;
}
} // namespace christofides_detail
bool is_tsp_metric(const std::vector<std::vector<double>>& distance,double tolerance) {
  christofides_detail::validate(distance);
  if(!std::isfinite(tolerance) || tolerance<0)throw std::invalid_argument("TSP: invalid metric tolerance");
  for(std::size_t i=0;i<distance.size();++i) {
    if(distance[i][i]>tolerance)return false;
    for(std::size_t j=0;j<distance.size();++j) {
      if(std::fabs(static_cast<long double>(distance[i][j])-distance[j][i])>tolerance)return false;
      for(std::size_t k=0;k<distance.size();++k)
        if(static_cast<long double>(distance[i][j])-distance[i][k]-distance[k][j]>tolerance)return false;
    }
  }
  return true;
}
std::vector<std::vector<double>> euclidean_tsp_distances(const std::vector<std::pair<double,double>>& points) {
  for(const auto [x,y]:points)if(!std::isfinite(x) || !std::isfinite(y))throw std::invalid_argument("TSP: nonfinite coordinate");
  std::vector<std::vector<double>> d(points.size(),std::vector<double>(points.size()));
  for(std::size_t i=0;i<points.size();++i)for(std::size_t j=i+1;j<points.size();++j) {
    const auto dx=static_cast<long double>(points[i].first)-points[j].first;
    const auto dy=static_cast<long double>(points[i].second)-points[j].second;
    d[i][j]=d[j][i]=christofides_detail::narrow(std::hypot(dx,dy));
  }
  return d;
}
std::vector<std::vector<double>> hub_tsp_distances(std::size_t leaves,double spoke,double rim) {
  if(!std::isfinite(spoke) || spoke<0 || !std::isfinite(rim) || rim<0)throw std::invalid_argument("TSP: invalid hub/leaf cost");
  if(leaves==std::numeric_limits<std::size_t>::max())throw std::length_error("TSP: hub vertex count overflow");
  const auto n=leaves+1;std::vector<std::vector<double>> d(n,std::vector<double>(n,rim));
  for(std::size_t v=0;v<n;++v){d[v][v]=0;if(v)d[0][v]=d[v][0]=spoke;}
  return d;
}
TourResult christofides_tsp(const std::vector<std::vector<double>>& distance,std::size_t start) {
  return christofides_detail::run<false>(distance,start,{}).result;
}
TourResult christofides_tsp_with_options(const std::vector<std::vector<double>>& distance,std::size_t start,ChristofidesOptions options) {
  return christofides_detail::run<false>(distance,start,options).result;
}
ChristofidesTrace christofides_trace(const std::vector<std::vector<double>>& distance,std::size_t start,ChristofidesOptions options) {
  return christofides_detail::run<true>(distance,start,options);
}
} // namespace datamunge::algorithms
