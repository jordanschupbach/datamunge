/// @file
/// @brief Shared checked Kahn/longest-path kernel, actual records and safe local witness recovery.
#include <datamunge/algorithms/dag_longest_path.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Private complete validation, checked arithmetic and actual graph operations.
namespace dag_longest_detail {
/// @brief Check all directed edge endpoints/weights and then the source before allocation.
/// @param n Vertex count.
/// @param edges Borrowed raw directed edges.
/// @param source Requested source id.
/// @throws std::invalid_argument Invalid endpoint/source or nonfinite weight.
void validate(std::size_t n,const std::vector<WeightedEdge>& edges,std::size_t source) {
  for(const auto& e:edges) if(std::get<0>(e)>=n || std::get<1>(e)>=n || !std::isfinite(std::get<2>(e))) throw std::invalid_argument("DAG longest paths: invalid edge");
  if(source>=n) throw std::invalid_argument("DAG longest paths: source out of range");
}
/// @brief Increment a recorded actual operation count without wrapping.
/// @param count Actual counter, changed by one.
/// @throws std::overflow_error Count cannot be incremented.
void tick(std::size_t& count) {
  if(count==std::numeric_limits<std::size_t>::max()) throw std::overflow_error("DAG longest paths: work overflow");++count;
}
/// @brief Add two finite path operands in ordinary double arithmetic, rejecting infinity.
/// @param a Finalized finite source distance.
/// @param b Finite edge weight.
/// @return Rounded finite sum.
/// @throws std::overflow_error Candidate addition produces a nonfinite value.
double sum(double a,double b) {
  const double c=a+b;if(!std::isfinite(c)) throw std::overflow_error("DAG longest paths: nonfinite reachable candidate");return c;
}
/// @brief Execute whole-graph FIFO Kahn ordering, then source-reachable maximum relaxations.
/// @param n Valid positive vertex count.
/// @param edges Valid borrowed input edges.
/// @param source Valid source id.
/// @param trace Optional actual-history destination; null avoids capture/counter work.
/// @return Owned original-layout distance/predecessor result.
/// @throws std::invalid_argument Any whole-graph cycle.
/// @throws std::overflow_error Reachable candidate or captured counter overflow.
/// @throws std::length_error Requested container capacity exceeded.
/// @throws std::bad_alloc Graph/order/output/history allocation failure.
LongestPathResult run(std::size_t n,const std::vector<WeightedEdge>& edges,std::size_t source,DagLongestPathTrace* trace) {
  std::vector<std::vector<std::size_t>> outgoing(n);std::vector<std::size_t> indegree(n,0),order;order.reserve(n);
  for(std::size_t i=0;i<edges.size();++i) {outgoing[std::get<0>(edges[i])].push_back(i);++indegree[std::get<1>(edges[i])];}
  for(std::size_t u=0;u<n;++u) if(!indegree[u]) order.push_back(u);
  for(std::size_t head=0;head<order.size();++head) {
    const auto u=order[head];if(trace) tick(trace->work.topological_pops);
    for(auto edge:outgoing[u]) {if(trace) tick(trace->work.topological_edge_visits);const auto v=std::get<1>(edges[edge]);if(--indegree[v]==0) order.push_back(v);}
  }
  if(order.size()!=n) throw std::invalid_argument("DAG longest paths: graph contains a cycle");
  LongestPathResult r;r.distance.assign(n,-std::numeric_limits<double>::infinity());r.predecessor.assign(n,n);r.distance[source]=0;
  if(trace) {trace->source=source;trace->edges=edges;trace->predecessor_edge.assign(n,edges.size());}
  for(auto u:order) if(std::isfinite(r.distance[u])) {
    if(trace) tick(trace->work.reachable_vertices);
    for(auto edge:outgoing[u]) {
      const auto v=std::get<1>(edges[edge]);const double candidate=sum(r.distance[u],std::get<2>(edges[edge]));const bool improved=candidate>r.distance[v];
      if(trace) {tick(trace->work.relaxation_attempts);trace->relaxations.push_back({edge,r.distance[u],r.distance[v],candidate,improved});}
      if(improved) {r.distance[v]=candidate;r.predecessor[v]=u;if(trace) {trace->predecessor_edge[v]=edge;tick(trace->work.improvements);}}
    }
  }
  if(trace) trace->topological_order=std::move(order);return r;
}
/// @brief Multiply exact edge-count factors without wrapping.
/// @param a First nonnegative factor.
/// @param b Second nonnegative factor.
/// @return Exact product.
/// @throws std::length_error Product cannot fit size_t.
std::size_t product(std::size_t a,std::size_t b) {
  if(a && b>std::numeric_limits<std::size_t>::max()/a) throw std::length_error("DAG band: count overflow");return a*b;
}
} // namespace dag_longest_detail

LongestPathResult dag_longest_paths(std::size_t n,const std::vector<WeightedEdge>& edges,std::size_t source) {
  dag_longest_detail::validate(n,edges,source);return dag_longest_detail::run(n,edges,source,nullptr);
}
DagLongestPathTrace dag_longest_paths_trace(std::size_t n,const std::vector<WeightedEdge>& edges,std::size_t source,std::size_t max_vertices,std::size_t max_edges) {
  dag_longest_detail::validate(n,edges,source);
  if((max_vertices && n>max_vertices)||(max_edges && edges.size()>max_edges)) throw std::length_error("DAG longest paths: complete input cap exceeded");
  DagLongestPathTrace t;t.result=dag_longest_detail::run(n,edges,source,&t);return t;
}
DagPath dag_longest_path_to(const DagLongestPathTrace& t,std::size_t target) {
  const auto n=t.result.distance.size(),m=t.edges.size();if(target>=n) throw std::out_of_range("DAG path: target out of range");
  if(t.source>=n || t.result.predecessor.size()!=n || t.predecessor_edge.size()!=n || t.result.distance[t.source]!=0 || t.result.predecessor[t.source]!=n || t.predecessor_edge[t.source]!=m) throw std::invalid_argument("DAG path: malformed result/source");
  DagPath p;p.weight=t.result.distance[target];
  if(p.weight==-std::numeric_limits<double>::infinity()) {
    if(t.result.predecessor[target]!=n || t.predecessor_edge[target]!=m) throw std::invalid_argument("DAG path: inconsistent unreachable markers");return p;
  }
  if(!std::isfinite(p.weight)) throw std::invalid_argument("DAG path: nonfinite reachable distance");
  std::size_t v=target;
  for(std::size_t steps=0;;++steps) {
    if(steps>=n) throw std::invalid_argument("DAG path: predecessor cycle");p.vertices.push_back(v);if(v==t.source) break;
    const auto u=t.result.predecessor[v],edge=t.predecessor_edge[v];
    if(u>=n || edge>=m) throw std::invalid_argument("DAG path: missing predecessor edge");
    const auto& e=t.edges[edge];const double weight=std::get<2>(e);
    if(std::get<0>(e)!=u || std::get<1>(e)!=v || !std::isfinite(weight) || !std::isfinite(t.result.distance[u])) throw std::invalid_argument("DAG path: inconsistent predecessor edge");
    const double candidate=t.result.distance[u]+weight;
    if(!std::isfinite(candidate) || candidate!=t.result.distance[v]) throw std::invalid_argument("DAG path: inconsistent distance recurrence");
    p.edge_indices.push_back(edge);v=u;
  }
  std::reverse(p.vertices.begin(),p.vertices.end());std::reverse(p.edge_indices.begin(),p.edge_indices.end());p.found=true;return p;
}
std::vector<WeightedEdge> dag_forward_band_edges(std::size_t n,std::size_t width,double weight,std::size_t max_edges) {
  if(!std::isfinite(weight)) throw std::invalid_argument("DAG band: nonfinite weight");width=std::min(width,n?n-1:0);
  const auto rectangle=dag_longest_detail::product(width,n-width);std::size_t a=width,b=width?width-1:0;if(a%2==0) a/=2;else b/=2;
  const auto triangle=dag_longest_detail::product(a,b);if(triangle>std::numeric_limits<std::size_t>::max()-rectangle) throw std::length_error("DAG band: count sum overflow");
  const auto count=rectangle+triangle;if(max_edges && count>max_edges) throw std::length_error("DAG band: complete edge cap exceeded");
  std::vector<WeightedEdge> edges;if(!count) return edges;edges.reserve(count);
  for(std::size_t u=0;u<n-1;++u) for(std::size_t d=1;d<=std::min(width,n-1-u);++d) edges.emplace_back(u,u+d,weight);
  return edges;
}
} // namespace datamunge::algorithms
