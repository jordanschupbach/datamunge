/// @file chu_liu_edmonds.cpp
/// @brief Iterative checked contractions, compact original-edge expansion and optional actual round snapshots.
#include <datamunge/algorithms/chu_liu_edmonds.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked counts/costs, expansion history and one shared executing kernel.
namespace branching_detail {
/// @brief Add a named count without wrapping size_t.
/// @param target Mutable accumulated count.
/// @param amount Nonnegative actual increment, default one.
/// @throws std::overflow_error If the new count cannot fit size_t.
void count(std::size_t& target,std::size_t amount=1) {
  if(amount>std::numeric_limits<std::size_t>::max()-target)throw std::overflow_error("branching: work counter overflow");target+=amount;
}
/// @brief Add finite signed working costs and reject nonfinite arithmetic.
/// @param a First finite cost.
/// @param b Second finite cost.
/// @return Finite widened sum.
/// @throws std::overflow_error If the working sum is nonfinite.
long double add(long double a,long double b) {
  const auto sum=a+b;if(!std::isfinite(sum))throw std::overflow_error("branching: working cost overflow");return sum;
}
/// @brief Check final signed cost conversion to the unchanged public double field.
/// @param value Finite widened sum of selected original edges.
/// @return Finite double total.
/// @throws std::overflow_error If the total cannot be represented as finite double.
double narrow(long double value) {
  if(!std::isfinite(value) || std::fabs(value)>std::numeric_limits<double>::max())throw std::overflow_error("branching: total exceeds double range");
  return static_cast<double>(value);
}
/// @brief Per-call actual counters and complete-call admission caps.
struct Context {
  BranchingOptions options; ///< Caller-supplied phase and named edge-visit caps.
  BranchingWork work; ///< Actual completed validation and executed operations.
  std::size_t edge_visits{0}; ///< Combined actual incoming/reweight inspections.

  /// @brief Admit a new round before its allocations and scan.
  /// @throws std::length_error If the positive whole-call round cap is exhausted.
  /// @throws std::overflow_error If the phase counter cannot fit.
  void phase() {
    if(options.max_phases && work.phases==options.max_phases)throw std::length_error("branching: phase cap exhausted");count(work.phases);
  }
  /// @brief Admit one actual edge inspection, including ignored/internal arcs.
  /// @param reweight Whether this is contraction transformation rather than incoming selection.
  /// @throws std::length_error If the positive whole-call visit cap is exhausted.
  /// @throws std::overflow_error If a visit counter cannot fit.
  void visit(bool reweight) {
    if(options.max_edge_visits && edge_visits==options.max_edge_visits)throw std::length_error("branching: edge-visit cap exhausted");
    count(edge_visits);count(reweight?work.reweight_edge_visits:work.incoming_edge_visits);
  }
};
/// @brief Compact actual level choices and original-vertex partition needed for reverse expansion.
struct History {
  std::size_t root{0}; ///< This level's root, excluded from incoming choices.
  std::vector<std::size_t> incoming; ///< Actual selected original input edge per level vertex.
  std::vector<std::size_t> original_to_vertex; ///< Map from each original vertex to this level.
};
/// @brief Solve once with actual level recovery and optional full round snapshots.
/// @tparam Record Whether to retain public round snapshots.
/// @param original_n Positive original vertex count.
/// @param input Borrowed finite signed directed weighted edges.
/// @param original_root Valid prescribed original root.
/// @param options Whole-call phase and edge-visit admission limits.
/// @return Recovered original tree, actual reduced total and counts, optionally full rounds.
/// @throws std::invalid_argument If root/endpoints are invalid or any weight is nonfinite.
/// @throws std::length_error If a cap or container capacity is exceeded.
/// @throws std::overflow_error If a working sum/count or final total cannot fit.
/// @throws std::bad_alloc If allocation fails.
template<bool Record>
BranchingTrace run(std::size_t original_n,const std::vector<WeightedEdge>& input,std::size_t original_root,BranchingOptions options) {
  if(original_root>=original_n)throw std::invalid_argument("chu_liu_edmonds: root outside nonempty graph");
  Context context{options,{},{}};
  for(const auto& [u,v,w]:input) {
    if(u>=original_n || v>=original_n || !std::isfinite(w))throw std::invalid_argument("chu_liu_edmonds: invalid endpoint or nonfinite weight");
    count(context.work.validated_edges);
  }
  std::vector<BranchingArc> edges;edges.reserve(input.size());
  for(std::size_t i=0;i<input.size();++i){const auto [u,v,w]=input[i];edges.push_back({u,v,w,i});}
  std::size_t n=original_n,root=original_root;
  std::vector<std::size_t> original_to_vertex(original_n);std::iota(original_to_vertex.begin(),original_to_vertex.end(),0);
  std::vector<History> history;std::vector<std::size_t> selected;BranchingTrace out;
  while(true) {
    context.phase();BranchingPhase phase;phase.vertex_count=n;phase.root=root;
    if constexpr(Record){phase.edges=edges;phase.original_to_vertex=original_to_vertex;}
    phase.incoming_edges.assign(n,kNoBranchingEdge);phase.incoming_weights.assign(n,std::numeric_limits<long double>::infinity());
    std::vector<std::size_t> parent(n,kNoBranchingEdge);parent[root]=root;phase.incoming_weights[root]=0;
    for(const auto& e:edges) {
      context.visit(false);
      if(e.from!=e.to && e.to!=root && e.weight<phase.incoming_weights[e.to]) {
        phase.incoming_edges[e.to]=e.original_index;phase.incoming_weights[e.to]=e.weight;parent[e.to]=e.from;
      }
    }
    for(std::size_t v=0;v<n;++v)if(v!=root && phase.incoming_edges[v]==kNoBranchingEdge)phase.missing_vertices.push_back(v);
    if(!phase.missing_vertices.empty()) {
      phase.cumulative_cost=out.reduced_total;if constexpr(Record)out.phases.push_back(std::move(phase));
      out.work=context.work;return out;
    }
    for(std::size_t v=0;v<n;++v)phase.added_cost=add(phase.added_cost,phase.incoming_weights[v]);
    out.reduced_total=add(out.reduced_total,phase.added_cost);phase.cumulative_cost=out.reduced_total;
    std::vector<std::uint8_t> state(n);state[root]=2;std::vector<std::size_t> position(n),path;
    phase.component.assign(n,kNoBranchingEdge);
    for(std::size_t v=0;v<n;++v)if(!state[v]) {
      path.clear();auto x=v;
      while(!state[x]){state[x]=1;position[x]=path.size();path.push_back(x);count(context.work.predecessor_steps);x=parent[x];}
      if(state[x]==1) {
        const auto id=phase.cycles.size();phase.cycles.emplace_back(path.begin()+position[x],path.end());
        for(const auto member:phase.cycles.back())phase.component[member]=id;
        count(context.work.cycles_contracted);
      }
      for(const auto member:path)state[member]=2;
    }
    if(phase.cycles.empty()) {
      phase.component.clear();
      for(std::size_t v=0;v<n;++v)if(v!=root)selected.push_back(phase.incoming_edges[v]);
      if constexpr(Record)out.phases.push_back(std::move(phase));break;
    }
    auto next_n=phase.cycles.size();
    for(auto& component:phase.component)if(component==kNoBranchingEdge)component=next_n++;
    history.push_back({root,phase.incoming_edges,original_to_vertex});
    count(context.work.saved_vertex_entries,phase.incoming_edges.size());count(context.work.saved_vertex_entries,original_to_vertex.size());
    std::vector<BranchingArc> next;next.reserve(edges.size());
    for(const auto& e:edges) {
      context.visit(true);const auto u=phase.component[e.from],v=phase.component[e.to];
      const auto weight=u!=v?add(e.weight,-phase.incoming_weights[e.to]):e.weight;
      next.push_back({u,v,weight,e.original_index});
    }
    for(auto& vertex:original_to_vertex)vertex=phase.component[vertex];root=phase.component[root];n=next_n;edges=std::move(next);
    if constexpr(Record)out.phases.push_back(std::move(phase));
  }
  for(auto i=history.rbegin();i!=history.rend();++i) {
    auto lower=i->incoming;
    for(const auto edge:selected) {
      const auto target=i->original_to_vertex[std::get<1>(input[edge])];lower[target]=edge;count(context.work.expansion_replacements);
    }
    selected.clear();for(std::size_t v=0;v<lower.size();++v)if(v!=i->root)selected.push_back(lower[v]);
  }
  out.tree.parent.assign(original_n,kNoBranchingEdge);out.tree.parent[original_root]=original_root;
  out.tree.incoming_edges.assign(original_n,kNoBranchingEdge);
  for(const auto id:selected){const auto [u,v,w]=input[id];out.tree.parent[v]=u;out.tree.incoming_edges[v]=id;}
  long double total=0;
  for(std::size_t v=0;v<original_n;++v)if(v!=original_root)total=add(total,std::get<2>(input[out.tree.incoming_edges[v]]));
  out.tree.result={narrow(total),true};out.work=context.work;return out;
}
} // namespace branching_detail
DirectedBranchingResult chu_liu_edmonds(std::size_t vertex_count,const std::vector<WeightedEdge>& directed_edges,std::size_t root) {
  return branching_detail::run<false>(vertex_count,directed_edges,root,{}).tree.result;
}
DirectedArborescence chu_liu_edmonds_tree(std::size_t vertex_count,const std::vector<WeightedEdge>& directed_edges,std::size_t root,BranchingOptions options) {
  return branching_detail::run<false>(vertex_count,directed_edges,root,options).tree;
}
BranchingTrace chu_liu_edmonds_trace(std::size_t vertex_count,const std::vector<WeightedEdge>& directed_edges,std::size_t root,BranchingOptions options) {
  return branching_detail::run<true>(vertex_count,directed_edges,root,options);
}
std::vector<WeightedEdge> rooted_cycle_edges(std::size_t cycle_vertices,double cycle_cost,double entry_cost) {
  if(cycle_vertices<2 || !std::isfinite(cycle_cost) || !std::isfinite(entry_cost))throw std::invalid_argument("rooted cycle: invalid size or nonfinite cost");
  if(cycle_vertices==std::numeric_limits<std::size_t>::max() || cycle_vertices>std::vector<WeightedEdge>().max_size()/2)throw std::length_error("rooted cycle: edge/vertex count exceeds capacity");
  std::vector<WeightedEdge> edges;edges.reserve(2*cycle_vertices);
  for(std::size_t v=1;v<=cycle_vertices;++v)edges.emplace_back(0,v,entry_cost);
  for(std::size_t v=1;v<=cycle_vertices;++v)edges.emplace_back(v,v==cycle_vertices?1:v+1,cycle_cost);
  return edges;
}
} // namespace datamunge::algorithms
