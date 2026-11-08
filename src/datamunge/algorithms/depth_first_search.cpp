/// @file
/// @brief Recursive-order DFS using heap continuation frames and optional actual transitions.
#include <datamunge/algorithms/depth_first_search.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private ordered adjacency, explicit activations and common traced/untraced kernel.
namespace {
/// @brief Successor/original-edge-id pair, retaining input arc identity.
using Arc=std::pair<std::size_t,std::size_t>;
/// @brief Per-source ordered outgoing adjacency with duplicate/self-loop entries retained.
using Adjacency=std::vector<std::vector<Arc>>;
/// @brief One suspended recursive activation with its exact next successor position.
struct Frame {
  std::size_t vertex; ///< Active current vertex.
  std::size_t next{0}; ///< Next adjacency entry to inspect after returning from a child.
};
/// @brief Append one actual event after checking its optional inclusive count cap.
/// @param out Mutable actual trace.
/// @param event Fully computed scalar event.
/// @param limit Zero disables only this optional ceiling.
/// @throws std::length_error Event cap or vector capacity exceeded.
/// @throws std::bad_alloc Allocation fails.
void emit(DepthFirstTrace& out,DepthFirstEvent event,std::size_t limit) {
  if(limit && out.events.size()>=limit)throw std::length_error("DFS actual event cap exceeded");out.events.push_back(event);
}
/// @brief Validate a complete graph domain without allocation or target shortcuts.
/// @param n Vertex count.
/// @param edges Original directed arcs.
/// @param start Required valid source.
/// @param goal Required valid target unless complete=true.
/// @param complete Whether no target is requested.
/// @throws std::invalid_argument Any required terminal or endpoint is outside [0,n).
void validate(std::size_t n,const std::vector<Arc>& edges,std::size_t start,std::size_t goal,bool complete) {
  if(start>=n || (!complete && goal>=n))throw std::invalid_argument("DFS terminal outside graph");
  for(const auto [u,v]:edges)if(u>=n||v>=n)throw std::invalid_argument("DFS endpoint outside graph");
}
/// @brief Execute the original global-marking directed DFS with explicit continuation frames.
/// @tparam Record Retain actual metadata/events when true; no history allocation otherwise.
/// @tparam Complete Exhaust the source component instead of stopping at the goal.
/// @param n Complete vertex count.
/// @param edges Complete directed arcs in input order.
/// @param start Validated source after domain checking inside this function.
/// @param goal Requested target, ignored for Complete.
/// @param options Optional traced-call ceilings, ignored for untraced calls.
/// @return Actual first-path result plus optional chronological trace, without replay.
/// @throws std::invalid_argument Invalid graph domain.
/// @throws std::length_error Optional limit, timestamp arithmetic or capacity exceeded.
/// @throws std::bad_alloc Allocation fails.
template<bool Record,bool Complete=false>
DepthFirstTrace run(std::size_t n,const std::vector<Arc>& edges,std::size_t start,std::size_t goal,DepthFirstOptions options) {
  validate(n,edges,start,goal,Complete);DepthFirstTrace out;out.n=n;out.start=start;out.goal=Complete?n:goal;out.goal_requested=!Complete;
  std::size_t none=0,clock=0;
  if constexpr(Record) {
    if((options.max_vertices&&n>options.max_vertices)||(options.max_edges&&edges.size()>options.max_edges))throw std::length_error("DFS complete graph cap exceeded");
    if(n>std::numeric_limits<std::size_t>::max()/2)throw std::length_error("DFS timestamp sentinel overflow");none=2*n;
  }
  Adjacency graph(n);for(std::size_t i=0;i<edges.size();++i)graph[edges[i].first].push_back({edges[i].second,i});
  std::vector<DepthFirstState> state(n,DepthFirstState::unseen);std::vector<Frame> stack;
  if constexpr(Record) {out.edges=edges;out.parent.assign(n,n);out.parent_edge.assign(n,edges.size());out.depth.assign(n,n);out.discovered_at.assign(n,none);out.finished_at.assign(n,none);}
  /// @brief Mark/push one genuinely unseen vertex and record its actual entry.
  /// @param vertex New vertex, currently unseen.
  /// @param parent Parent vertex, or n for the root.
  /// @param via Original incoming edge id, or m for the root.
  const auto enter=[&](std::size_t vertex,std::size_t parent,std::size_t via) {
    state[vertex]=DepthFirstState::active;stack.push_back({vertex,0});++out.result.expanded;++out.work.discovered;out.work.max_stack_size=std::max(out.work.max_stack_size,stack.size());
    if constexpr(Record) {out.parent[vertex]=parent;out.parent_edge[vertex]=via;out.depth[vertex]=stack.size()-1;out.discovered_at[vertex]=clock++;out.preorder.push_back(vertex);emit(out,{DepthFirstEventKind::discover,vertex,n,via,stack.size()-1,stack.size(),DepthFirstState::unseen},options.max_events);}
  };
  enter(start,n,edges.size());
  while(!stack.empty()) {
    const auto vertex=stack.back().vertex,depth=stack.size()-1;
    if(!Complete && vertex==goal) {
      out.result.found=true;for(const auto& f:stack)out.result.path.push_back(f.vertex);
      if constexpr(Record)emit(out,{DepthFirstEventKind::goal,vertex,n,edges.size(),depth,stack.size(),DepthFirstState::unseen},options.max_events);
      break;
    }
    if(stack.back().next==graph[vertex].size()) {
      state[vertex]=DepthFirstState::finished;stack.pop_back();++out.work.finished;
      if constexpr(Record) {out.finished_at[vertex]=clock++;out.postorder.push_back(vertex);emit(out,{DepthFirstEventKind::finish,vertex,n,edges.size(),depth,stack.size(),DepthFirstState::unseen},options.max_events);}
      continue;
    }
    const auto [next,id]=graph[vertex][stack.back().next++];const auto before=state[next];++out.work.edges_examined;
    if(before==DepthFirstState::unseen)++out.work.unseen_edges;else if(before==DepthFirstState::active)++out.work.active_edges;else ++out.work.finished_edges;
    if constexpr(Record)emit(out,{DepthFirstEventKind::examine,vertex,next,id,depth,stack.size(),before},options.max_events);
    if(before==DepthFirstState::unseen)enter(next,vertex,id);
  }
  if constexpr(Record)out.state=std::move(state);return out;
}
} // namespace
BasicPathResult depth_first_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal) {return run<false>(n,edges,start,goal,{}).result;}
DepthFirstTrace depth_first_search_trace(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal,DepthFirstOptions options) {return run<true>(n,edges,start,goal,options);}
DepthFirstTrace depth_first_traversal(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,DepthFirstOptions options) {return run<true,true>(n,edges,start,n,options);}
std::vector<std::pair<std::size_t,std::size_t>> rooted_tree_edges(std::size_t n,std::size_t branching,std::size_t max_edges) {
  if(!branching)throw std::invalid_argument("rooted tree branching must be positive");const auto count=n?n-1:0;
  if(max_edges&&count>max_edges)throw std::length_error("rooted tree complete edge cap exceeded");std::vector<std::pair<std::size_t,std::size_t>> out;out.reserve(count);
  for(std::size_t i=1;i<n;++i)out.push_back({(i-1)/branching,i});return out;
}
} // namespace datamunge::algorithms
