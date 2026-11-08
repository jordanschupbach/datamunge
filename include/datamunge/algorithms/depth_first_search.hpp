#pragma once
/// @file
/// @brief Explicit-frame directed DFS preserving first-path order, with actual events and source traversal.
#include <datamunge/algorithms/basic_path_result.hpp>
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief DFS state at the stopping time or immediately before a scanned arc.
/// Active vertices are exactly the current frame path. Finished vertices have
/// genuinely exhausted adjacency; unseen vertices have never been entered.
/// A successful early stop leaves the goal and its ancestors active.
/// @par Example
/// @code
/// const auto t=dm::algorithms::depth_first_search_trace(2,{{0,1}},0,1);
/// std::print("{}\n",t.state[1]==dm::algorithms::DepthFirstState::active); // true
/// @endcode
/// @see DepthFirstTrace
 enum class DepthFirstState { unseen, ///< Never entered at stopping time.
                              active, ///< Entered and still on the explicit frame path.
                              finished ///< All outgoing adjacency entries genuinely exhausted.
};
/// @brief One actual directed DFS transition, with no synthetic early-stop finishes.
/// Discover occurs after marking/pushing a new vertex; examine before deciding
/// whether to descend; finish after a frame pop; goal after its discovery.
/// @par Example
/// @code
/// const auto t=dm::algorithms::depth_first_search_trace(2,{{0,1}},0,1);
/// std::print("{}\n",t.events.back().kind==dm::algorithms::DepthFirstEventKind::goal); // true
/// @endcode
/// @see DepthFirstEvent
 enum class DepthFirstEventKind { discover, ///< A previously unseen vertex was marked and pushed.
                                  examine, ///< One original adjacency entry is inspected.
                                  finish, ///< An exhausted frame was actually popped.
                                  goal ///< Newly entered target terminates search before its adjacency.
};
/// @brief Inclusive limits on complete graph storage and actual retained event count.
/// Complete terminal/endpoint validation precedes all caps and allocation.
/// Original edges count loops, duplicates and unreachable components. Event caps
/// are checked online before appending each actual event. Zero disables an
/// optional cap; arithmetic/capacity checks remain. Failures return no partial trace.
/// @par Example
/// @code
/// const dm::algorithms::DepthFirstOptions o;
/// std::print("{} {} {}\n",o.max_vertices,o.max_edges,o.max_events); // 1000000 1000000 3000000
/// @endcode
/// @see depth_first_search_trace(), depth_first_traversal()
struct DepthFirstOptions {
  std::size_t max_vertices{1000000}; ///< Complete n cap, including unreachable vertices; zero disables.
  std::size_t max_edges{1000000}; ///< Complete original arc count cap; zero disables.
  std::size_t max_events{3000000}; ///< Actual retained transition count cap; zero disables.
};
/// @brief Owned scalar data for one transition from the executing DFS kernel.
/// vertex is the current/new/finished vertex. neighbor exists only for examine;
/// otherwise it is n. edge_index is the scanned original id for examine, incoming
/// tree-edge id for discover, and m for the root or finish/goal events. depth counts
/// tree arcs from start. stack_size is after push/pop for discover/finish and
/// current size for examine/goal. neighbor_state is sampled before examination
/// descent, and is the unused unseen placeholder for every other event kind.
/// @par Example
/// @code
/// const auto t=dm::algorithms::depth_first_search_trace(2,{{0,1}},0,1);
/// std::print("{} {} {}\n",t.events[1].vertex,t.events[1].neighbor,t.events[1].edge_index); // 0 1 0
/// @endcode
/// @see DepthFirstTrace, DepthFirstEventKind
struct DepthFirstEvent {
  DepthFirstEventKind kind{DepthFirstEventKind::discover}; ///< Actual transition kind, default discover.
  std::size_t vertex{0}; ///< Source/current vertex for this event.
  std::size_t neighbor{0}; ///< Examined target, or n when unused in a native result.
  std::size_t edge_index{0}; ///< Original examined/incoming edge id, or m when unused/root.
  std::size_t depth{0}; ///< Current vertex's DFS-tree depth, zero at the source.
  std::size_t stack_size{0}; ///< Actual active-frame count at the event boundary.
  DepthFirstState neighbor_state{DepthFirstState::unseen}; ///< Target state before an examine event's descent.
};
/// @brief Actual vertex entries/exits, adjacency scans and maximum live frame count.
/// Each vertex is discovered at most once. Every examined arc has exactly one
/// target-state category; unseen arcs cause discovery, active arcs close a
/// directed ancestor cycle (including self-loops), finished arcs need no descent.
/// Counters exclude domain validation, adjacency construction, allocation/copies,
/// result-path extraction and timestamp/event bookkeeping. They are not timings.
/// @par Example
/// @code
/// const auto t=dm::algorithms::depth_first_traversal(3,{{0,1},{1,2},{2,0}},0);
/// std::print("{} {} {}\n",t.work.discovered,t.work.edges_examined,t.work.active_edges); // 3 3 1
/// @endcode
/// @see DepthFirstTrace
struct DepthFirstWork {
  std::size_t discovered{0}; ///< Actual distinct entries, including a reached goal.
  std::size_t finished{0}; ///< Frames genuinely exhausted/popped before stopping.
  std::size_t edges_examined{0}; ///< Actual adjacency entries inspected, including loops/duplicates.
  std::size_t unseen_edges{0}; ///< Scans leading to a previously unseen target.
  std::size_t active_edges{0}; ///< Scans to a target on the current active path.
  std::size_t finished_edges{0}; ///< Scans to a genuinely completed target.
  std::size_t max_stack_size{0}; ///< Maximum active explicit-frame count, including the source.
};
/// @brief Owned goal-search or complete source-component DFS state and actual events.
/// n is the parent/depth sentinel; edges.size() is the absent parent-edge marker;
/// 2*n is the discovery/finish-time sentinel, checked representable before graph
/// allocation. Times tick only at discoveries and genuine finishes, starting zero.
/// Seen-but-unfinished vertices therefore have a valid discovery and missing finish.
/// Parent/edge/depth are set on first entry only. preorder contains all entries,
/// postorder only actual finishes. Successful goal search keeps its path active;
/// it does not fabricate unwinding. Full traversal sets goal_requested=false,
/// goal=n, result.path empty and result.found=false, while expanded/preorder exhaust
/// all source-reachable vertices. Unseen after an early stop need not mean unreachable.
/// Every container owns data, including original edges; no shared mutable state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::depth_first_search_trace(3,{{0,1},{1,2}},0,2);
/// std::print("{} {} {}\n",t.preorder,t.postorder,t.work.max_stack_size); // [0, 1, 2] [] 3
/// @endcode
/// @see depth_first_search_trace(), depth_first_traversal()
struct DepthFirstTrace {
  BasicPathResult result; ///< Legacy first-path result and distinct entry count.
  std::size_t n{0}; ///< Original vertex count; parent/depth sentinel.
  std::size_t start{0}; ///< Original valid source.
  std::size_t goal{0}; ///< Original goal, or n in full traversal.
  bool goal_requested{true}; ///< True only when the kernel may stop at a target.
  std::vector<std::pair<std::size_t,std::size_t>> edges; ///< Owned original ordered directed arcs.
  std::vector<std::size_t> preorder; ///< Actual first-entry order.
  std::vector<std::size_t> postorder; ///< Actual exhausted-frame finish order.
  std::vector<std::size_t> parent; ///< First-discovery parent, n for root/unseen.
  std::vector<std::size_t> parent_edge; ///< Incoming original edge id, m for root/unseen.
  std::vector<std::size_t> depth; ///< First-entry tree depth, n for unseen.
  std::vector<std::size_t> discovered_at; ///< Actual discovery ticks, 2*n for unseen.
  std::vector<std::size_t> finished_at; ///< Actual finish ticks, 2*n for active/unseen.
  std::vector<DepthFirstState> state; ///< Actual unseen/active/finished state at stopping time.
  std::vector<DepthFirstEvent> events; ///< Chronological actual transitions, without replay or stack snapshots.
  DepthFirstWork work; ///< Actual counts excluding graph preparation and recording overhead.
};
/// @brief Find the first directed source-to-goal path in recursive DFS adjacency order.
/// Preserves the original four-argument signature, BasicPathResult layout and
/// search_more.hpp include. Uses heap-stored {vertex,next-successor} frames instead
/// of recursive calls, preserving descent and resume order without a C++ call-stack
/// depth limit. Vertices are marked globally on first entry and never entered again.
/// Outgoing arcs retain input order; parallel arcs/self-loops are accepted but cannot
/// rediscover a marked vertex. Stop immediately on first entering the goal, before
/// examining its outgoing arcs. Start=goal returns one vertex; failure an empty path.
/// expanded counts distinct first entries including the goal, not edge scans. The
/// selected simple path need not be shortest and may change with edge order.
/// All terminals/edges are validated before allocation, even for start=goal or
/// unreachable bad arcs. O(n+m) total worst-case time/storage including adjacency;
/// the traversal alone uses O(reached vertices+scanned arcs) work and at most n
/// active frames. No trace/history allocation, input mutation or shared state.
/// @param n Number of vertices labelled 0..n-1, must admit both terminals.
/// @param edges Borrowed ordered directed arcs; unchanged.
/// @param start Valid source id.
/// @param goal Valid target id.
/// @return Owned first DFS path, distinct entry count and success flag.
/// @throws std::invalid_argument Any terminal/endpoint is outside [0,n), including n=0.
/// @throws std::length_error Container capacity exceeded.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::depth_first_search(2,{{0,1}},0,1);
/// std::print("{} {} {}\n",r.path,r.expanded,r.found); // [0, 1] 2 true
/// @endcode
/// @par References
/// - R. Tarjan (1972), Depth-First Search and Linear Graph Algorithms, SIAM J. Comput. 1(2),146–160, doi:10.1137/0201010.
/// - Sedgewick and Wayne, Algorithms, directed DFS path reference: https://algs4.cs.princeton.edu/42digraph/DepthFirstDirectedPaths.java.html . This variant uses explicit continuation frames and stops on first goal entry.
/// @see depth_first_search_trace(), depth_first_traversal(), breadth_first_search()
BasicPathResult depth_first_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal);
/// @brief Record actual entry, edge-target state, finish and goal events from the same DFS.
/// No replay, full stack copies or synthetic early-stop finishes. Returns the same
/// path/entry count as depth_first_search(), plus state and timing described by
/// DepthFirstTrace. Retains O(n+m) input/state/event storage: at most two events per
/// reached vertex, one per scanned arc and one goal event. A successful early stop
/// may leave vertices unseen or active. Complete graph validation precedes all
/// optional caps; timestamp 2*n and container capacities remain checked when caps
/// are disabled. Graph preparation remains O(n+m), even for an immediate goal.
/// @param n Complete vertex count, including unreachable vertices.
/// @param edges Complete ordered directed arcs, copied into the returned trace.
/// @param start Valid source id.
/// @param goal Valid target id.
/// @param options Vertex/edge/event ceilings, default 1000000/1000000/3000000; zero disables each.
/// @return Owned actual trace, first DFS path and distinct-entry count.
/// @throws std::invalid_argument Any terminal/endpoint outside [0,n).
/// @throws std::length_error Optional cap, timestamp arithmetic or container capacity exceeded.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::depth_first_search_trace(2,{{0,1}},0,1,{2,1,4});
/// std::print("{} {} {}\n",t.events.size(),t.work.finished,t.work.edges_examined); // 4 0 1
/// @endcode
/// @par References
/// - Tarjan (1972), doi:10.1137/0201010; actual directed DFS events under this library's first-goal rule.
/// @see depth_first_search(), DepthFirstTrace
DepthFirstTrace depth_first_search_trace(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,
                                         std::size_t start,std::size_t goal,DepthFirstOptions options={});
/// @brief Exhaust the source-reachable directed component and record genuine finishes.
/// Uses the same explicit continuation kernel without a goal stop. Discovery and
/// finish intervals are complete for every reachable vertex; isolated/unreachable
/// vertices remain unseen. This is one source tree, not a forest over all n vertices.
/// The path/success fields are intentionally empty/false; expanded counts all entries.
/// Work, caps, timestamps, ownership and O(n+m) storage match the traced goal call.
/// @param n Complete vertex count, must admit start.
/// @param edges Complete ordered directed arcs, copied into the trace.
/// @param start Valid starting vertex.
/// @param options Inclusive vertex/edge/event caps; defaults as depth_first_search_trace().
/// @return Complete owned source traversal with goal_requested=false and goal=n.
/// @throws std::invalid_argument Invalid source/edge endpoint, including n=0.
/// @throws std::length_error Optional cap, timestamp arithmetic or container capacity exceeded.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::depth_first_traversal(3,{{0,1},{1,2}},0);
/// std::print("{} {}\n",t.preorder,t.postorder); // [0, 1, 2] [2, 1, 0]
/// @endcode
/// @par References
/// - Tarjan (1972), doi:10.1137/0201010; one complete source component under supplied adjacency order.
/// @see depth_first_search_trace()
DepthFirstTrace depth_first_traversal(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,
                                      std::size_t start,DepthFirstOptions options={});
/// @brief Generate a labelled forward rooted tree with fixed maximum branching.
/// For child i=1..n-1, emit ((i-1)/branching,i) in increasing child order.
/// Root is zero. Branching one gives a chain, two a heap-indexed binary tree,
/// and branching>=n gives a star. Zero/one vertices give no edges, but branching
/// must still be positive. O(n) work/output storage, owned deterministic arcs,
/// no randomness/shared state. Complete n-1 output count is capped before allocation.
/// @param n Vertex count; zero is allowed for generation.
/// @param branching Positive maximum children per parent, default two.
/// @param max_edges Complete output cap, default 1000000; zero disables only optional cap.
/// @return Owned forward tree arcs, n-1 for n>0 and empty for n=0.
/// @throws std::invalid_argument branching is zero.
/// @throws std::length_error Optional cap or vector capacity exceeded before allocation.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rooted_tree_edges(4,2)); // [(0, 1), (0, 2), (1, 3)]
/// @endcode
/// @see depth_first_traversal()
std::vector<std::pair<std::size_t,std::size_t>> rooted_tree_edges(std::size_t n,std::size_t branching=2,std::size_t max_edges=1000000);
} // namespace datamunge::algorithms
