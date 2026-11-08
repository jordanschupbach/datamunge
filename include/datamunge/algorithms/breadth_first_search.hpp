#pragma once
/// @file breadth_first_search.hpp
/// @brief FIFO shortest-hop graph search with actual queue/discovery traces.
#include <datamunge/algorithms/unweighted_search_result.hpp>
#include <utility>
namespace datamunge::algorithms {
/// @brief One actual dequeue and the discoveries made while expanding it.
/// The goal dequeue is included but its adjacency is not scanned. Discovered
/// vertices are recorded in input adjacency order and are marked before enqueueing.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::breadth_first_search_trace(3,{{0,1},{1,2}},0,2);
/// std::print("{} {}\n",trace.steps[0].vertex,trace.steps[0].discovered); // 0 [1]
/// @endcode
/// @see BreadthFirstTrace
struct BreadthFirstStep {
    std::size_t vertex{0}; ///< Vertex removed from the FIFO front.
    std::size_t depth{0}; ///< Minimum number of edges from the start to vertex.
    std::size_t queue_size_before{0}; ///< Active queue length immediately before this removal.
    std::size_t queue_size_after{0}; ///< Active queue length after the removal and any new enqueues.
    std::vector<std::size_t> queue_before; ///< Owned pre-removal FIFO snapshot, empty when capture_queue=false.
    std::vector<std::size_t> discovered; ///< Newly marked/enqueued vertices from this expansion, each listed once overall.
    std::size_t edges_examined{0}; ///< Adjacency entries inspected here, including duplicates, loops and already-seen endpoints.
    bool reached_goal{false}; ///< This dequeue is the goal, so no successors are inspected.
};
/// @brief Owned BFS result, discovered distances/predecessors and chronological queue events.
/// Distance sentinel n means undiscovered at stopping time, not necessarily
/// unreachable when the goal ended the search early. Predecessor sentinel n is
/// used for start and undiscovered vertices. Discovered vertices may still be
/// queued after the goal is removed, including vertices one layer deeper.
/// A full breadth_first_traversal() has goal_requested=false, empty result.path
/// and result.found=false; result.order then exhausts all reachable vertices.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::breadth_first_search_trace(3,{{0,1},{1,2}},0,2);
/// std::print("{} {}\n",trace.distance,trace.edges_examined); // [0, 1, 2] 2
/// @endcode
/// @see breadth_first_search_trace(), BreadthFirstStep
struct BreadthFirstTrace {
    UnweightedSearchResult result; ///< Single-goal BFS result, or complete dequeue order with no path/target success in full traversal.
    std::vector<std::size_t> distance; ///< Minimum-hop distances for discovered vertices; n for undiscovered vertices.
    std::vector<std::size_t> predecessor; ///< First-discovery parent per vertex; n for start/undiscovered vertices.
    std::vector<BreadthFirstStep> steps; ///< Actual dequeue and expansion records, including the terminal goal dequeue.
    std::vector<std::size_t> layer_counts; ///< Number discovered at each distance, including vertices still pending at termination.
    std::size_t vertices_enqueued{0}; ///< Distinct vertices ever enqueued, including the start.
    std::size_t max_queue_size{0}; ///< Maximum active FIFO length, including after individual enqueue operations.
    std::size_t edges_examined{0}; ///< Sum of actual adjacency inspections, excluding graph construction/validation.
    std::size_t adjacency_entries{0}; ///< Stored adjacency entries: E for directed input, 2E for undirected input, including loops.
    bool directed{true}; ///< Whether each supplied edge contributes only its given orientation.
    bool queues_recorded{false}; ///< Whether steps own full FIFO snapshots rather than only queue lengths.
    bool goal_requested{true}; ///< True for a single-goal search; false when all reachable vertices are traversed.
};
/// @brief Find a minimum-edge start-to-goal path by FIFO breadth-first search.
/// Vertices are 0..n-1. Every edge has unit cost. Mark vertices on enqueue, retain
/// the first predecessor and visit adjacency in input order, so equal-length path
/// ties are deterministic. Loops/parallel edges are accepted and inspected but do
/// not cause repeated enqueues. Undirected input adds both orientations, including
/// two entries for a self-loop. Stop when the goal is dequeued, before expanding it.
/// Start=goal returns that single vertex; an unreachable goal returns an empty path.
/// All endpoints/terminals are validated before adjacency allocation, even when
/// start=goal. O(n+E) work and owned adjacency/work storage; search state alone is
/// O(n). The returned path/order own O(n) storage. No trace records are allocated,
/// inputs are unchanged and independent calls share no state.
/// @param n Number of vertices, including isolated vertices; must admit the terminals.
/// @param edges Borrowed endpoint pairs; order determines adjacency tie behavior.
/// @param start Valid starting vertex in [0,n).
/// @param goal Valid target vertex in [0,n).
/// @param directed True (default) uses supplied arcs; false inserts each edge both ways.
/// @return Owned minimum-hop path, actual dequeue order including goal if reached, and success flag.
/// @throws std::invalid_argument If a terminal or edge endpoint is outside [0,n).
/// @par Example
/// @code
/// const auto result=dm::algorithms::breadth_first_search(3,{{0,1},{1,2}},0,2);
/// std::print("{} {}\n",result.path.size(),result.order); // 3 [0, 1, 2]
/// @endcode
/// @par References
/// - E. F. Moore (1959), The shortest path through a maze, Proceedings of the
///   International Symposium on the Theory of Switching, Part II, 285-292.
/// - C. Y. Lee (1961), An Algorithm for Path Connections and Its Applications,
///   IRE Transactions on Electronic Computers EC-10(3), 346-365, doi:10.1109/TEC.1961.5219222.
///   This implementation uses FIFO traversal of explicit unweighted adjacency lists.
/// @see breadth_first_search_trace(), UnweightedSearchResult
UnweightedSearchResult breadth_first_search(std::size_t n,
    const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal,bool directed=true);
/// @brief Record actual BFS discoveries, predecessors, queue lengths and optional FIFO snapshots.
/// Path/order and stopping semantics match breadth_first_search(). Omitting queue
/// snapshots retains O(n+E) time/storage, including linear event storage. With
/// capture_queue=true, copying the active queue before each dequeue adds O(Q)
/// time/storage, Q=sum of those lengths, potentially O(n^2). Layer counts describe
/// discoveries at stopping time; vertices can remain pending after the goal pop.
/// Records own their data, inputs are unchanged and calls share no mutable state.
/// @param n Number of vertices, including isolates.
/// @param edges Borrowed endpoint pairs in adjacency tie order.
/// @param start Valid starting vertex.
/// @param goal Valid target vertex.
/// @param directed True (default) uses directed arcs; false stores both orientations.
/// @param capture_queue True (default) stores full pre-removal queues; false retains lengths/events only.
/// @return Owned result, discovered state, logical counters and actual dequeue records.
/// @throws std::invalid_argument If a terminal or edge endpoint is invalid, including n=0.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::breadth_first_search_trace(3,{{0,1},{1,2}},0,2,true,false);
/// std::print("{} {}\n",trace.steps.size(),trace.queues_recorded); // 3 false
/// @endcode
/// @par References
/// - C. Y. Lee (1961), doi:10.1109/TEC.1961.5219222; records the same FIFO kernel as ordinary BFS.
/// @see breadth_first_search(), BreadthFirstTrace
BreadthFirstTrace breadth_first_search_trace(std::size_t n,
    const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal,
    bool directed=true,bool capture_queue=true);

/// @brief Traverse all reachable vertices for minimum-hop distances without requesting a goal.
/// Uses the same queue/discovery kernel as breadth_first_search_trace(). All
/// reachable vertices are dequeued and expanded; distance sentinel n now means
/// unreachable. Result order is complete, while result.path is empty and
/// result.found=false because no target was requested; goal_requested=false.
/// Queue snapshots default off, giving O(n+E) work/storage including events and
/// adjacency. Enabling them adds O(Q) copying as in the single-goal trace.
/// All outputs are owned; borrowed inputs are unchanged and calls share no state.
/// @param n Number of vertices, including isolates.
/// @param edges Borrowed endpoint pairs in adjacency tie order.
/// @param start Valid starting vertex in [0,n).
/// @param directed True (default) uses supplied arcs; false inserts both orientations.
/// @param capture_queue False (default) retains only lengths/events; true also records full queues.
/// @return Owned complete distances, predecessors, order, layers and actual queue events.
/// @throws std::invalid_argument If start or any edge endpoint is invalid, including n=0.
/// @par Example
/// @code
/// const auto tree=dm::algorithms::breadth_first_traversal(3,{{0,1}},0);
/// std::print("{} {}\n",tree.distance,tree.layer_counts); // [0, 1, 3] [1, 1]
/// @endcode
/// @par References
/// - C. Y. Lee (1961), doi:10.1109/TEC.1961.5219222; FIFO expansion without an early target stop.
/// @see breadth_first_search_trace(), BreadthFirstTrace
BreadthFirstTrace breadth_first_traversal(std::size_t n,
    const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,
    bool directed=true,bool capture_queue=false);
} // namespace datamunge::algorithms
