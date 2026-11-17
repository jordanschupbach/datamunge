#pragma once
/// @file kosaraju.hpp
/// @brief Checked iterative two-pass SCC producer with actual discovery/edge/finish/assignment events.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual operation phase of the two-pass SCC producer.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kosaraju_trace(1,{{}});
/// std::print("{}\n",t.events[0].kind==dm::algorithms::SCCEventKind::Discover); // true
/// @endcode
/// @see SCCEvent
enum class SCCEventKind {
  Discover,     ///< First-pass vertex mark before stack insertion.
  FirstEdge,    ///< Actual first-pass original adjacency entry examination.
  Finish,       ///< Actual first-pass stack exit and finish-order append.
  Assign,       ///< Actual second-pass discovery/component assignment, not stack-pop order.
  TransposeEdge ///< Actual second-pass transpose adjacency entry examination.
};
/// @brief Owned actual first-discovery, edge, finish or component-assignment operation.
/// u is the actual vertex/source; v is the actual target only for edge events and
/// -1 otherwise. component is meaningful on Assign/TransposeEdge and-1 otherwise.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kosaraju_trace(1,{{}});
/// std::print("{} {}\n",t.events.back().u,t.events.back().component); // 0 0
/// @endcode
/// @see KosarajuTrace
struct SCCEvent {
  SCCEventKind kind{SCCEventKind::Discover}; ///< Actual producing operation kind.
  int          u{0};          ///< Actual source or discovered/finished/assigned vertex.
  int          v{-1};         ///< Actual examined edge target, otherwise-1.
  int          component{-1}; ///< Actual second-pass component label, otherwise-1.
};
/// @brief Actual selected DFS operations excluding graph validation/transpose construction/copies/time.
/// Every original and transpose adjacency entry is examined once, including
/// duplicates/loops. Each vertex finishes and is assigned once. Root counts are
/// actual forest roots; second_roots equals the number of SCCs.
/// @par Example
/// @code
/// const auto a=dm::algorithms::kosaraju_analyze(2,{{1},{}});
/// std::print("{} {}\n",a.work.first_edge_tests,a.work.second_roots); // 1 2
/// @endcode
/// @see kosaraju_analyze()
struct KosarajuWork {
  std::size_t first_roots{0};      ///< Actual first-pass DFS roots, ascending original start order.
  std::size_t first_edge_tests{0}; ///< Actual original adjacency entries examined.
  std::size_t finishes{0};         ///< Actual complete first-pass vertex exits.
  std::size_t second_roots{0}; ///< Actual transpose forest roots/components, reverse finish order.
  std::size_t second_edge_tests{0}; ///< Actual transpose adjacency entries examined.
  std::size_t assignments{0};       ///< Actual second-pass first component labels assigned.
};
/// @brief Owned component labels and actual same-execution selected work without history.
/// Labels are0..c-1 in reverse-finish transpose-root order, not minimum-vertex ids.
/// Cross-component original edges go from a lower to a higher returned label.
/// @par Example
/// @code
/// const auto a=dm::algorithms::kosaraju_analyze(2,{{1},{0}});
/// std::print("{} {}\n",a.component[0],a.component[1]); // 0 0
/// @endcode
/// @see kosaraju_trace()
struct KosarajuAnalysis {
  std::vector<int> component; ///< Owned actual label per original vertex, empty for n0.
  KosarajuWork     work;      ///< Actual selected producing operations.
};
/// @brief Owned original graph, actual transpose/finish order/labels and chronological events.
/// No second DFS/replay. Capture quota counts complete original n vertex records,
/// E adjacency records and each actual event. Excludes actual transpose/order/
/// labels/output copies, individual event fields, capacities and bytes/RAM.
/// Complete original count admitted before copies, each event before its producing
/// mutation/examination. Default million, zero off, no partial result on failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kosaraju_trace(2,{{1},{0}});
/// std::print("{} {}\n",t.finish_order[0],t.component[1]); // 1 0
/// @endcode
/// @see kosaraju_trace()
struct KosarajuTrace {
  std::vector<std::vector<int>>
      adjacency; ///< Owned original rows/entry order, duplicates retained.
  std::vector<std::vector<int>>
                   transpose;    ///< Actual producing transpose, built by original source order.
  std::vector<int> finish_order; ///< Actual first-pass exits, earliest first.
  std::vector<int> component;    ///< Actual returned second-pass labels.
  KosarajuWork     work;         ///< Actual same-execution selected operations.
  std::vector<SCCEvent>
              events; ///< Actual chronological operations, first pass then transpose pass.
  std::size_t captured_entries{0}; ///< Original n+E records plus actual event count.
};
/// @brief Partition a finite directed multigraph into strongly connected components by two DFS passes.
/// Require n>=0, exactly n adjacency rows and every target in[0,n), validated
/// before vertex access. Empty graph returns empty. Parallel edges/loops allowed.
/// First DFS uses ascending roots and caller adjacency order, recording exits;
/// transpose rows follow original source order. Second iterative DFS follows
/// reverse finish order and original stack push order, assigning on discovery.
/// Preserves historical valid-input labels and member discovery order. O(n+E)
/// time/storage, no recursion/input mutation or retained input; calls independent.
/// Labels order the condensation DAG topologically, but arbitrary label equality
/// alone is not an independent certificate of mutual reachability.
/// @param n Nonnegative original vertex count.
/// @param adjacency Borrowed complete directed adjacency list, original order retained.
/// @return Owned actual component labels0..c-1, one per original vertex.
/// @throws std::invalid_argument On negative n, row mismatch or invalid endpoint.
/// @throws std::length_error On container/count representability limits.
/// @throws std::bad_alloc On transpose/DFS/output allocation failure.
/// @par Example
/// @code
/// const auto c=dm::algorithms::kosaraju_scc(3,{{1},{0,2},{}});
/// std::print("{} {} {}\n",c[0],c[1],c[2]); // 0 0 1
/// @endcode
/// @par References
/// - M. Sharir(1981), A strong-connectivity algorithm and its applications in data flow analysis, Computers and Mathematics with Applications7(1),67–72, https://doi.org/10.1016/0898-1221(81)90008-0.
/// @see kosaraju_analyze(), kosaraju_trace()
std::vector<int> kosaraju_scc(int n, const std::vector<std::vector<int>>& adjacency);
/// @brief Return actual component labels and selected producing work from the shared two-pass kernel.
/// Same checked input, deterministic ordering and O(n+E) contracts as kosaraju_scc().
/// No original graph/history capture or optional capture quota.
/// @param n Nonnegative original vertex count.
/// @param adjacency Borrowed complete directed adjacency list.
/// @return Owned labels and selected actual work.
/// @throws std::invalid_argument On invalid n/shape/endpoint.
/// @throws std::length_error On count/container limits.
/// @throws std::bad_alloc On native DFS/transpose/output allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::kosaraju_analyze(0,{}).work.second_roots); // 0
/// @endcode
/// @par References
/// - Sharir(1981), A strong-connectivity algorithm and its applications in data flow analysis, https://doi.org/10.1016/0898-1221(81)90008-0.
/// @see kosaraju_scc(), KosarajuWork
KosarajuAnalysis kosaraju_analyze(int n, const std::vector<std::vector<int>>& adjacency);
/// @brief Capture original graph and actual producing DFS events, transpose, exits and labels once.
/// Same numerical-free deterministic kernel as kosaraju_scc(); optional capture
/// admission follows KosarajuTrace and adds owned O(n+E+events) history/copies.
/// @param n Nonnegative original vertex count.
/// @param adjacency Borrowed complete original directed adjacency list.
/// @param max_entries Original n+E plus actual-event ceiling, default1000000, zero off.
/// @return Owned complete actual observations, never prefix replay or partial output.
/// @throws std::invalid_argument On invalid n/shape/endpoint before copying.
/// @throws std::length_error On enabled capture ceiling or count/container limits.
/// @throws std::bad_alloc On native/capture allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kosaraju_trace(1,{{}},4);
/// std::print("{} {}\n",t.events.size(),t.captured_entries); // 3 4
/// @endcode
/// @par References
/// - Sharir(1981), A strong-connectivity algorithm and its applications in data flow analysis, https://doi.org/10.1016/0898-1221(81)90008-0.
/// @see KosarajuTrace, kosaraju_analyze()
KosarajuTrace kosaraju_trace(
    int                                  n,
    const std::vector<std::vector<int>>& adjacency,
    std::size_t                          max_entries = 1000000);
} // namespace datamunge::algorithms
