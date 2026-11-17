#pragma once
/// @file mark_sweep.hpp
/// @brief Checked object-slot mark/sweep model with actual shared marking and sweep classification evidence.
#include <datamunge/algorithms/object_liveness.hpp>
namespace datamunge::algorithms {
/// @brief Owned survivor/reclaimed identities from the object-slot model; no C++ objects are freed.
/// @par Example
/// @code
/// auto t=dm::algorithms::mark_sweep_trace(dm::algorithms::mark_sweep_demo_graph());
/// std::print("{}\n",t.analysis.result.live.size()); // 3
/// @endcode
/// @par References
/// - McCarthy(1960), Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I, https://www-formal.stanford.edu/jmc/recursive.html; this API classifies ids rather than freeing C++ objects.
/// @see mark_sweep_analyze(), mark_sweep_trace()
struct MarkSweepResult {
  std::vector<char> marked; ///< Actual root-reachable mark per original dense object id.
  std::vector<int>  live;   ///< Actual surviving original ids in increasing heap-slot order.
  std::vector<int>  freed;  ///< Actual unreachable original ids in increasing heap-slot order.
};
/// @brief Actual selected full-validation, live-marking and complete sweep work.
/// @par Example
/// @code
/// auto t=dm::algorithms::mark_sweep_trace(dm::algorithms::mark_sweep_demo_graph());
/// std::print("{}\n",t.analysis.work.scanned); // 8
/// @endcode
/// @par References
/// - McCarthy(1960), Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I, https://www-formal.stanford.edu/jmc/recursive.html; this API classifies ids rather than freeing C++ objects.
/// @see mark_sweep_analyze(), mark_sweep_trace()
struct MarkSweepWork {
  ObjectLivenessWork
      marking; ///< Actual shared canonical full-graph validation and mark-on-push operations.
  std::size_t scanned{0}; ///< Actual sweep slots examined, exactly num_objects.
  std::size_t live{0};    ///< Actual survivor ids appended.
  std::size_t reclaimed{
      0}; ///< Actual unreachable ids appended; a model classification, not a deallocation count.
};
/// @brief Owned actual mark/sweep result and selected same-run work.
/// @par Example
/// @code
/// auto t=dm::algorithms::mark_sweep_trace(dm::algorithms::mark_sweep_demo_graph());
/// std::print("{}\n",t.analysis.result.freed.size()); // 5
/// @endcode
/// @par References
/// - McCarthy(1960), Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I, https://www-formal.stanford.edu/jmc/recursive.html; this API classifies ids rather than freeing C++ objects.
/// @see mark_sweep_analyze(), mark_sweep_trace()
struct MarkSweepAnalysis {
  MarkSweepResult result; ///< Actual original-id survivor/reclaimed partition.
  MarkSweepWork   work;   ///< Actual producing mark and sweep counts.
};
/// @brief One actual complete-heap sweep classification and its append outcome.
/// @par Example
/// @code
/// auto t=dm::algorithms::mark_sweep_trace(dm::algorithms::mark_sweep_demo_graph());
/// std::print("{}\n",t.sweep.front().live_after); // 1
/// @endcode
/// @par References
/// - McCarthy(1960), Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I, https://www-formal.stanford.edu/jmc/recursive.html; this API classifies ids rather than freeing C++ objects.
/// @see mark_sweep_analyze(), mark_sweep_trace()
struct MarkSweepStep {
  int         object;       ///< Actual old object/slot id visited in increasing order.
  bool        live;         ///< Actual final mark read for this object.
  std::size_t live_before;  ///< Actual survivor count before append.
  std::size_t freed_before; ///< Actual reclaimed count before append.
  std::size_t live_after;   ///< Actual survivor count after append.
  std::size_t freed_after;  ///< Actual reclaimed count after append.
};
/// @brief Owned original liveness history and actual sweep records from one collector execution.
/// @par Example
/// @code
/// auto t=dm::algorithms::mark_sweep_trace({3,{{1},{},{}},{0},{}});
/// std::print("{} {}\n",t.analysis.result.live.size(),t.analysis.result.freed.size()); // 2 1
/// @endcode
/// @par References
/// - McCarthy(1960), Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I, https://www-formal.stanford.edu/jmc/recursive.html; this API classifies ids rather than freeing C++ objects.
/// @see mark_sweep_analyze(), mark_sweep_trace()
struct MarkSweepTrace {
  ObjectLivenessTrace
      marking; ///< Actual original graph/root/pop/field evidence and final marks, never rerun.
  std::vector<MarkSweepStep>
      sweep; ///< Every actual old-id sweep classification, including live and unreachable objects.
  MarkSweepAnalysis analysis; ///< Actual same-run survivor/reclaimed partition and selected work.
  std::size_t captured_entries{0}; ///< Actual marking capture plus one record per real sweep slot.
};
/// @brief Mark actual root reachability then classify every original slot without relocating identities or references.
/// @param graph Complete dense object graph; sizes metadata is ignored.
/// @return Owned marks/live/freed vectors; unreachable cycles are classified reclaimable just like other garbage.
/// @throws std::invalid_argument For complete malformed graph structure, including unreachable references.
/// @throws std::length_error For count/container limits.
/// @note O(N+E+R) complete validation/mark/sweep work, O(N) output/stack storage. This model does not free C++ objects,
/// manage an allocator/free list, move payloads, update raw pointers or implement concurrent/incremental collection.
/// @code
/// auto r=dm::algorithms::mark_sweep(dm::algorithms::mark_sweep_demo_graph());
/// std::print("{} {}\n",r.live.size(),r.freed.size()); // 3 5
/// @endcode
MarkSweepResult mark_sweep(const ObjectGraph& graph);
/// @brief Run the same shared marking/sweep producer and retain actual selected work without history.
/// @param graph Complete original dense object graph; size metadata ignored.
/// @return Actual original-id partition and selected validation/traversal/sweep counts.
/// @throws std::invalid_argument For complete structural defects.
/// @throws std::length_error For count/container limits.
MarkSweepAnalysis mark_sweep_analyze(const ObjectGraph& graph);
/// @brief Capture actual canonical marking and complete old-slot sweep, without reachability or phase replay.
/// @param graph Complete original graph, owned once in the marking capture.
/// @param max_entries Optional logical history cap, default one million; zero disables it.
/// @return Actual original graph/mark events, all sweep steps and same-run result/work.
/// @throws std::invalid_argument Before capture on malformed complete graph structure.
/// @throws std::length_error Before the next capture copy exceeds actual logical cap/geometry.
/// @note Counts marking base2*N+E+R+S and its actual events, plus each real sweep record; fields/capacities,
/// returned/runtime mark/result copies and physical bytes are excluded. Complete history or exception.
MarkSweepTrace mark_sweep_trace(const ObjectGraph& graph, std::size_t max_entries = 1000000);
/// @brief Own the original eight-object report graph, including unreachable3-to4 cycle and6-to7 chain.
/// @return Exact original refs{{1},{2},{},{4},{3},{},{7},{}} with root0 and ignored empty sizes.
ObjectGraph mark_sweep_demo_graph();
} // namespace datamunge::algorithms
