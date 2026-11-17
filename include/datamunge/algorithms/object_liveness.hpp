#pragma once
/// @file object_liveness.hpp
/// @brief Shared checked iterative object marking with actual root/pop/reference traversal evidence.
#include <datamunge/algorithms/object_graph.hpp>
namespace datamunge::algorithms {
/// @brief Actual selected complete-validation and live-traversal work, excluding allocation/capture copies.
/// @par Example
/// @code
/// auto t=dm::algorithms::object_liveness_trace({3,{{1},{},{}},{0},{}});
/// std::print("{}\n",t.analysis.work.marks); // 2
/// @endcode
/// @par References
/// - McCarthy(1960), Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I, https://www-formal.stanford.edu/jmc/recursive.html; checked iterative graph marking here.
/// @see object_liveness_analyze(), object_liveness_trace()
struct ObjectLivenessWork {
  ObjectGraphWork
      validation; ///< Actual full-graph guard visits, including unreachable fields; sizes are ignored.
  std::size_t initialized{0}; ///< Actual dense mark slots initialized, one per object.
  std::size_t roots{0};       ///< Actual root slots examined, including duplicates.
  std::size_t pops{0};        ///< Actual marked objects popped for outgoing-field traversal.
  std::size_t references{
      0}; ///< Actual live-owner outgoing field visits, including duplicates/self/cyclic edges.
  std::size_t marks{0};  ///< Actual zero-to-one discoveries, one per live object.
  std::size_t pushes{0}; ///< Actual stack pushes, one per discovery.
  std::size_t maximum_stack{
      0}; ///< Actual largest pending stack size, excluding already popped objects.
};
/// @brief Owned actual root-reachable marks and same-run selected work.
/// @par Example
/// @code
/// auto t=dm::algorithms::object_liveness_trace({3,{{1},{},{}},{0},{}});
/// std::print("{}\n",int(t.analysis.marked[1])); // 1
/// @endcode
/// @par References
/// - McCarthy(1960), Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I, https://www-formal.stanford.edu/jmc/recursive.html; checked iterative graph marking here.
/// @see object_liveness_analyze(), object_liveness_trace()
struct ObjectLivenessAnalysis {
  std::vector<char>  marked; ///< Actual final mark per dense old object id, one iff root reachable.
  ObjectLivenessWork work;   ///< Actual full validation plus iterative traversal operations.
};
/// @brief Actual iterative traversal event, preserving root/field order and LIFO popping.
/// @par Example
/// @code
/// auto kind=dm::algorithms::ObjectMarkEventKind::Root;
/// std::print("{}\n",int(kind)); // 0
/// @endcode
/// @par References
/// - McCarthy(1960), Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I, https://www-formal.stanford.edu/jmc/recursive.html; checked iterative graph marking here.
/// @see object_liveness_analyze(), object_liveness_trace()
enum class ObjectMarkEventKind {
  Root,     ///< Examine an original root slot, possibly discovering/pushing its object.
  Pop,      ///< Pop one pending live object before visiting its fields.
  Reference ///< Examine one actual original field of a popped live object.
};
/// @brief One actual root/pop/field event and its mark/stack consequences.
/// @par Example
/// @code
/// auto t=dm::algorithms::object_liveness_trace({3,{{1},{},{}},{0},{}});
/// std::print("{}\n",t.events.front().pushed); // true
/// @endcode
/// @par References
/// - McCarthy(1960), Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I, https://www-formal.stanford.edu/jmc/recursive.html; checked iterative graph marking here.
/// @see object_liveness_analyze(), object_liveness_trace()
struct ObjectMarkEvent {
  ObjectMarkEventKind kind;   ///< Actual producing event category.
  int                 object; ///< Actual root/field target, or actual popped object for Pop.
  int                 parent; ///< Actual owner for Reference, minus1 for Root/Pop.
  std::size_t         slot;   ///< Actual original root/field slot; SIZE_MAX for Pop.
  bool        already_marked; ///< Actual preevent target mark, always true for a popped object.
  bool        pushed;         ///< Actual newly marked/pushed root or field target; false for Pop.
  std::size_t stack_before;   ///< Actual pending stack size before this event.
  std::size_t stack_after;    ///< Actual pending stack size after this event.
};
/// @brief Owned original graph, actual iterative events and final same-run liveness.
/// @par Example
/// @code
/// auto t=dm::algorithms::object_liveness_trace({3,{{1},{0},{}},{0},{}});
/// std::print("{} {}\n",t.analysis.work.marks,t.analysis.work.references); // 2 2
/// @endcode
/// @par References
/// - McCarthy(1960), Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I, https://www-formal.stanford.edu/jmc/recursive.html; checked iterative graph marking here.
/// @see object_liveness_analyze(), object_liveness_trace()
struct ObjectLivenessTrace {
  ObjectGraph graph; ///< Complete owned original graph, including ignored size metadata.
  std::vector<ObjectMarkEvent>
      events; ///< Actual root/pop/reference records, not closure reconstruction or stack snapshots.
  ObjectLivenessAnalysis analysis; ///< Actual final marks and selected producer work.
  std::size_t captured_entries{0}; ///< Actual2*N+E+R+S base slots plus each real event record.
};
/// @brief Run the canonical mark-on-push iterative root traversal after complete structural validation.
/// @param graph Complete dense object graph; size metadata is ignored by this object-slot liveness model.
/// @return Owned marks and actual validation/traversal work, without history.
/// @throws std::invalid_argument For any complete shape/root/field defect, including unreachable fields.
/// @throws std::length_error For count/container limits.
/// @note Roots are pushed in original order; live fields are pushed in original order and popped LIFO.
/// Duplicate roots/fields, self references, sharing and cycles cause no duplicate pushes. O(N+E+R) total validation/traversal,
/// O(N) mark/stack storage. This models reachability and never frees or moves C++ objects.
/// @code
/// auto a=dm::algorithms::object_liveness_analyze({3,{{1},{0},{}},{0},{}});
/// std::print("{} {} {}\n",int(a.marked[0]),int(a.marked[1]),int(a.marked[2])); // 1 1 0
/// @endcode
ObjectLivenessAnalysis object_liveness_analyze(const ObjectGraph& graph);
/// @brief Capture actual canonical root/pop/field events without a second reachability traversal.
/// @param graph Complete original graph, copied only after complete validation/base capture admission.
/// @param max_entries Optional actual logical capture cap, default one million; zero disables it.
/// @return Owned original graph, actual events and same-run final marks/work.
/// @throws std::invalid_argument For complete graph defects before capture.
/// @throws std::length_error Before input/event copies if actual logical capture count exceeds cap/geometry.
/// @note Base2*N+E+R+S counts original object records, final mark slots, all fields/roots and supplied ignored-size entries;
/// each actual root/pop/live-field event adds one record. Excludes fields within records, row headers, capacities,
/// returned/runtime copies and physical bytes. No stack snapshots, unknown future events or prefix replay are fabricated.
ObjectLivenessTrace object_liveness_trace(
    const ObjectGraph& graph,
    std::size_t        max_entries = 1000000);
} // namespace datamunge::algorithms
