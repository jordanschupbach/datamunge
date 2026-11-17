#pragma once
/// @file reference_counting.hpp
/// @brief Checked reference-count simulation, actual cascading releases and cycle-held leaks.
#include <cstddef>
#include <datamunge/algorithms/object_graph.hpp>
#include <vector>
/// @brief Canonical owned heap simulations and records.
namespace datamunge::algorithms {
/// @brief Legacy initial-count/reclaimed/leaked field layout.
/// @par Example
/// @code
/// const dm::algorithms::RefCountResult value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RefCountResult {
  std::vector<int>  refcount; ///< Initial incoming root+field counts before reclamation.
  std::vector<char> freed;    ///< Actual reclaimed mark per object.
  std::vector<int>  leaked;   ///< Unreachable unreclaimed objects, including tails held by cycles.
};
/// @brief Actual reference-count mutation kind.
/// @par Example
/// @code
/// const auto value=dm::algorithms::RefCountEventKind::Root;
/// std::print("{}\n",value==dm::algorithms::RefCountEventKind::Root); // true
/// @endcode
enum class RefCountEventKind {
  Root,   ///< One actual root-reference increment.
  Field,  ///< One actual object-field increment.
  Free,   ///< One actual zero-count object reclamation.
  Release ///< One actual outgoing-reference decrement from a reclaimed object.
};
/// @brief One actual count mutation or object reclamation.
/// @par Example
/// @code
/// const dm::algorithms::RefCountEvent value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RefCountEvent {
  RefCountEventKind kind{};     ///< Actual mutation type.
  int               source{-1}; ///< Referring/reclaimed object; -1 for a root reference.
  int               target{-1}; ///< Object whose count is changed; same as source for Free.
  int               before{};   ///< Actual target count before this event.
  int               after{};    ///< Actual target count after this event.
  std::size_t       pending{};  ///< Actual LIFO work-stack length immediately after this event.
};
/// @brief Input/work/record policies; zero disables the corresponding cap.
/// @par Example
/// @code
/// const dm::algorithms::RefCountOptions value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RefCountOptions {
  std::size_t max_objects{1000000}; ///< Complete object-count allocation policy.
  std::size_t max_operations{
      10000000}; ///< Selected count updates, zero scans, pops and reachability-result scans.
  std::size_t max_trace_entries{1000000}; ///< Actual mutation records retained.
};
/// @brief Owned simulation outcome and actual selected work.
/// @par Example
/// @code
/// const dm::algorithms::RefCountRun value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RefCountRun {
  RefCountResult             result;       ///< Initial counts and actual reclamation/leak outcome.
  std::vector<int>           final_counts; ///< Actual counts after the cascade.
  std::vector<int>           freed_order;  ///< Actual LIFO reclamation order.
  std::vector<RefCountEvent> events;       ///< Actual optional initialization/release history.
  std::size_t                increments{}; ///< Actual root/field count increments.
  std::size_t                decrements{}; ///< Actual outgoing-reference decrements.
  std::size_t
      operations{}; ///< Selected count updates, zero scans, pops and final reachability-result scans.
};
/// @brief Simulate checked reference counting with the historical LIFO cascade order.
/// Validates the complete ObjectGraph before allocation, including unreachable references and roots, using its canonical strict nonnegative-id contract. Duplicate roots/fields and self/cyclic references are permitted; size metadata is unused. Counts every root and field, frees zero-count objects in LIFO order and decrements their outgoing references. Separately uses canonical root reachability to classify unreclaimed garbage. Leaked includes cycle-held descendants as well as cycle vertices; it is not a strong-component enumeration. O(V+E+roots) time/storage apart from optional history and canonical validation/reachability overhead. Caller graph unchanged; no C++ object lifetimes or real memory are managed.
/// @param graph Complete valid owned heap graph, borrowed unchanged.
/// @param options Input, selected-work and history policies.
/// @return Initial/final counts, actual free order, leak marks and selected work.
/// @throws std::invalid_argument Any graph shape/root/field id invalid.
/// @throws std::length_error Input/work/count/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::reference_counting_run({2,{{1},{}},{},{}});
/// std::print("{}\n",r.freed_order); // [0, 1]
/// @endcode
/// @see reference_counting_trace(), reachable_from_roots()
RefCountRun reference_counting_run(const ObjectGraph& graph, const RefCountOptions& options = {});
/// @brief Capture actual root/field increments, frees and outgoing decrements.
/// Each actual mutation record costs one history entry, admitted before its count change. The pending length is the real work stack after the event, including a child pushed by its final release. Initial increment records have pending0 because the zero queue has not yet been constructed. Other semantics and complexity match reference_counting_run(), plus O(actual events) owned history.
/// @param graph Complete valid heap graph.
/// @param options Input, selected-work and retained-event policies.
/// @return Actual count mutations, reclamation order and final leak classification.
/// @throws std::invalid_argument Invalid complete graph.
/// @throws std::length_error Any execution/history/count capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::reference_counting_trace({1,{{}},{},{}});
/// std::print("{}\n",r.events.size()); // 1
/// @endcode
/// @see reference_counting_run()
RefCountRun reference_counting_trace(const ObjectGraph& graph, const RefCountOptions& options = {});
/// @brief Return the legacy-layout reference-count result through the shared checked producer.
/// Preserves initial count semantics and LIFO reclamation order on valid graphs. Invalid ids, including previously ignored negative roots, now reject under the canonical ObjectGraph contract. Default input/work policies apply; use reference_counting_run() to configure them.
/// @param graph Complete valid graph, borrowed unchanged.
/// @return Initial counts, reclaimed marks and all unreachable unreclaimed object ids.
/// @throws std::invalid_argument Invalid graph.
/// @throws std::length_error Default input/work/count policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::reference_counting_collect({2,{{1},{0}},{},{}});
/// std::print("{}\n",r.leaked); // [0, 1]
/// @endcode
/// @see reference_counting_run()
RefCountResult reference_counting_collect(const ObjectGraph& graph);
/// @brief Construct a native ordered reference chain for controlled cascade experiments.
/// Creates edges i to i+1 and optionally a root to0. Empty input creates no root. All objects have default unit size metadata. O(n) time and owned graph storage.
/// @param count Objects to create, no greater than INT_MAX.
/// @param rooted Retain a root to the first object, default false.
/// @param max_objects Allocation policy, default1000000; zero disables.
/// @return Owned checked-shape chain graph.
/// @throws std::length_error Allocation or representable object-id policy exceeded.
/// @par Example
/// @code
/// auto g=dm::algorithms::reference_count_chain(3);
/// std::print("{}\n",g.refs[1]); // [2]
/// @endcode
/// @see reference_counting_run()
ObjectGraph
reference_count_chain(std::size_t count, bool rooted = false, std::size_t max_objects = 1000000);
} // namespace datamunge::algorithms
