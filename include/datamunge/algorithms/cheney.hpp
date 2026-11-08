#pragma once
/// @file cheney.hpp
/// @brief Checked breadth-first object evacuation, rewritten references and actual scan/free traces.
#include <datamunge/algorithms/object_graph.hpp>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned breadth-first old-id order and dense old-to-new slot map, retaining the legacy layout.
/// Each reachable object occurs once, ordered by first discovery from the ordered
/// roots and reference fields. Unreachable old ids have forwarding=-1. Slots are
/// object indices, not byte addresses; sizes are ignored by cheney_copy().
/// @par Example
/// @code
/// const auto r=dm::algorithms::cheney_copy({3,{{},{0},{1}},{2},{}});
/// std::print("{} {}\n",r.to_space_order,r.forwarding); // [2, 1, 0] [2, 1, 0]
/// @endcode
/// @see cheney_copy(), CheneyCopy
struct CheneyResult {
  std::vector<int> to_space_order; ///< Old ids in new slot order; the append-only traversal queue.
  std::vector<int> forwarding; ///< forwarding[old]=new slot, or -1 when unreachable.
};
/// @brief Actual validation, forward requests, completed scans and forwarding outcomes.
/// root_probes includes duplicate root slots; reference_probes includes every
/// live field, including repeated/shared/cyclic edges. Each forward request is
/// either one copy or one reuse, so root_probes+reference_probes=copies+reuses.
/// Validation visits the entire input, including garbage fields and supplied sizes.
/// @par Example
/// @code
/// const auto c=dm::algorithms::cheney_evacuate({2,{{1},{0}},{0,0},{}},2);
/// std::print("{} {} {}\n",c.work.copies,c.work.reference_probes,c.work.reuses); // 2 2 2
/// @endcode
/// @see CheneyCopy, cheney_evacuate()
struct CheneyWork {
  ObjectGraphWork validation; ///< Actual checks, including unreachable graph rows/edges.
  std::size_t root_probes{0}; ///< Actual root forward requests, retaining duplicates.
  std::size_t reference_probes{0}; ///< Actual forward requests from scanned live fields.
  std::size_t scans{0}; ///< Completed to-space object scans, equal to copies on success.
  std::size_t copies{0}; ///< First discoveries that append one object to to-space.
  std::size_t reuses{0}; ///< Requests resolved through an already-installed forwarding slot.
};
/// @brief Owned symbolic evacuated heap with rewritten slots and packed integral size offsets.
/// heap ids are new slots; its roots and fields are rewritten during the actual
/// traversal, preserving duplicate fields, duplicate roots, cycles and sharing.
/// heap.sizes contains one positive size per survivor, including default ones.
/// offsets[new] is the prefix sum of those sizes, in caller-chosen units, without
/// alignment/header padding. This models layout and references, not raw allocation,
/// object payload copying, deallocation, mutator suspension or an arena swap.
/// @par Example
/// @code
/// const auto c=dm::algorithms::cheney_evacuate({3,{{},{0},{1}},{2},{2,3,4}},9);
/// std::print("{} {}\n",c.offsets,c.used_units); // [0, 4, 7] 9
/// @endcode
/// @see cheney_evacuate(), CheneyResult, CheneyWork
struct CheneyCopy {
  CheneyResult relocation; ///< Legacy-compatible order and dense forwarding map.
  ObjectGraph heap; ///< Owned reachable graph, with every root and field expressed in new slots.
  std::vector<std::size_t> offsets; ///< Packed prefix offsets in new slot order.
  std::size_t used_units{0}; ///< Sum of survivor sizes, checked against the requested capacity.
  CheneyWork work; ///< Actual scalar work counts; ordinary calls keep no event records.
};
/// @brief Kind of actual chronological event in a symbolic Cheney evacuation.
/// Root/reference events describe a forward request; scan events close one
/// object scan after all its reference events have been recorded.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cheney_copy_trace({1,{{0}},{0},{}},1);
/// std::print("{}\n",t.events.front().phase==dm::algorithms::CheneyPhase::Root); // true
/// @endcode
/// @see CheneyEvent
enum class CheneyPhase {
  Root, ///< Root slot forwarded before scanning starts.
  Reference, ///< Field of the currently scanned copied object forwarded.
  Scan ///< Current object scan completed; the scan index advances after this record.
};
/// @brief Owned actual forward or scan-completion record, without replay or queue snapshots.
/// For Root, source=-1 and field is the root-slot index. For Reference, source is
/// the scanned old id and field its original reference-slot index. For Scan,
/// source=target is the scanned old id, field=0, destination=scan, copied=false.
/// scan is the current object-slot head; completed scans cover [0,scan) before
/// the event. free_before/free_after count allocated object slots. At scan events
/// they span that entire object's field processing; at forwards they span only
/// that request. used_before/after measure packed units with the same convention.
/// After Scan, [scan+1,free_after) is the pending queue. Records own all values.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cheney_copy_trace({2,{{1},{}},{0},{}},2);
/// std::print("{} {} {}\n",t.events[1].target,t.events[1].destination,t.events[1].copied); // 1 1 true
/// @endcode
/// @see CheneyTrace, CheneyPhase
struct CheneyEvent {
  CheneyPhase phase{CheneyPhase::Root}; ///< Actual request or scan-completion kind.
  int source{-1}; ///< Old source id, or -1 for a root slot.
  std::size_t field{0}; ///< Ordered root/reference slot index; zero for scan records.
  int target{-1}; ///< Old target id; scanned old id for scan records.
  int destination{-1}; ///< Actual forwarding slot, or current scan slot for scan records.
  bool copied{false}; ///< Whether this forward created a new copy; false for scans/reuses.
  std::size_t scan{0}; ///< Current to-space object index before advancing the scan head.
  std::size_t free_before{0}; ///< Allocated slots before this request or the entire object scan.
  std::size_t free_after{0}; ///< Allocated slots after this request or the entire object scan.
  std::size_t used_before{0}; ///< Packed units before this request or the entire object scan.
  std::size_t used_after{0}; ///< Packed units after this request or the entire object scan.
};
/// @brief Owned evacuated heap plus every actual forward and scan-completion event.
/// Uses the ordinary kernel; no discovery, forwarding or rewriting is replayed.
/// There are R+E_live+L events on success. Input graphs are unchanged, and copied
/// trace records can be modified independently. Trace memory is additional to
/// the dense forwarding table and evacuated graph; it is not constant workspace.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cheney_copy_trace({2,{{1},{}},{0},{}},2);
/// std::print("{} {}\n",t.events.size(),t.copy.work.scans); // 4 2
/// @endcode
/// @see cheney_copy_trace(), CheneyEvent
struct CheneyTrace {
  CheneyCopy copy; ///< Completed actual heap, relocation, packed offsets and work counts.
  std::vector<CheneyEvent> events; ///< Actual chronological root/field/scan records.
};
/// @brief Private shared allocation-free counters and ordinary/traced evacuation kernel.
namespace cheney_detail {
/// @brief Increment one size_t work counter, rejecting overflow before mutation.
/// @param value Counter to increment.
/// @throws std::length_error If the counter is already SIZE_MAX.
inline void increment(std::size_t& value) {
  if(value==std::numeric_limits<std::size_t>::max())throw std::length_error("cheney: work counter overflow");
  ++value;
}
/// @brief Validate and execute one append-only object-slot evacuation.
/// @param graph Input old-id graph, unchanged.
/// @param materialize Whether to validate sizes and construct a rewritten heap/offsets.
/// @param capacity Maximum packed units; SIZE_MAX means the representable limit.
/// @param events Optional destination for actual event records.
/// @param max_events Positive record cap, or zero for no explicit cap.
/// @return Complete local result, with no externally visible partial heap on failure.
/// @throws std::invalid_argument If checked input structure or size metadata is invalid.
/// @throws std::length_error If capacity, event budget or vector/counter range is exceeded.
/// @throws std::bad_alloc If local result allocation fails.
inline CheneyCopy run(const ObjectGraph& graph,bool materialize,std::size_t capacity,
                      std::vector<CheneyEvent>* events,std::size_t max_events) {
  CheneyCopy result;result.work.validation=validate_object_graph(graph,materialize);
  result.relocation.forwarding.assign(static_cast<std::size_t>(graph.num_objects),-1);
  auto& to=result.relocation.to_space_order;
  /// @brief Reject the next event before its action when the explicit retained-record cap is full.
  /// @throws std::length_error If the next retained event would exceed the explicit cap.
  const auto check_event=[&] {
    if(events && max_events && events->size()==max_events)throw std::length_error("cheney: event budget exhausted");
  };
  /// @brief Forward one actual root/field, appending only on its first discovery.
  /// All references to growing output vectors are reacquired after possible reallocations.
  /// @param old Valid old target id.
  /// @param phase Root or Reference request kind.
  /// @param source Old scanned source id, or -1 for roots.
  /// @param field Ordered field/root-slot index.
  /// @param scan Current to-space scan head.
  /// @return Actual forwarding destination slot.
  /// @throws std::length_error If capacity or an event/vector/counter limit is reached.
  /// @throws std::bad_alloc If local metadata/event allocation fails.
  const auto forward=[&](int old,CheneyPhase phase,int source,std::size_t field,std::size_t scan) {
    check_event();
    const auto before=to.size(),units_before=result.used_units;
    int destination=result.relocation.forwarding[old];const bool fresh=destination<0;
    if(fresh) {
      const auto size=materialize?static_cast<std::size_t>(object_size(graph,old)):std::size_t{1};
      if(size>capacity-result.used_units)throw std::length_error("cheney: survivor size exceeds to-space capacity");
      destination=static_cast<int>(to.size());to.push_back(old);
      if(materialize) {
        result.heap.refs.emplace_back();result.heap.sizes.push_back(static_cast<int>(size));
        result.offsets.push_back(result.used_units);++result.heap.num_objects;
      }
      result.used_units+=size;result.relocation.forwarding[old]=destination;increment(result.work.copies);
    } else increment(result.work.reuses);
    if(phase==CheneyPhase::Root)increment(result.work.root_probes);else increment(result.work.reference_probes);
    if(events)events->push_back({phase,source,field,old,destination,fresh,scan,before,to.size(),units_before,result.used_units});
    return destination;
  };
  for(std::size_t field=0;field<graph.roots.size();++field) {
    const auto destination=forward(graph.roots[field],CheneyPhase::Root,-1,field,0);
    if(materialize)result.heap.roots.push_back(destination);
  }
  for(std::size_t scan=0;scan<to.size();++scan) {
    const int old=to[scan];const auto free_before=to.size(),units_before=result.used_units;
    for(std::size_t field=0;field<graph.refs[old].size();++field) {
      const auto destination=forward(graph.refs[old][field],CheneyPhase::Reference,old,field,scan);
      if(materialize)result.heap.refs[scan].push_back(destination);
    }
    check_event();increment(result.work.scans);
    if(events)events->push_back({CheneyPhase::Scan,old,0,old,static_cast<int>(scan),false,scan,free_before,to.size(),units_before,result.used_units});
  }
  return result;
}
} // namespace cheney_detail
/// @brief Compute the legacy checked breadth-first forwarding layout in object slots.
/// Keeps the original call shape, inline availability and CheneyResult layout.
/// Validates every graph row/id before copying; sizes is entirely ignored.
/// Roots are forwarded in order, then fields in order while scan chases the
/// append-only to-space vector. No separate traversal queue or recursion is used.
/// Empty roots copy nothing; cycles/shared/duplicate edges copy each live object
/// once. Does not rewrite roots/fields, copy payload bytes or reclaim memory.
/// Total work is O(N+E+R), including dense map initialisation and full validation;
/// O(N+L) owned storage. This graph model does not provide constant auxiliary space.
/// @param graph Valid old-id graph with exact row count, unchanged on success/failure.
/// @return Owned old-id copy order and old-to-new slot map.
/// @throws std::invalid_argument If any graph id or adjacency shape is invalid.
/// @throws std::length_error If a vector/counter range is exceeded.
/// @throws std::bad_alloc If output allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::cheney_copy({3,{{},{0},{1}},{2,2},{}});
/// std::print("{}\n",r.to_space_order); // [2, 1, 0]
/// @endcode
/// @par References
/// - C. J. Cheney (1970), A Nonrecursive List Compacting Algorithm,
///   Communications of the ACM 13(11), 677-678, doi:10.1145/362790.362798.
///   This generic object-graph model uses a side forwarding table, not overwritten list cells.
/// @see cheney_evacuate(), cheney_copy_trace(), validate_object_graph()
inline CheneyResult cheney_copy(const ObjectGraph& graph) {
  return cheney_detail::run(graph,false,std::numeric_limits<std::size_t>::max(),nullptr,0).relocation;
}
/// @brief Evacuate a symbolic heap, rewriting all live roots/fields and packing positive sizes.
/// Same discovery order as cheney_copy(), with full size-prefix validation and
/// default size one for omitted entries. Checks each first copy against capacity
/// before adding units; exactly full heaps succeed, zero capacity allows no live
/// objects. No input mutation or partial result on any failure. Ordinary calls
/// retain scalar work counts but allocate no provenance records. Validation plus
/// copying costs O(N+E+R); owns O(N+L+E_live+R) metadata. No alignment, payload
/// copying, allocator lifecycle or mutator protocol is simulated.
/// @param graph Valid old-id graph with positive size prefix, unchanged.
/// @param capacity Maximum total survivor size units, default SIZE_MAX; zero is a strict zero capacity.
/// @return Owned rewritten graph, slot forwarding, packed offsets and actual counts.
/// @throws std::invalid_argument If graph structure/ids or supplied size metadata is invalid.
/// @throws std::length_error If survivor capacity or a vector/counter range is exceeded.
/// @throws std::bad_alloc If local result allocation fails.
/// @par Example
/// @code
/// const auto c=dm::algorithms::cheney_evacuate({3,{{},{0},{1}},{2,2},{2,3,4}},9);
/// std::print("{} {}\n",c.heap.roots,c.heap.refs[0]); // [0, 0] [1]
/// @endcode
/// @par References
/// - C. J. Cheney (1970), doi:10.1145/362790.362798; symbolic graph/size adaptation here.
/// @see cheney_copy(), cheney_copy_trace(), CheneyCopy
inline CheneyCopy cheney_evacuate(const ObjectGraph& graph,
    std::size_t capacity=std::numeric_limits<std::size_t>::max()) {
  return cheney_detail::run(graph,true,capacity,nullptr,0);
}
/// @brief Retain every actual root/field forwarding and scan completion from the shared evacuation.
/// Same checked capacity and rewritten-heap semantics as cheney_evacuate(). A
/// positive max_events caps the total retained records (R+E_live+L); zero adds
/// no explicit cap. Reaching the cap throws before the next corresponding action
/// or completed-scan record. Failed calls expose no partial heap or trace and leave
/// input unchanged. Event storage adds O(R+E_live+L), with no queue snapshot copies.
/// @param graph Valid old-id graph with positive size prefix, unchanged.
/// @param capacity Maximum survivor size units, default SIZE_MAX; zero is strict.
/// @param max_events Positive whole-call event cap, default zero for no explicit cap.
/// @return Owned evacuated graph and actual chronological event records, without replay.
/// @throws std::invalid_argument If graph structure/ids or supplied size metadata is invalid.
/// @throws std::length_error If capacity, event cap or vector/counter range is exceeded.
/// @throws std::bad_alloc If local result/event allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cheney_copy_trace({2,{{1},{}},{0},{}},2,4);
/// std::print("{} {}\n",t.events.size(),t.copy.heap.num_objects); // 4 2
/// @endcode
/// @par References
/// - C. J. Cheney (1970), doi:10.1145/362790.362798; actual-event instrumentation here.
/// @see cheney_evacuate(), CheneyTrace, CheneyEvent
inline CheneyTrace cheney_copy_trace(const ObjectGraph& graph,
    std::size_t capacity=std::numeric_limits<std::size_t>::max(),std::size_t max_events=0) {
  CheneyTrace result;result.copy=cheney_detail::run(graph,true,capacity,&result.events,max_events);return result;
}
} // namespace datamunge::algorithms
