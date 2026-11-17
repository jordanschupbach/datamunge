#pragma once
/// @file mark_compact.hpp
/// @brief Checked stable object-slot forwarding with actual shared marking, field rewrites and relocated roots.
#include <datamunge/algorithms/object_liveness.hpp>
namespace datamunge::algorithms {
/// @brief Owned stable object-slot relocation result, preserving the historical public field layout.
/// @par Example
/// @code
/// auto t=dm::algorithms::mark_compact_trace(dm::algorithms::mark_compact_demo_graph());
/// std::print("{}\n",t.analysis.result.new_address[2]); // 1
/// @endcode
/// @par References
/// - Morris(1978), A Time- and Space-Efficient Garbage Compaction Algorithm, https://doi.org/10.1145/359576.359583; this allocating unit-slot graph model is not Morris physical compaction.
/// @see mark_compact_analyze(), mark_compact_trace()
struct MarkCompactResult {
  std::vector<char> marked; ///< Actual final root-reachable marks at old ids.
  std::vector<int>
      new_address; ///< Actual dense new slot per old object, minus1 for unreachable ids.
  std::vector<int>
      order; ///< Actual surviving old ids in increasing old-slot order, indexed by new slot.
  std::vector<std::vector<int>>
      updated_refs; ///< Actual survivor fields translated to new slots, preserving order/duplicates/cycles.
};
/// @brief Actual selected canonical marking and stable forwarding/rewriting work.
/// @par Example
/// @code
/// auto t=dm::algorithms::mark_compact_trace(dm::algorithms::mark_compact_demo_graph());
/// std::print("{}\n",t.analysis.work.scanned); // 10
/// @endcode
/// @par References
/// - Morris(1978), A Time- and Space-Efficient Garbage Compaction Algorithm, https://doi.org/10.1145/359576.359583; this allocating unit-slot graph model is not Morris physical compaction.
/// @see mark_compact_analyze(), mark_compact_trace()
struct MarkCompactWork {
  ObjectLivenessWork
              marking;    ///< Actual same-run full validation and canonical mark-on-push traversal.
  std::size_t scanned{0}; ///< Actual old slots scanned for forwarding, exactly num_objects.
  std::size_t forwarded{0};  ///< Actual surviving objects assigned a new dense slot.
  std::size_t references{0}; ///< Actual original fields read from surviving owners.
  std::size_t rewritten{
      0}; ///< Actual surviving fields translated, one per read because liveness is closed under references.
  std::size_t roots{0}; ///< Actual root slots translated, including duplicates.
};
/// @brief Owned actual stable forwarding result, relocated roots and selected producing work.
/// @par Example
/// @code
/// auto t=dm::algorithms::mark_compact_trace(dm::algorithms::mark_compact_demo_graph());
/// std::print("{}\n",t.analysis.relocated_roots.size()); // 1
/// @endcode
/// @par References
/// - Morris(1978), A Time- and Space-Efficient Garbage Compaction Algorithm, https://doi.org/10.1145/359576.359583; this allocating unit-slot graph model is not Morris physical compaction.
/// @see mark_compact_analyze(), mark_compact_trace()
struct MarkCompactAnalysis {
  MarkCompactResult
      result; ///< Actual historical-layout result with stable dense new slot addresses.
  std::vector<int>
      relocated_roots; ///< Actual original roots translated in original order, preserving duplicates.
  MarkCompactWork work; ///< Actual selected marking/forwarding/field/root operations.
};
/// @brief One actual old-slot forwarding decision, including dead slots.
/// @par Example
/// @code
/// auto t=dm::algorithms::mark_compact_trace(dm::algorithms::mark_compact_demo_graph());
/// std::print("{}\n",t.forwarding.front().assigned); // 0
/// @endcode
/// @par References
/// - Morris(1978), A Time- and Space-Efficient Garbage Compaction Algorithm, https://doi.org/10.1145/359576.359583; this allocating unit-slot graph model is not Morris physical compaction.
/// @see mark_compact_analyze(), mark_compact_trace()
struct MarkCompactForward {
  int         object;      ///< Actual original old slot scanned.
  bool        live;        ///< Actual final mark read.
  std::size_t next_before; ///< Actual next compacted slot before this decision.
  int         assigned;    ///< Actual new slot, minus1 for an unreachable object.
  std::size_t next_after;  ///< Actual next compacted slot after this decision.
};
/// @brief One actual surviving field rewritten through the completed forwarding map.
/// @par Example
/// @code
/// auto t=dm::algorithms::mark_compact_trace(dm::algorithms::mark_compact_demo_graph());
/// std::print("{}\n",t.references.front().target_new); // 1
/// @endcode
/// @par References
/// - Morris(1978), A Time- and Space-Efficient Garbage Compaction Algorithm, https://doi.org/10.1145/359576.359583; this allocating unit-slot graph model is not Morris physical compaction.
/// @see mark_compact_analyze(), mark_compact_trace()
struct MarkCompactReference {
  int         owner_old;  ///< Actual surviving original owner id.
  int         owner_new;  ///< Actual new owner slot.
  std::size_t field;      ///< Actual original field position, including duplicate targets.
  int         target_old; ///< Actual original referenced old id.
  int         target_new; ///< Actual translated target slot.
};
/// @brief One actual original root slot rewritten through the same forwarding map.
/// @par Example
/// @code
/// auto t=dm::algorithms::mark_compact_trace(dm::algorithms::mark_compact_demo_graph());
/// std::print("{}\n",t.roots.front().new_slot); // 0
/// @endcode
/// @par References
/// - Morris(1978), A Time- and Space-Efficient Garbage Compaction Algorithm, https://doi.org/10.1145/359576.359583; this allocating unit-slot graph model is not Morris physical compaction.
/// @see mark_compact_analyze(), mark_compact_trace()
struct MarkCompactRoot {
  std::size_t slot;     ///< Actual original root slot index.
  int         old_id;   ///< Actual original root object id.
  int         new_slot; ///< Actual translated root slot, preserving aliases.
};
/// @brief Owned actual shared marking, complete forwarding scan and real field/root rewrites.
/// @par Example
/// @code
/// auto t=dm::algorithms::mark_compact_trace(dm::algorithms::mark_compact_demo_graph());
/// std::print("{} {}\n",t.analysis.result.order.size(),t.analysis.relocated_roots.front()); // 5 0
/// @endcode
/// @par References
/// - Morris(1978), A Time- and Space-Efficient Garbage Compaction Algorithm, https://doi.org/10.1145/359576.359583; this allocating unit-slot graph model is not Morris physical compaction.
/// @see mark_compact_analyze(), mark_compact_trace()
struct MarkCompactTrace {
  ObjectLivenessTrace
      marking; ///< Complete original graph and actual canonical traversal, with no rerun.
  std::vector<MarkCompactForward>   forwarding; ///< Every actual old-slot forwarding decision.
  std::vector<MarkCompactReference> references; ///< Every actual surviving field translation.
  std::vector<MarkCompactRoot>
                      roots; ///< Every actual original root translation, including duplicate slots.
  MarkCompactAnalysis analysis; ///< Actual same-run stable output, relocated roots and work.
  std::size_t         captured_entries{
      0}; ///< Actual marking history plus all real forward/field/root records.
};
/// @brief Compute stable unit-slot forwarding and updated survivor fields after actual canonical marking.
/// @param graph Complete dense object graph; arbitrary size metadata is ignored, since every object occupies one modeled slot.
/// @return Actual historical-layout dense relocation map/order/updated fields; use analyze() for relocated root slots too.
/// @throws std::invalid_argument For malformed complete graph structure, including unreachable references.
/// @throws std::length_error For count/container limits.
/// @note O(N+E+R) validation/mark/forward/rewrite work and O(N+E_live+R) result/stack storage.
/// This computes a graph/layout model: it does not move C++ payloads, update raw pointers or perform in-place physical garbage collection.
/// @code
/// auto r=dm::algorithms::mark_compact(dm::algorithms::mark_compact_demo_graph());
/// std::print("{} {}\n",r.new_address[2],r.order.size()); // 1 5
/// @endcode
MarkCompactResult mark_compact(const ObjectGraph& graph);
/// @brief Return the same actual stable layout plus all translated roots and selected producing phase work.
/// @param graph Complete original dense graph, unchanged; size metadata ignored.
/// @return Owned actual forwarding, fields, roots and same-run validation/traversal/rewriting work.
/// @throws std::invalid_argument For complete structural defects.
/// @throws std::length_error For count/container limits.
MarkCompactAnalysis mark_compact_analyze(const ObjectGraph& graph);
/// @brief Capture actual marking, complete forwarding and every field/root translation without phase replay.
/// @param graph Complete original graph, owned once inside marking history.
/// @param max_entries Optional whole logical history cap, default one million; zero disables it.
/// @return Owned original graph, actual producer decisions and complete same-run slot result.
/// @throws std::invalid_argument Before capture on malformed complete graph structure.
/// @throws std::length_error Before the next actual record copy exceeds cap/geometry.
/// @note Counts marking base2*N+E+R+S and its actual events, plus each real old-slot forward/live-field/root rewrite record.
/// Excludes scalar fields, row headers, capacities, returned/runtime result/mark copies and physical bytes. Complete history or exception.
MarkCompactTrace mark_compact_trace(const ObjectGraph& graph, std::size_t max_entries = 1000000);
/// @brief Own the original ten-slot even-id live chain, with odd slots unreachable.
/// @return Original0-to2-to4-to6-to8 reference chain with root0 and no size metadata.
ObjectGraph mark_compact_demo_graph();
} // namespace datamunge::algorithms
