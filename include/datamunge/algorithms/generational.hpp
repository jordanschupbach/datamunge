#pragma once

/// @file generational.hpp
/// @brief Checked generational graph analysis, actual tracing and a maintained old-to-young slot model.
#include <datamunge/algorithms/object_graph.hpp>
#include <set>
#include <string>
#include <utility>

namespace datamunge::algorithms {

/// @brief Owned legacy minor-analysis flags, candidates and full-trace comparison.
/// promoted[i] names a retained young object proposed for promotion; young_freed
/// names unretained young objects. No real objects, ages or references are changed.
/// All old objects conservatively supply boundary references, including unreachable
/// old garbage, so promoted need not imply full-root reachability. Counts are
/// object visits, not total collector work. Legacy field order/types are stable.
/// @par Example
/// @code
/// const auto r=dm::algorithms::minor_gc({2,{{1},{}},{0},{}},{1,0});
/// std::cout << r.minor_scanned << ' ' << r.full_scanned; // 1 2
/// @endcode
/// @see minor_gc_run(), GenerationalWork
struct GenerationalResult {
    std::vector<char> promoted; ///< Dense retained-young flags, zero for old/unretained objects.
    std::vector<int> young_freed; ///< Ascending unretained young ids; candidates, not actual deallocation.
    std::vector<int> remembered_set; ///< Ascending old ids with at least one current young reference.
    int minor_scanned; ///< Distinct retained young objects popped by the minor DFS.
    int full_scanned; ///< Distinct root-reachable objects popped by the separate full DFS.
};

/// @brief Actual selected validation, discovery, tracing and output-loop work.
/// Counts include stored duplicate fields/roots. They are visits, not timings or
/// a complete instruction count; allocation, container internals, copies and
/// trace capture are excluded. Full-comparison fields are zero for young-only
/// collection. preparation_work() records construction only, not later writes.
/// @par Example
/// @code
/// const auto r=dm::algorithms::minor_gc_run({2,{{1},{}},{0},{}},{1,0});
/// std::cout << r.work.old_fields << ' ' << r.work.young_objects; // 1 1
/// @endcode
/// @see GenerationalRun, YoungCollection
struct GenerationalWork {
    ObjectGraphWork validation; ///< Actual whole-graph checks; zero in a prepared heap's collect().
    std::size_t ages_checked{}; ///< Binary generation entries checked before any use.
    std::size_t discovery_objects{}; ///< Object age/index tests when discovering old boundary slots.
    std::size_t root_checks{}; ///< Actual root slots tested for young-generation membership.
    std::size_t old_fields{}; ///< Stored fields of old sources examined during discovery.
    std::size_t remembered_fields{}; ///< Maintained old-to-young slots enumerated in prepared collect().
    std::size_t seed_candidates{}; ///< Young-root/boundary target candidates including duplicates.
    std::size_t seed_marks{}; ///< Candidates newly marked before the DFS begins.
    std::size_t young_objects{}; ///< Distinct young objects popped from the explicit stack.
    std::size_t young_fields{}; ///< Every field of a popped young object, including old targets.
    std::size_t classified_young{}; ///< Young ids classified as retained or unretained, in ascending order.
    std::size_t dense_result_slots{}; ///< Dense legacy promotion output slots initialized, zero in sparse collect().
    std::size_t full_roots{}; ///< Full-comparison root slots tested, including duplicates.
    std::size_t full_objects{}; ///< Distinct objects popped by the full comparison.
    std::size_t full_fields{}; ///< Fields of popped full-comparison objects.
    std::size_t full_result_slots{}; ///< Dense full marks inspected to count the comparison result.
    std::size_t peak_young_stack{}; ///< Largest actual stack of distinct young ids.
};

/// @brief Owned legacy result, actual full marks and work from one executing analysis.
/// Full marks distinguish true root reachability from conservative minor retention.
/// Same values as minor_gc; includes boundary discovery and a full comparison,
/// O(N+E+R) work/storage bounds, with no caller mutation or hidden replay.
/// @par Example
/// @code
/// const auto r=dm::algorithms::minor_gc_run({2,{{1},{}},{},{}},{1,0});
/// std::cout << r.result.minor_scanned << ' ' << r.result.full_scanned << ' ' << int(r.full_reachable[1]); // 1 0 0
/// @endcode
/// @see minor_gc_trace()
struct GenerationalRun {
    GenerationalResult result; ///< Owned legacy flags/lists/counts, including full comparison.
    std::vector<char> full_reachable; ///< Actual full-root marks; independent owned vector.
    GenerationalWork work; ///< Actual work from this execution, no diagnostic replay.
};

/// @brief One primitive collector event with actual object, decision and stack size.
/// kind is root/old_field/remember/seed/mark/visit/young_field/promote/free or
/// full_root/full_mark/full_visit/full_field. object is source/current id;
/// target is the referenced id or the same id, except remember uses -1.
/// accepted records young membership for root/old_field, a newly marked candidate
/// for seed/young_field/full_root/full_field, false for free, true otherwise.
/// stack_size is actual current size (after mark-push or visit-pop); discovery
/// and classification events use zero. No per-event vector or path is replayed.
/// @par Example
/// @code
/// const auto t=dm::algorithms::minor_gc_trace({1,{{}},{0},{}},{0});
/// std::cout << t.events[0].kind << ' ' << t.events[0].accepted; // root 1
/// @endcode
/// @see GenerationalTrace
struct GenerationalEvent {
    std::string kind; ///< Owned bounded event label from the documented primitive set.
    int object{}; ///< Valid dense source/current id.
    int target{}; ///< Actual target/current id, or -1 for remembered-source notification.
    bool accepted{}; ///< Actual decision at the emitting control-flow point.
    std::size_t stack_size{}; ///< Actual young/full stack size at capture, or zero outside traversal.
};

/// @brief Owned input graph/ages, ordinary run and chronological actual collector events.
/// The online quota counts complete event records, not numeric entries or bytes.
/// Owned input/name vectors, root/work/final output, working tables/stacks and
/// allocator storage are excluded; it is not a peak RAM limit. Invalid inputs
/// are rejected before copies/capture. No partial/truncated trace is returned.
/// @par Example
/// @code
/// const auto t=dm::algorithms::minor_gc_trace({1,{{}},{0},{}},{0});
/// std::cout << t.run.result.minor_scanned << ' ' << t.graph.num_objects; // 1 1
/// @endcode
/// @see minor_gc_trace()
struct GenerationalTrace {
    ObjectGraph graph; ///< Independent original graph, including ignored size metadata.
    std::vector<int> generations; ///< Independent original binary ages.
    GenerationalRun run; ///< Ordinary final result/work from the same execution.
    std::vector<GenerationalEvent> events; ///< Actual primitive events, no replay or per-event graph copy.
};

/// @brief Sparse young-only outcome from a validated maintained heap model.
/// survivors/freed partition its fixed young ids in ascending order. These are
/// candidates only: collect() does not promote, deallocate or change the graph.
/// No full traversal, dense old-object output or global age scan is performed.
/// Work/storage per call is O(Y+R+W+E_y), for Y young objects, R root slots,
/// W remembered old-to-young slots and E_y fields of retained young objects.
/// @par Example
/// @code
/// dm::algorithms::GenerationalHeap h({2,{{1},{}},{0},{}},{1,0});
/// const auto r=h.collect();
/// std::cout << r.survivors[0] << ' ' << r.work.old_fields; // 1 0
/// @endcode
/// @see GenerationalHeap
struct YoungCollection {
    std::vector<int> survivors; ///< Ascending retained young ids, possibly conservative floating garbage.
    std::vector<int> freed; ///< Ascending unretained young ids, not actual deallocation.
    GenerationalWork work; ///< Actual per-call work; setup/old discovery/full-comparison counts are zero.
};

/// @brief Owned graph and binary ages for a generated generational example.
/// Sizes are absent and references use dense slots; no real memory is allocated
/// for modeled objects. Vectors are independent of later calls.
/// @par Example
/// @code
/// const auto x=dm::algorithms::generational_chain_example(4);
/// std::cout << x.graph.num_objects << ' ' << x.generations[4]; // 7 0
/// @endcode
/// @see generational_chain_example()
struct GenerationalInput {
    ObjectGraph graph; ///< Owned old chain, two young survivors and optional young garbage.
    std::vector<int> generations; ///< One binary generation per object, old chain first.
};

/// @brief Analyze the young generation and compute a separate full-root comparison.
/// Validates complete graph shape/ids then exact binary ages before indexing.
/// Rebuilds boundary roots from ALL old fields, including unreachable old sources;
/// this conservatively retains every full-root-reachable young object, possibly
/// more. Young roots precede old-source/field-order targets; all seeds are marked
/// before ordered LIFO traversal. Duplicate fields/roots/cycles are allowed.
/// Sizes are entirely ignored. No graph, generation or real object is changed.
/// Discovery, dense outputs and full comparison make total work O(N+E+R), not
/// merely proportional to young visits. O(N+E+R) worst-case auxiliary storage.
/// @param g Valid dense ObjectGraph, including valid unreachable fields.
/// @param gen Exactly num_objects entries, each 0 young or 1 old; empty matches empty graph.
/// @return Owned legacy flags/lists/counts; fields keep their existing order/types.
/// @throws std::invalid_argument Invalid graph or generation length/value.
/// @throws std::length_error Actual validation/work/container capacities overflow.
/// @par Example
/// @code
/// const auto r=dm::algorithms::minor_gc({2,{{1},{}},{0},{}},{1,0});
/// std::cout << r.remembered_set[0] << ' ' << int(r.promoted[1]); // 0 1
/// @endcode
/// @par References
/// - D. Ungar (1984), Generation Scavenging: A Non-disruptive High Performance
///   Storage Reclamation Algorithm, https://doi.org/10.1145/800020.808261 , §8.1.
///   This graph analysis rebuilds boundary sources and proposes promotion; it does
///   not copy objects or implement Ungar's three-space/tenuring policy.
/// @see minor_gc_run(), GenerationalHeap
GenerationalResult minor_gc(const ObjectGraph& g, const std::vector<int>& gen);

/// @brief Run the same checked collector with actual work and full-root marks.
/// Same semantics, ordering, ownership and complexity as minor_gc; no replay.
/// Every stored old field is visited for discovery and a separate full DFS runs.
/// @param g Valid dense graph; sizes ignored completely.
/// @param gen Exact matching binary ages, each 0 young or 1 old.
/// @return Owned legacy result, actual full marks and counted selected work.
/// @throws std::invalid_argument Invalid graph or ages, before indexing.
/// @throws std::length_error Actual validation/work/container capacities overflow.
/// @par Example
/// @code
/// const auto r=dm::algorithms::minor_gc_run({2,{{1},{}},{0},{}},{1,0});
/// std::cout << r.work.validation.references << ' ' << r.work.full_fields; // 1 1
/// @endcode
/// @see minor_gc(), GenerationalRun
GenerationalRun minor_gc_run(const ObjectGraph& g, const std::vector<int>& gen);

/// @brief Capture actual discovery, young traversal, classification and full-comparison events.
/// Same execution as minor_gc_run. Complete validation precedes input copies and
/// capture. Each next actual event is admitted online before append; zero quota
/// disables only that optional cap, not capacity/counter checks. No replay,
/// truncation, partial return or caller mutation. O(events) extra record storage
/// plus owned input bytes and ordinary working/output storage.
/// @param g Valid dense graph; sizes copied but ignored for collection.
/// @param gen Exact binary ages.
/// @param max_events Positive actual event-record cap, default 1000000; zero disables quota.
/// @return Owned input snapshots, actual chronological events and ordinary run.
/// @throws std::invalid_argument Invalid graph/ages, checked before quota.
/// @throws std::length_error Next actual record exceeds quota/capacity or work counters overflow.
/// @par Example
/// @code
/// const auto t=dm::algorithms::minor_gc_trace({1,{{}},{0},{}},{0},0);
/// std::cout << t.run.work.young_objects << ' ' << t.run.work.full_objects; // 1 1
/// @endcode
/// @see GenerationalTrace
GenerationalTrace minor_gc_trace(const ObjectGraph& g, const std::vector<int>& gen,
                                std::size_t max_events=1000000);

/// @brief Build an old chain with a direct young root, an old-to-young boundary and garbage.
/// Old ids 0..old_count-1 form a rooted chain; its last source points to young
/// old_count+1. The second root is young old_count. Additional young ids are
/// isolated/unrooted. old_count 4/unreachable_young 1 reproduces the original report.
/// Complete object count is checked before allocation; O(N) construction/storage.
/// @param old_count Positive old-chain length.
/// @param unreachable_young Number of isolated young garbage objects, default 1.
/// @param max_objects Positive whole-object cap, default 1000000; zero disables optional quota.
/// @return Independent valid graph and fixed binary generations.
/// @throws std::invalid_argument old_count is zero.
/// @throws std::length_error Total exceeds int ids, vector capacity or positive quota.
/// @par Example
/// @code
/// const auto x=dm::algorithms::generational_chain_example(4,1,7);
/// const auto r=dm::algorithms::minor_gc(x.graph,x.generations);
/// std::cout << r.minor_scanned << ' ' << r.full_scanned; // 2 6
/// @endcode
/// @see GenerationalInput
GenerationalInput generational_chain_example(std::size_t old_count,
    std::size_t unreachable_young=1, std::size_t max_objects=1000000);

/// @brief Owned fixed-age graph with an exact maintained set of old-to-young reference slots.
/// Construction validates once and indexes young ids and every boundary slot.
/// Only checked existing-field replacement is permitted; ages, roots, object ids
/// and field counts stay fixed. set_reference maintains the slot invariant as a
/// modeled write barrier. It is not a concurrent/runtime collector: collect only
/// proposes retention/reclamation, with no age changes, copying or deallocation.
/// Copies own all data. Concurrent collect calls are safe absent concurrent writes.
/// @par Example
/// @code
/// dm::algorithms::GenerationalHeap h({3,{{0},{},{}},{0},{}},{1,0,0});
/// h.set_reference(0,0,2);
/// std::cout << h.collect().survivors[0]; // 2
/// @endcode
/// @par References
/// - Ungar (1984), https://doi.org/10.1145/800020.808261 , §8.1 store-side remembered set;
///   this fixed-age graph model indexes exact slots and performs no copying/tenuring.
/// @see minor_gc_run(), YoungCollection
class GenerationalHeap {
public:
    /// @brief Validate/copy a graph and build a young index and exact boundary-slot set.
    /// O(N+E+R) setup with sorted hinted slot insertion; retained graph/index storage
    /// is O(N+E+R). Sizes are ignored. No caller mutation; failed construction returns no object.
    /// @param graph Valid dense graph with fixed roots and field counts.
    /// @param generations Exact binary ages, fixed for the model's lifetime.
    /// @throws std::invalid_argument Invalid graph or ages before copying/indexing.
    /// @throws std::length_error Actual validation/work/container capacities overflow.
    /// @par Example
    /// @code
    /// dm::algorithms::GenerationalHeap h({2,{{1},{}},{0},{}},{1,0});
    /// std::cout << h.preparation_work().old_fields; // 1
    /// @endcode
    /// @see collect()
    GenerationalHeap(const ObjectGraph& graph, const std::vector<int>& generations);

    /// @brief Replace one existing field and maintain exact old-to-young slot membership.
    /// Checks source/target/slot before mutation. If a new set node is needed,
    /// allocate it before changing the graph; invalid input/allocation failure
    /// leaves the model unchanged. Expected pointer-write model costs O(log(W+1))
    /// ordered-set operations, W boundary slots; not a runtime instruction benchmark.
    /// @param source Existing source object id.
    /// @param field Zero-based existing reference field index.
    /// @param target Existing target object id.
    /// @return Whether this field is now an old-to-young boundary slot.
    /// @throws std::out_of_range Invalid source, target or field index, before mutation.
    /// @par Example
    /// @code
    /// dm::algorithms::GenerationalHeap h({2,{{0},{}},{0},{}},{1,0});
    /// std::cout << h.set_reference(0,0,1) << ' ' << h.remembered_sources()[0]; // 1 0
    /// @endcode
    /// @see snapshot(), collect()
    bool set_reference(int source, std::size_t field, int target);

    /// @brief Collect a sparse young-only analysis from validated maintained boundary slots.
    /// No old field scan, whole-graph revalidation, full comparison, dense N-vector,
    /// age change or mutation. Young roots precede ordered remembered-slot targets;
    /// the same young traversal kernel is used as minor_gc. O(Y+R+W+E_y) work and
    /// temporary/output storage, with E_y fields of retained young objects.
    /// @return Owned ascending survivor/freed ids and actual per-call work.
    /// @throws std::length_error Actual work/container capacities overflow.
    /// @par Example
    /// @code
    /// dm::algorithms::GenerationalHeap h({2,{{1},{}},{0},{}},{1,0});
    /// const auto r=h.collect();
    /// std::cout << r.work.remembered_fields << ' ' << r.work.full_objects; // 1 0
    /// @endcode
    /// @see YoungCollection
    YoungCollection collect() const;

    /// @brief Return ascending distinct old sources currently containing young-target fields.
    /// O(W) visits to the maintained slots; no old reference discovery or graph mutation.
    /// @return Independent owned source ids, excluding duplicate fields from the same source.
    /// @par Example
    /// @code
    /// dm::algorithms::GenerationalHeap h({2,{{1,1},{}},{0},{}},{1,0});
    /// std::cout << h.remembered_sources().size(); // 1
    /// @endcode
    /// @see set_reference()
    std::vector<int> remembered_sources() const;

    /// @brief Copy the current modeled graph after any checked field replacements.
    /// O(N+E+R) owned copying; this explicit snapshot cost is not part of collect().
    /// @return Independent graph with original roots/sizes and current references.
    /// @par Example
    /// @code
    /// dm::algorithms::GenerationalHeap h({2,{{0},{}},{0},{}},{1,0});
    /// h.set_reference(0,0,1);
    /// std::cout << h.snapshot().refs[0][0]; // 1
    /// @endcode
    /// @see set_reference()
    ObjectGraph snapshot() const;

    /// @brief Copy the constructor's actual validation/index/discovery counters.
    /// Constant-sized output; later writes/collect calls do not change this record.
    /// @return Owned setup-only work, separate from per-call collect work.
    /// @par Example
    /// @code
    /// dm::algorithms::GenerationalHeap h({2,{{1},{}},{0},{}},{1,0});
    /// std::cout << h.preparation_work().ages_checked; // 2
    /// @endcode
    /// @see GenerationalWork
    GenerationalWork preparation_work() const;
private:
    ObjectGraph graph_; ///< Owned validated graph; only existing field targets can change.
    std::vector<int> generations_; ///< Owned binary ages, fixed after construction.
    std::vector<int> young_ids_; ///< Ascending young ids, used for sparse output/classification.
    std::vector<int> young_index_; ///< Dense id-to-young-index map, -1 for old objects.
    std::set<std::pair<int,std::size_t>> slots_; ///< Exact sorted (source,field) keys whose source is old and target young.
    GenerationalWork preparation_; ///< Constructor-only actual validation and discovery work.
};
} // namespace datamunge::algorithms
