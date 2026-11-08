#pragma once
/// @file fp_growth.hpp
/// @brief Checked iterative FP-growth, owned actual trees and conditional pattern bases.
#include <datamunge/algorithms/association_rules.hpp>
#include <limits>
namespace datamunge::algorithms {
/// @brief One actual FP node in creation order, with a checked full-width parent ID.
/// Node0 is the dummy root with item SIZE_MAX, count0 and parent SIZE_MAX. Other nodes may legitimately carry item SIZE_MAX; root identity comes from index0, never the item value. Counts are weighted prefix occurrence totals, not edge probabilities.
/// @par Example
/// @code
/// dm::algorithms::FPGrowthNode x{};
/// std::cout << x.count; // 0
/// @endcode
/// @see fp_growth_trace()
struct FPGrowthNode {
    std::size_t item{}; ///< Opaque item ID; dummy root also uses SIZE_MAX.
    std::size_t count{}; ///< Actual weighted prefix count; root stays zero.
    std::size_t parent{}; ///< Parent node index, SIZE_MAX only for root.
};
/// @brief Owned node-link indices for one observed item in a captured tree.
/// Headers are returned by ascending item ID; each node-index list preserves actual creation order. It replaces a linked-chain view with independent owned indices. Sum the referenced node counts for item support in this conditional tree.
/// @par Example
/// @code
/// dm::algorithms::FPGrowthHeader x{};
/// std::cout << x.nodes.size(); // 0
/// @endcode
/// @see fp_growth_trace()
struct FPGrowthHeader {
    std::size_t item{}; ///< Observed item ID, including SIZE_MAX.
    std::vector<std::size_t> nodes{}; ///< Owned indices into the containing tree nodes.
};
/// @brief Owned actual fully built tree, including conditional trees pruned to a root.
/// Trees occur in actual build order. Prefix stores the current DFS suffix sequence, not sorted canonical output. Order is frequency descending with ID-ascending ties; header rows are ID ascending. Nodes/headers own storage. Only the frequent-item projection is compressed; raw rows, infrequent items and empty projections are not recoverable from this tree alone.
/// @par Example
/// @code
/// dm::algorithms::FPGrowthTree x{};
/// std::cout << x.nodes.size(); // 0
/// @endcode
/// @see fp_growth_trace()
struct FPGrowthTree {
    std::vector<std::size_t> prefix{}; ///< Actual current unsorted DFS extension sequence; empty for initial tree.
    std::vector<FPGrowthNode> nodes{}; ///< Owned creation-order nodes, including dummy root at0.
    std::vector<std::size_t> order{}; ///< Retained insertion order: descending support, then ascending item ID.
    std::vector<FPGrowthHeader> header{}; ///< Owned item-ID-ascending header rows, preserving node creation order.
};
/// @brief One actual nonempty conditional prefix path with its target-node count.
/// Items run from the nearest ancestor toward the root, excluding the target and root. The child builder sorts them again using its own conditional frequency order. Weight is the target node count, not an ancestor count or number of path records.
/// @par Example
/// @code
/// dm::algorithms::FPGrowthWeightedPath x{};
/// std::cout << x.weight; // 0
/// @endcode
/// @see fp_growth_trace()
struct FPGrowthWeightedPath {
    std::vector<std::size_t> items{}; ///< Owned nearest-ancestor-first path, with no repeated item.
    std::size_t weight{}; ///< Positive multiplicity of that whole conditional path.
};
/// @brief One actual DFS emission followed by its conditional pattern base.
/// Step order is the original depth-first mining schedule. Pattern is the emitted unsorted suffix sequence. Empty ancestor paths are omitted from base. A nonempty base always builds a child tree, even if filtering leaves only its root. SIZE_MAX child means no nonempty base and no child build; no replay is used.
/// @par Example
/// @code
/// dm::algorithms::FPGrowthStep x{};
/// std::cout << x.support; // 0
/// @endcode
/// @see fp_growth_trace()
struct FPGrowthStep {
    std::size_t tree{}; ///< Source captured-tree index.
    std::size_t item{}; ///< Item appended in this emission.
    std::size_t support{}; ///< Sum of source header-node counts.
    std::size_t child{std::numeric_limits<std::size_t>::max()}; ///< Built child-tree index; SIZE_MAX when base is empty.
    std::vector<std::size_t> pattern{}; ///< Actual unsorted emitted item sequence.
    std::vector<FPGrowthWeightedPath> base{}; ///< Owned actual weighted conditional paths in header-node order.
};
/// @brief Actual build, insertion, header, parent-walk and DFS counts.
/// Counts exclude validation, allocation, sort comparisons, hashing internals and capture copies. Weighted counts contribute one visit per stored item, not weight visits. Root creation is counted through trees_built; nodes_created excludes roots. Peak node records include only trees resident in active DFS frames, excluding temporary child/conditional buffers and retained history.
/// @par Example
/// @code
/// dm::algorithms::FPGrowthWork x{};
/// std::cout << x.nodes_created; // 0
/// @endcode
/// @see fp_growth_trace()
struct FPGrowthWork {
    std::size_t transactions{}; ///< Raw basket rows, including empty/repeated rows.
    std::size_t raw_items{}; ///< Raw item occurrences, including within-row duplicates.
    std::size_t incidences{}; ///< Normalized unique row-item incidences.
    std::size_t trees_built{}; ///< All actual builds, including initial empty/pruned children.
    std::size_t nonempty_trees{}; ///< Built trees with at least one nonroot node.
    std::size_t nodes_created{}; ///< Actual nonroot node creations across all builds.
    std::size_t frequency_item_visits{}; ///< Actual weighted frequency accumulation visits.
    std::size_t filter_item_visits{}; ///< Actual item visits deciding rank retention.
    std::size_t insert_visits{}; ///< Actual retained-item child lookups/count updates.
    std::size_t reused_nodes{}; ///< Insert visits following an existing child.
    std::size_t header_count_visits{}; ///< Actual node visits summing header supports for mining.
    std::size_t header_extract_visits{}; ///< Actual header-node visits extracting conditional bases.
    std::size_t parent_hops{}; ///< Actual nonroot ancestor appends to conditional paths.
    std::size_t conditional_paths{}; ///< Actual nonempty weighted conditional path records.
    std::size_t emitted_itemsets{}; ///< Actual itemset emissions.
    std::size_t emitted_items{}; ///< Sum of emitted pattern lengths before canonical sorting.
    std::size_t peak_frames{}; ///< Maximum explicit DFS frames, including initial empty frame.
    std::size_t peak_live_nodes{}; ///< Maximum frame-resident tree nodes, including their roots.
};
/// @brief Owned canonical frequent sets and actual work without history.
/// Uses the same executing kernel as capture. The ordinary output is never counted as retained trace-history quota. No input mutation or shared state.
/// @par Example
/// @code
/// dm::algorithms::FPGrowthResult x{};
/// std::cout << x.itemsets.size(); // 0
/// @endcode
/// @see fp_growth_trace()
struct FPGrowthResult {
    std::vector<FrequentItemset> itemsets{}; ///< Owned canonical sorted itemsets/supports.
    FPGrowthWork work{}; ///< Actual once-only work.
};
/// @brief Owned raw baskets, actual trees, emissions and weighted conditional bases.
/// Retained history is admitted online before each associated copy/move. Base entries are N+raw occurrences. Each tree costs1+prefix+nodes+order+sum(1+header node IDs); each step costs1+pattern+sum(1+path items). Node/step/tree/header aggregates count one record each; arrays add their item/index entries. Root nodes and pruned child trees count. This quota excludes ordinary canonical output, work, normalized input, active trees, conditional buffers and allocator bytes; it is not bytes or peak RAM. Working records may be formed before their size is known. No partial result escapes failure.
/// @par Example
/// @code
/// dm::algorithms::FPGrowthTrace x{};
/// std::cout << x.captured_entries; // 0
/// @endcode
/// @see fp_growth_trace()
struct FPGrowthTrace {
    std::vector<std::vector<std::size_t>> transactions{}; ///< Owned raw rows, preserving order and duplicates.
    std::size_t min_support_count{}; ///< Caller threshold, including the historical zero case.
    std::vector<FPGrowthTree> trees{}; ///< Owned actual fully built initial/conditional trees.
    std::vector<FPGrowthStep> steps{}; ///< Owned actual DFS emissions and bases.
    std::vector<FrequentItemset> itemsets{}; ///< Ordinary owned canonical output, excluded from history quota.
    FPGrowthWork work{}; ///< Actual work from this same execution.
    std::size_t captured_entries{}; ///< Exact admitted retained-history entry count.
};
/// @brief Measure the executing FP miner without retaining history.
/// Baskets are sets: sort and deduplicate each without changing the caller.
/// Empty/repeated rows retain their identities; IDs are opaque size_t including
/// SIZE_MAX. No empty itemset is returned. Positive thresholds return all sets
/// whose row count meets the threshold; above N returns none. Threshold0 retains
/// the historical positive-support-only FP behavior, unlike Apriori/Eclat0.
/// Build order is support descending then ID ascending; mining order is support
/// ascending then ID ascending. Explicit DFS preserves the original emit/base/
/// child-build/descent schedule without recursive call-stack depth. No single-path
/// shortcut, Apriori candidate join or extra database scan per candidate is used.
/// Output can have exponentially many sets and item IDs. Sorting, hash operations,
/// conditional rebuilding and output copying remain real work; no universal
/// compression/runtime guarantee. Independent calls share no mutable state.
/// Counts/capacities are checked; errors return no partial result.
///
/// No optional output cap; working/output storage may be exponential in observed item count.
/// @param transactions Raw unchanged baskets.
/// @param min_support_count Absolute threshold;0 retains positive-support cooccurrences.
/// @return Owned canonical itemsets and actual work.
/// @throws std::length_error Count/container overflow or exceeded optional history cap.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::fp_growth_frequent_itemsets_work({{0,1},{0},{1}},2);
/// std::cout << r.work.emitted_itemsets; // 2
/// @endcode
/// @par References
/// - J. Han, J. Pei and Y. Yin (2000), Mining Frequent Patterns without Candidate Generation, SIGMOD2000,1–12, doi:10.1145/342009.335372; https://sigmodrecord.org/?download_id=6623&smd_process_download=1; explicit DFS, vector headers, no single-path shortcut.
/// @see fp_growth_frequent_itemsets(), fp_growth_trace_base_entries()
FPGrowthResult fp_growth_frequent_itemsets_work(const std::vector<std::vector<std::size_t>>& transactions, std::size_t min_support_count);
/// @brief Capture actual FP-tree builds and weighted conditional bases without replay.
/// Baskets are sets: sort and deduplicate each without changing the caller.
/// Empty/repeated rows retain their identities; IDs are opaque size_t including
/// SIZE_MAX. No empty itemset is returned. Positive thresholds return all sets
/// whose row count meets the threshold; above N returns none. Threshold0 retains
/// the historical positive-support-only FP behavior, unlike Apriori/Eclat0.
/// Build order is support descending then ID ascending; mining order is support
/// ascending then ID ascending. Explicit DFS preserves the original emit/base/
/// child-build/descent schedule without recursive call-stack depth. No single-path
/// shortcut, Apriori candidate join or extra database scan per candidate is used.
/// Output can have exponentially many sets and item IDs. Sorting, hash operations,
/// conditional rebuilding and output copying remain real work; no universal
/// compression/runtime guarantee. Independent calls share no mutable state.
/// Counts/capacities are checked; errors return no partial result.
///
/// Check N+raw occurrences before input copying/normalization. Later tree/step records are admitted online before history copies; working records must first be formed. See FPGrowthTrace for exact units and excluded ordinary output/working memory. Zero disables only optional quota; all count/container checks remain.
/// @param transactions Raw baskets copied after base admission, before normalization.
/// @param min_support_count Absolute threshold, including historical zero semantics.
/// @param max_entries Online retained-history cap, default1million; zero disables only cap.
/// @return Fully owned actual trace and ordinary result, without truncation.
/// @throws std::length_error Count/container overflow or exceeded optional history cap.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fp_growth_trace({{0,1},{0,1},{0},{1}},2,38);
/// std::cout << t.captured_entries; // 38
/// @endcode
/// @par References
/// - J. Han, J. Pei and Y. Yin (2000), Mining Frequent Patterns without Candidate Generation, SIGMOD2000,1–12, doi:10.1145/342009.335372; https://sigmodrecord.org/?download_id=6623&smd_process_download=1; explicit DFS, vector headers, no single-path shortcut.
/// @see fp_growth_frequent_itemsets(), fp_growth_trace_base_entries()
FPGrowthTrace fp_growth_trace(const std::vector<std::vector<std::size_t>>& transactions, std::size_t min_support_count, std::size_t max_entries=1000000);
/// @brief Count raw retained-history base entries without allocation or normalization.
/// Adds one row record and every raw item occurrence, including duplicates.
/// Does not forecast the unknown future conditional trees or ordinary output.
/// @param transactions Raw unchanged baskets.
/// @return Checked N+raw occurrences.
/// @throws std::length_error Count sum overflows size_t.
/// @par Example
/// @code
/// std::cout << dm::algorithms::fp_growth_trace_base_entries({{7,7},{},{2}}); // 6
/// @endcode
/// @see fp_growth_trace()
std::size_t fp_growth_trace_base_entries(const std::vector<std::vector<std::size_t>>& transactions);
/// @brief Return the original eight grocery baskets.
/// Deterministic unchanged fixture, no shared state or random engine.
/// @return Owned rows using bread0,milk1,butter2,beer3,diapers4.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// std::cout << dm::algorithms::fp_growth_example_transactions().size(); // 8
/// @endcode
/// @see fp_growth_frequent_itemsets(), itemset_label()
std::vector<std::vector<std::size_t>> fp_growth_example_transactions();
/// @brief Return the original five grocery item names.
/// Deterministic unchanged fixture, no shared state or random engine.
/// @return Owned bread,milk,butter,beer,diapers labels indexed0..4.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// std::cout << dm::algorithms::fp_growth_example_labels()[3]; // beer
/// @endcode
/// @see fp_growth_frequent_itemsets(), itemset_label()
std::vector<std::string> fp_growth_example_labels();
} // namespace datamunge::algorithms
