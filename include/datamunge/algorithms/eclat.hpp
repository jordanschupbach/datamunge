#pragma once
/// @file eclat.hpp
/// @brief Sorted vertical tidsets, iterative Eclat work and actual intersection evidence.
#include <datamunge/algorithms/association_rules.hpp>
#include <limits>

namespace datamunge::algorithms {

/// @brief An observed item and its ascending, unique transaction indices.
///
/// Returned records own their buffers; item identifiers are opaque size_t values,
/// including SIZE_MAX. Transaction indices refer to original row order.
/// @par Example
/// @code
/// const auto v = dm::algorithms::vertical_tidsets({{7,7},{},{7,2}});
/// std::print("{} {}\n", v[0].item, v[1].tids.size()); // 2 2
/// @endcode
/// @see vertical_tidsets()
struct ItemTidset {
    std::size_t item = 0; ///< Observed item identifier.
    std::vector<std::size_t> tids; ///< Sorted unique row indices containing the item.
};

/// @brief Checked logical work counters for one sorted-tidset Eclat run.
///
/// Counts describe this algorithm, not elapsed time, CPU instructions or bytes.
/// An intersection step examines one pair of current heads before advancing.
/// Counters use size_t and throw length_error rather than wrap.
/// @par Example
/// @code
/// const auto r = dm::algorithms::eclat_frequent_itemsets_work({{0,1},{0}},1);
/// std::print("{} {}\n", r.work.intersections, r.work.emitted_itemsets); // 1 3
/// @endcode
/// @see eclat_frequent_itemsets_work()
struct EclatWork {
    std::size_t transactions = 0; ///< Original row count, including empty/repeated rows.
    std::size_t raw_items = 0; ///< Item occurrences before within-row deduplication.
    std::size_t incidences = 0; ///< Sum of singleton tidset sizes after deduplication.
    std::size_t frequent_singletons = 0; ///< Observed singleton supports meeting threshold.
    std::size_t intersections = 0; ///< Conditional pair intersections attempted, including failures.
    std::size_t intersection_steps = 0; ///< Total paired-head merge iterations.
    std::size_t matched_tids = 0; ///< Sum of all intersection result lengths, including failures.
    std::size_t emitted_itemsets = 0; ///< Frequent nonempty sets emitted exactly once.
    std::size_t peak_frames = 0; ///< Maximum live iterative DFS frames; zero if no output.
};

/// @brief Canonical frequent itemsets and work, without retained intersection history.
///
/// Owns its result. Calls leave inputs unchanged and share no mutable state.
/// @par Example
/// @code
/// const auto r = dm::algorithms::eclat_frequent_itemsets_work({{0,1},{0}},2);
/// std::print("{} {}\n", r.itemsets.size(), r.itemsets[0].support); // 1 2
/// @endcode
/// @see EclatWork, eclat_frequent_itemsets_work()
struct EclatResult {
    std::vector<FrequentItemset> itemsets; ///< Sorted by size, then lexicographically.
    EclatWork work; ///< Work of the same mining run.
};

/// @brief One observed singleton or actual conditional intersection in an Eclat run.
///
/// Nonroot operands are earlier state indices with a common prefix: P+a and P+b,
/// a<b. Their tidsets are intersected to obtain P+a+b. Roots have both indices
/// equal to numeric_limits<size_t>::max(). States include infrequent attempts.
/// @par Example
/// @code
/// const auto t = dm::algorithms::eclat_trace({{0,1},{0}},1);
/// std::print("{} {}\n", t.states[2].left, t.states[2].right); // 0 1
/// @endcode
/// @see EclatTrace, eclat_trace()
struct EclatState {
    FrequentItemset itemset; ///< Sorted set and support, exactly tids.size().
    std::vector<std::size_t> tids; ///< Owned ascending unique transaction indices.
    std::size_t left = std::numeric_limits<std::size_t>::max(); ///< First operand; sentinel for roots.
    std::size_t right = std::numeric_limits<std::size_t>::max(); ///< Second operand; sentinel for roots.
    std::size_t steps = 0; ///< Paired-head iterations for this attempt; zero for roots.
    bool frequent = false; ///< Whether support meets the supplied threshold.
};

/// @brief Owned evidence captured during the actual iterative Eclat traversal.
///
/// First singleton_count states are all observed roots, ordered by item id.
/// Remaining states follow actual intersection creation order. emitted records
/// depth-first visits; creation and emission orders differ. Canonical itemsets
/// are the same result as the ordinary entry point, not an independent replay.
/// Captured entries count raw rows/occurrences, state headers/item ids/tids and
/// emitted ids. Scalar fields, result copies, normalization/working frames and
/// allocator bytes are excluded; this is a retained-history quota, not peak RAM.
/// @par Example
/// @code
/// const auto t = dm::algorithms::eclat_trace({{0,1},{0,1},{0},{1}},2);
/// std::print("{} {} {}\n", t.states.size(), t.emitted.size(), t.captured_entries); // 3 3 28
/// @endcode
/// @see eclat_trace(), eclat_trace_base_entries()
struct EclatTrace {
    std::vector<std::vector<std::size_t>> transactions; ///< Exact owned raw input, including duplicates.
    std::size_t min_support_count = 0; ///< Absolute support threshold.
    std::size_t singleton_count = 0; ///< Number of observed roots before filtering.
    std::vector<EclatState> states; ///< Roots and actual attempted intersections.
    std::vector<std::size_t> emitted; ///< State indices visited/emitted in DFS order.
    std::vector<FrequentItemset> itemsets; ///< Canonical frequent result of this run.
    EclatWork work; ///< Checked work counters from the same traversal.
    std::size_t captured_entries = 0; ///< Exact retained logical entries, as documented above.
};

/// @brief Build an owned sorted vertical database, like an uncompressed item-to-row index.
///
/// Deduplicates each basket without mutating input; repeated baskets are separate
/// rows and empty rows retain their indices. Returns only observed items, ordered
/// by opaque id; no allocation indexed by the maximum item id is made.
/// @param transactions Raw baskets of item ids.
/// @return Ascending item records, each with ascending unique transaction ids.
/// @throws std::bad_alloc If allocation fails; std::length_error on container limits.
/// @par Example
/// @code
/// const auto v = dm::algorithms::vertical_tidsets({{9,9,2},{},{9}});
/// std::print("{} {} {}\n", v.size(), v[0].item, v[1].tids[1]); // 2 2 2
/// @endcode
/// @par References
/// - M. J. Zaki (2000), Scalable Algorithms for Association Mining, IEEE TKDE 12(3), 372–390. doi:10.1109/69.846291.
/// @see ItemTidset, eclat_trace()
std::vector<ItemTidset> vertical_tidsets(const std::vector<std::vector<std::size_t>>& transactions);

/// @brief Count raw-input entries required before an owned Eclat trace can start.
///
/// Pure checked shape pass; allocates nothing and does not normalize baskets.
/// The base is N + raw item occurrences, counting empty rows and duplicates.
/// @param transactions Raw baskets; input is unchanged.
/// @return Checked base entry count, zero for an empty database.
/// @throws std::length_error If the sum cannot be represented by size_t.
/// @par Example
/// @code
/// std::print("{}\n", dm::algorithms::eclat_trace_base_entries({{7,7},{}})); // 4
/// @endcode
/// @see eclat_trace()
std::size_t eclat_trace_base_entries(const std::vector<std::vector<std::size_t>>& transactions);

/// @brief Mine with the ordinary Eclat semantics and checked work counters.
///
/// Explicit DFS frames replace recursion; conditional sorted tidsets are merged
/// without rescanning transactions. No optional capture quota or history is kept.
/// Zero threshold emits every nonempty subset of the observed universe, including
/// zero-support sets. Output can be exponential. Calls are independent and input
/// is unchanged. Result order is size then lexicographic ids, not DFS order.
/// @param transactions Raw baskets; duplicate items count once in each row.
/// @param min_support_count Absolute threshold; values above N yield no output.
/// @return Owned canonical itemsets and counters from that run.
/// @throws std::length_error On shape, counter or container overflow.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r = dm::algorithms::eclat_frequent_itemsets_work({{0},{1}},0);
/// std::print("{} {}\n", r.itemsets.size(), r.itemsets.back().support); // 3 0
/// @endcode
/// @par References
/// - M. J. Zaki (2000), Scalable Algorithms for Association Mining, IEEE TKDE 12(3), 372–390. doi:10.1109/69.846291. Sorted tidset intersections; no diffsets or item reordering.
/// @see eclat_frequent_itemsets(), eclat_trace()
EclatResult eclat_frequent_itemsets_work(const std::vector<std::vector<std::size_t>>& transactions,
                                        std::size_t min_support_count);

/// @brief Capture actual conditional intersections, failures and DFS emissions.
///
/// Uses the same kernel as eclat_frequent_itemsets_work(). Raw base is checked
/// before owned input copies or normalization. Each state reserves 1+items+tids
/// entries before copying into history; each emitted id reserves one. A working
/// intersection must be formed to know its size before the state quota check.
/// max_entries=0 disables only the optional quota, never overflow checks. No
/// partial trace escapes failure. See EclatTrace for excluded working/result memory.
/// @param transactions Raw baskets, copied exactly on success.
/// @param min_support_count Absolute threshold, including zero as in the ordinary API.
/// @param max_entries Retained logical-entry limit, default 1000000; zero disables it.
/// @return Owned trace with canonical output and actual creation/emission evidence.
/// @throws std::length_error On quota, arithmetic or container overflow.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto t = dm::algorithms::eclat_trace({{0,1},{0,1},{0},{1}},2,28);
/// std::print("{} {}\n", t.itemsets.size(), t.captured_entries); // 3 28
/// @endcode
/// @par References
/// - M. J. Zaki (2000), Scalable Algorithms for Association Mining. doi:10.1109/69.846291.
/// @see EclatTrace, eclat_trace_base_entries(), eclat_frequent_itemsets_work()
EclatTrace eclat_trace(const std::vector<std::vector<std::size_t>>& transactions,
                       std::size_t min_support_count, std::size_t max_entries = 1000000);
} // namespace datamunge::algorithms
