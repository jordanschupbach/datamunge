#pragma once
/// @file branch_and_bound.hpp
/// @brief Exact integer knapsack branch-and-bound, real search records and seeded input fixtures.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace datamunge::algorithms {
/// @brief Optimal 0/1 knapsack value, original-index choices and actual visited-node count.
/// The historical field types/layout are retained. Ownership is independent of
/// input storage. Only strict improvements replace an incumbent solution.
/// @par Example
/// @code
/// const auto result=dm::algorithms::knapsack_branch_and_bound({10,9,6},{2,3,3},5);
/// std::print("{} {}\n",result.value,result.chosen); // 19 [1, 1, 0]
/// @endcode
/// @see knapsack_branch_and_bound(), KnapsackTrace
struct KnapsackResult {
    int value{0}; ///< Maximum feasible value; an unrepresentable optimum raises overflow_error.
    std::vector<int> chosen; ///< Original-input 0/1 selections; nonpositive-profit items remain zero.
    long long nodes_explored{0}; ///< Actual DFS entries, including leaves and bound-pruned nodes; zero for empty input.
};

/// @brief Exact fractional-relaxation components and its integer-objective upper bound.
/// LP value is whole_value+fractional_numerator/fractional_denominator. Integer
/// profits permit pruning with its exact floor integer_upper; relaxation_value
/// is a rounded double for display only and never controls pruning.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::knapsack_branch_and_bound_trace({10,8,5},{2,3,3},5);
/// const auto& bound=trace.nodes.back().bound;
/// std::print("{} {} {} {}\n",bound.whole_value,bound.fractional_numerator,
///     bound.fractional_denominator,bound.integer_upper); // 8 10 3 11
/// @endcode
/// @see KnapsackNode, knapsack_branch_and_bound_trace()
struct KnapsackBound {
    std::int64_t whole_value{0}; ///< Current feasible value plus whole remaining positive-profit items in density order.
    std::int64_t fractional_numerator{0}; ///< Positive item value times remaining capacity; zero when no fraction contributes.
    int fractional_denominator{1}; ///< Fractional item's positive weight; one when no fractional item is reached.
    std::size_t fractional_item{std::numeric_limits<std::size_t>::max()}; ///< Original item index, or size_t maximum when no fractional item is reached.
    std::int64_t integer_upper{0}; ///< Exact floor of the LP relaxation; still an upper bound on every integer completion.
    double relaxation_value{0}; ///< Rounded visualization of the exact LP components, not a pruning input.
};

/// @brief Disposition of an actual visited node after updating its feasible incumbent.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::knapsack_branch_and_bound_trace({10,8,5},{2,3,3},5);
/// std::print("{}\n",trace.nodes.back().state==dm::algorithms::KnapsackNodeState::pruned); // true
/// @endcode
/// @see KnapsackNode
enum class KnapsackNodeState {
    branch, ///< Bound exceeds the incumbent and an active item remains; attempt take, then skip.
    pruned, ///< Integer upper bound is no greater than the updated incumbent; visit no children.
    leaf ///< All positive-profit items have been decided.
};

/// @brief One DFS entry, its exact bound, incumbent transition and incoming decision.
/// Nodes are in visit order. Root parent/item_index are size_t maximum. A node's
/// level counts decided positive-profit items in KnapsackTrace::item_order.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::knapsack_branch_and_bound_trace({10,8,5},{2,3,3},5);
/// std::print("{} {}\n",trace.nodes.front().level,trace.nodes.front().bound.integer_upper); // 0 18
/// @endcode
/// @see KnapsackTrace, KnapsackNodeState
struct KnapsackNode {
    std::size_t parent{std::numeric_limits<std::size_t>::max()}; ///< Earlier node index, or size_t maximum for the root.
    std::size_t item_index{std::numeric_limits<std::size_t>::max()}; ///< Original item decided by the incoming edge; root has no incoming item.
    bool took_item{false}; ///< Incoming edge took this item; otherwise skipped it, ignored on root.
    std::size_t level{0}; ///< Number of active decisions already fixed on entry.
    int weight{0}; ///< Feasible total weight on entry, in [0,capacity].
    int value{0}; ///< Feasible total value on entry, checked against the legacy int range.
    KnapsackBound bound; ///< Exact relaxation components; at a leaf this equals the node's value.
    int incumbent_before{0}; ///< Best feasible value before considering this node's own partial selection.
    int incumbent_after{0}; ///< Best feasible value after that strict-improvement update, before descendants.
    KnapsackNodeState state{KnapsackNodeState::leaf}; ///< Whether this node branches, is pruned or is a leaf.
};

/// @brief Owned optimal result, sorted original indices and actual search/pruning records.
/// Nonpositive-profit items are excluded from the active order and remain unchosen.
/// Empty original input has no nodes; a nonempty input with no active items has one root leaf.
/// Recording adds O(N+k) storage for N visited nodes and k active items; no input is borrowed.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::knapsack_branch_and_bound_trace({10,8,5},{2,3,3},5);
/// std::print("{} {}\n",trace.result.value,trace.nodes.size()); // 18 5
/// @endcode
/// @see knapsack_branch_and_bound_trace(), KnapsackNode
struct KnapsackTrace {
    KnapsackResult result; ///< Exact same result and node count as the unrecorded solver.
    std::size_t original_size{0}; ///< Number of caller-supplied items, including ignored nonpositive profits.
    std::vector<std::size_t> item_order; ///< Positive-profit original indices in exact descending density order.
    std::vector<KnapsackNode> nodes; ///< Actual take-first DFS entries with parent links.
    long long bound_prunes{0}; ///< Visited nonleaf nodes whose integer upper bound cannot improve the incumbent.
    long long infeasible_takes{0}; ///< Attempted take branches suppressed by capacity, excluding already-pruned nodes.
    long long leaves{0}; ///< Visited nodes after all active items have been decided.
};

/// @brief Maximise 0/1 integer knapsack value with an exact fractional bound and take-first DFS.
/// Values/weights must have equal lengths; weights and capacity are nonnegative.
/// Signed profits are accepted: nonpositive items are omitted since the empty set
/// is allowed. Positive-profit free items precede weighted items. Remaining density
/// comparisons use exact widened cross-products; ties use original input index.
/// At entry, strictly improve the incumbent, then prune when floor(LP bound)<=best.
/// Both integrality and rational bound arithmetic are exact; no floating comparison
/// controls search. An explicit O(k) stack avoids recursive call-depth limits.
/// Preprocessing is O(n+k log k), with O(k*N) worst-case bound/copy work for N
/// visited nodes; N can be exponential. Auxiliary/result storage is O(n+k), with
/// no trace allocation. Inputs are unchanged and independent calls share no state.
/// Empty input returns zero value, no choices and zero visited nodes.
/// @param values Signed int profits, indexed in original order.
/// @param weights Nonnegative int weights, including zero, of the same length as values.
/// @param capacity Nonnegative int capacity.
/// @return Owned optimal value, original-index choices and actual node visits.
/// @throws std::invalid_argument If lengths differ, a weight is negative or capacity is negative.
/// @throws std::overflow_error If the optimum exceeds int, an exact bound exceeds int64, or visits exceed long long.
/// @par Example
/// @code
/// const auto result=dm::algorithms::knapsack_branch_and_bound({60,100,120,40,70},{10,20,30,15,25},50);
/// std::print("{} {}\n",result.value,result.chosen); // 220 [0, 1, 1, 0, 0]
/// @endcode
/// @par References
/// - A. H. Land and A. G. Doig (1960), An Automatic Method of Solving Discrete
///   Programming Problems, Econometrica 28(3), 497-520, doi:10.2307/1910129.
/// - P. J. Kolesar (1967), A Branch and Bound Algorithm for the Knapsack Problem,
///   Management Science 13(9), 723-735, doi:10.1287/mnsc.13.9.723.
///   This kernel uses take-first iterative DFS and the exact integer floor of the fractional relaxation.
/// @see knapsack_branch_and_bound_trace(), KnapsackResult
KnapsackResult knapsack_branch_and_bound(const std::vector<int>& values,const std::vector<int>& weights,int capacity);

/// @brief Record actual decisions, exact bounds, incumbents and pruning from the same solver kernel.
/// Parent links refer to earlier visit indices; node state is determined before
/// descendants execute. Result/tie behavior matches knapsack_branch_and_bound().
/// Recording adds O(N+k) owned storage and O(N) constant-size record work; no caller
/// data are mutated and no shared state is used. Empty input produces no nodes.
/// @param values Signed int profits; nonpositive values are left unchosen.
/// @param weights Nonnegative int weights, with exactly values.size() entries.
/// @param capacity Nonnegative int capacity, including zero.
/// @return Owned result, density order and actual search records/counters.
/// @throws std::invalid_argument If lengths, weights or capacity are invalid.
/// @throws std::overflow_error If the optimum, an exact bound or visit count exceeds its documented range.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::knapsack_branch_and_bound_trace({10,8,5},{2,3,3},5);
/// std::print("{} {}\n",trace.bound_prunes,trace.nodes.back().parent); // 3 0
/// @endcode
/// @par References
/// - P. J. Kolesar (1967), doi:10.1287/mnsc.13.9.723;
///   records this kernel's density-ordered fractional-bound variant.
/// @see knapsack_branch_and_bound(), KnapsackTrace
KnapsackTrace knapsack_branch_and_bound_trace(const std::vector<int>& values,const std::vector<int>& weights,int capacity);

/// @brief Owned integer knapsack input, suitable for deterministic method studies.
/// @par Example
/// @code
/// const auto inputs=dm::algorithms::knapsack_random_instances({4});
/// std::print("{} {}\n",inputs[0].values.size(),inputs[0].capacity); // 4 20
/// @endcode
/// @see knapsack_random_instances()
struct KnapsackInstance {
    std::vector<int> values; ///< Positive profits for generated instances, in original index order.
    std::vector<int> weights; ///< Positive weights, with the same length as values.
    int capacity{0}; ///< Nonnegative capacity for this instance.
};

/// @brief Generate independent-size knapsack fixtures from one local MT19937 stream.
/// For each requested n, draw all n values then all n weights from inclusive
/// uniform distributions; capacity is n*capacity_per_item. This preserves the
/// original report's seeded experiment order. Instances are not nested prefixes.
/// Engine output is standardised, but bounded-distribution mapping can vary between
/// standard libraries. Work/storage is O(sum sizes); no input/shared state is changed.
/// @param sizes Requested item counts in generation order, including zero.
/// @param capacity_per_item Nonnegative capacity multiplier, default 5.
/// @param seed MT19937 seed, default 777.
/// @param max_value Positive inclusive maximum profit, default 100.
/// @param max_weight Positive inclusive maximum weight, default 30.
/// @return Owned independent instances with capacities representable by int.
/// @throws std::invalid_argument If a distribution maximum is nonpositive or multiplier is negative.
/// @throws std::overflow_error If a requested capacity exceeds int.
/// @throws std::length_error If an allocation exceeds vector capacity; allocation errors propagate.
/// @par Example
/// @code
/// const auto inputs=dm::algorithms::knapsack_random_instances({4,6,8},5,777);
/// std::print("{} {}\n",inputs.size(),inputs.back().capacity); // 3 40
/// @endcode
/// @see KnapsackInstance, knapsack_branch_and_bound()
std::vector<KnapsackInstance> knapsack_random_instances(const std::vector<std::size_t>& sizes,
    int capacity_per_item=5,std::uint32_t seed=777,int max_value=100,int max_weight=30);
} // namespace datamunge::algorithms
