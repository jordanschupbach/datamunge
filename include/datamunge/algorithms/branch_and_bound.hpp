#pragma once

#include <vector>

namespace datamunge::algorithms {

/// @brief The result of a branch-and-bound knapsack solve: the optimal @ref value, the 0/1
///        @ref chosen vector (1 = item taken, indexed as the original input), and the number of
///        search-tree @ref nodes_explored (a measure of how much the bounding pruned).
struct KnapsackResult {
    int              value{0};          ///< maximum total value achievable within the capacity.
    std::vector<int> chosen;            ///< chosen[i] = 1 if item i is in the optimal set, else 0.
    long long        nodes_explored{0}; ///< branch-and-bound tree nodes actually visited.
};

/// @brief *Branch and bound* for the *0/1 knapsack problem*: choose a subset of items (each taken
///        whole or not at all) of maximum total value whose total weight fits the @p capacity.
///        Branch-and-bound explores the binary take/skip tree but *prunes* any node whose optimistic
///        *upper bound* -- here the fractional-knapsack (LP-relaxation) value of filling the remaining
///        capacity greedily by value density -- cannot beat the best solution found so far. That
///        bound lets it skip vast subtrees, solving instances far larger than brute force while still
///        returning the exact optimum.
///
/// @param values the item values (same length as @p weights).
/// @param weights the item weights.
/// @param capacity the knapsack weight capacity.
/// @return the optimal @ref KnapsackResult.
KnapsackResult knapsack_branch_and_bound(const std::vector<int>& values, const std::vector<int>& weights,
                                         int capacity);

} // namespace datamunge::algorithms
