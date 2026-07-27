#pragma once

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

struct AssignmentResult {
    /// @brief assignment[i] is the column (task) assigned to row (worker) i -- a permutation of 0..n-1.
    std::vector<std::size_t> assignment;
    /// @brief Total cost of the assignment: sum_i cost[i][assignment[i]].
    double cost{0.0};
};

/// @brief The Hungarian algorithm (Kuhn 1955; Munkres 1957), a.k.a. Kuhn-Munkres, for the
///        *assignment problem*: given an n x n matrix of real costs @p cost, where cost[i][j]
///        is the cost of assigning row (worker) i to column (task) j, find a one-to-one
///        assignment of rows to columns -- a permutation \(\pi\) -- of minimum total cost
///        \(\sum_i cost[i][\pi(i)]\). Equivalently, a minimum-cost perfect matching in the
///        complete weighted bipartite graph. This is the primal-dual O(n^3) formulation that
///        maintains row potentials \(u_i\) and column potentials \(v_j\) obeying
///        \(u_i + v_j \le cost[i][j]\) (LP dual feasibility), grows a matching over the *tight*
///        edges where equality holds, and -- when no augmenting path of tight edges exists --
///        raises the potentials by the minimum slack across the current alternating cut,
///        exposing a new tight edge without ever violating feasibility. It terminates with a
///        perfect matching whose cost equals the dual value, so complementary slackness
///        certifies optimality. Robust for arbitrary real-valued (including negative) costs.
///
///        To *maximize* total value instead, negate the matrix (pass -value[i][j]); the returned
///        cost is then the negation of the maximum.
///
/// @param cost a square, non-empty n x n cost matrix; cost[i][j] is the cost of row i -> column j.
/// @return the minimum-cost assignment and its total cost.
/// @throws std::invalid_argument if @p cost is empty or not square.
AssignmentResult hungarian(const std::vector<std::vector<double>>& cost);

} // namespace datamunge::algorithms
