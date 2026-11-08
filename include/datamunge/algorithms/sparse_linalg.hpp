#pragma once

#include <vector>
#include <datamunge/algorithms/cuthill_mckee.hpp>
#include <datamunge/algorithms/cannon.hpp>

namespace datamunge::algorithms {

/// @brief The *minimum degree algorithm*: chooses an *elimination ordering* of a symmetric sparse
///        matrix that heuristically *minimizes fill-in* (new nonzeros created) during Cholesky
///        factorization. It greedily eliminates the vertex of *smallest current degree*, adding the
///        "fill" edges that make its neighbours a clique, and repeats on the shrinking graph. Ordering
///        by minimum degree tends to defer the vertices that would create the most fill, so the factor
///        stays sparse -- it is the classic ordering behind sparse direct solvers (and the ancestor of
///        AMD/METIS).
///
/// @param adjacency the symmetric matrix pattern as an adjacency list.
/// @return a permutation giving the elimination order (@c perm[k] eliminated @c k-th).
std::vector<int> minimum_degree_ordering(const std::vector<std::vector<int>>& adjacency);

/// @brief *Symbolic Cholesky decomposition*: predicts, from the *pattern alone*, exactly which zeros
///        of a symmetric positive-definite matrix will become nonzero (*fill-in*) in its Cholesky
///        factor @c L -- before any numbers are touched. Eliminating column @c k makes all pairs of
///        its below-diagonal nonzeros mutually coupled (a clique), and those couplings propagate. The
///        resulting pattern lets a sparse solver *allocate @c L exactly once*, with no reallocation
///        during the numeric factorization -- the reason symbolic analysis precedes the numeric phase
///        in every sparse Cholesky code.
///
/// @param adjacency the symmetric matrix pattern (off-diagonal nonzeros) as an adjacency list.
/// @return the fill pattern of @c L: @c result[i] lists the columns @c j<i where @c L(i,j) is nonzero.
std::vector<std::vector<int>> symbolic_cholesky(const std::vector<std::vector<int>>& adjacency);

} // namespace datamunge::algorithms
