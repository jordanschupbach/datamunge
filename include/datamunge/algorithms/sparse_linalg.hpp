#pragma once

#include <vector>

namespace datamunge::algorithms {

/// @brief The *Cuthill-McKee algorithm*: reorders the rows/columns of a symmetric sparse matrix to
///        *reduce its bandwidth* -- the maximum distance of any nonzero from the diagonal. Treating
///        the matrix pattern as a graph, it runs a breadth-first search from a low-degree start
///        vertex, visiting each level's vertices in increasing-degree order; that BFS order keeps
///        connected (nonzero-coupled) rows close together, clustering the nonzeros near the diagonal.
///        The *reverse* Cuthill-McKee (RCM) ordering -- simply reversing the result -- usually reduces
///        *fill-in* under Cholesky further, and is the more common choice. A tight band means cheaper
///        banded solves and factorizations.
///
/// @param adjacency @c adjacency[i] lists the neighbours of node @c i (the off-diagonal nonzeros of
///        row @c i); must be symmetric.
/// @param reverse if true, return the reverse Cuthill-McKee ordering.
/// @return a permutation: @c perm[k] is the original index placed at new position @c k.
std::vector<int> cuthill_mckee(const std::vector<std::vector<int>>& adjacency, bool reverse = false);

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

/// @brief *Cannon's algorithm* for dense matrix multiplication @c C = A·B of two @c n×n matrices.
///        Designed for a processor mesh, it *skews* the two matrices with cyclic row/column shifts so
///        that at each of @c n steps every processor holds a matching @c A- and @c B-block to multiply
///        and accumulate, then shifts again -- computing the full product with only *nearest-neighbour*
///        communication and @c O(n) memory per node. This sequential implementation follows the same
///        shift-multiply-accumulate schedule and returns the exact product.
///
/// @param A,B the input matrices in row-major order (each @c n·n entries).
/// @param n the matrix dimension.
/// @return the product @c C = A·B in row-major order.
std::vector<double> cannon_matmul(const std::vector<double>& A, const std::vector<double>& B, int n);

} // namespace datamunge::algorithms
