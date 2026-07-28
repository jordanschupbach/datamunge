#pragma once

#include <vector>

namespace datamunge::algorithms {

/// @brief The result of an exact-cover search: whether a cover was @ref solved and, if so, the list
///        of chosen @ref rows (indices into the input) that cover every column exactly once.
struct ExactCoverResult {
    bool             solved{false};
    std::vector<int> rows; ///< indices of the selected rows forming an exact cover (if solved).
};

/// @brief *Exact cover* via *Algorithm X* implemented with *Dancing Links* (Knuth's DLX). Given a
///        0/1 matrix (each row a subset of columns), find a set of rows that covers *every column
///        exactly once*. Algorithm X is the natural nondeterministic recursion -- pick an uncovered
///        column, try each row covering it, recurse -- and *Dancing Links* is the trick that makes it
///        fast: the matrix is a toroidal mesh of doubly-linked nodes, and covering/uncovering a
///        column is done by the reversible pointer surgery @c L[R[x]]=L[x] (and its exact inverse on
///        backtrack), so no data is copied as the search explores and unexplores. Choosing the column
///        with the fewest rows (the @c S-heuristic) keeps the branching small. This is the standard
///        exact solver for tiling, N-queens, and Sudoku (each encoded as an exact-cover instance).
///
/// @param num_columns the number of columns (elements that must each be covered exactly once).
/// @param rows each row as the list of column indices (@c 0..num_columns-1) it covers.
/// @return the first @ref ExactCoverResult found; @c solved is false if no exact cover exists.
ExactCoverResult exact_cover(int num_columns, const std::vector<std::vector<int>>& rows);

} // namespace datamunge::algorithms
