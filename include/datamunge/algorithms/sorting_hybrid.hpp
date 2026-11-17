#pragma once
#include <datamunge/algorithms/timsort.hpp>
#include <datamunge/algorithms/stooge_sort.hpp>

/// @file sorting_hybrid.hpp
/// @brief Timsort, bead (gravity) sort with its partition view and gravity simulation, and stooge sort.

// Three more sorts:
//   - timsort   : the real-world adaptive merge/insertion hybrid (runs, minrun,
//                 binary insertion, and the merge-collapse stack invariants)
//   - bead_sort : "gravity" sort for non-negative integers
//   - stooge_sort: the recursive (and gloriously inefficient) sort

#include <algorithm>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// @brief Bead ("gravity") sort of non-negative integers, in place, ascending.
///
/// Number \f$a_i\f$ is a row of \f$a_i\f$ beads on rods \f$0,\dots,a_i-1\f$. When the beads fall, rod
/// \f$j\f$ holds \f$\lambda'_j=\#\{i:a_i>j\}\f$ beads (bead_columns()), and the row at height \f$r\f$
/// from the bottom then has \f$\#\{j:\lambda'_j>r\}\f$ beads, the \f$(r+1)\f$-th largest input.
/// Reading the column heights through a histogram of the values makes this \f$O(n+\max a)\f$ time
/// and \f$O(\max a)\f$ memory, the same as counting sort.
/// @param a Values to sort.
void bead_sort(std::vector<unsigned>& a);

/// @brief Heights of the bead columns after the fall: \f$\lambda'_j=\#\{i:a_i>j\}\f$ for
///        \f$j=0,\dots,\max a-1\f$, the conjugate of the partition formed by the values.
/// @param a Non-negative values (any order).
/// @return Non-increasing column heights; empty if every value is 0.
std::vector<unsigned> bead_columns(const std::vector<unsigned>& a);

/// @brief Conjugate (transpose) of a partition: part \f$j\f$ of the result counts the parts of
///        @p parts that exceed \f$j\f$. Applying it twice sorts @p parts into non-increasing order
///        and drops zeros.
/// @param parts Non-negative parts (any order).
/// @return The conjugate partition, non-increasing.
std::vector<unsigned> conjugate_partition(const std::vector<unsigned>& parts);

/// Tick-by-tick simulation of the beads falling.
struct BeadFall {
    /// @brief Bead grid before the fall: row r (top first) has a bead in column j iff j < a[r].
    std::vector<std::vector<unsigned char>> before;
    /// @brief Bead grid after the fall, same orientation.
    std::vector<std::vector<unsigned char>> after;
    /// @brief Ticks until no bead can move. In each tick every bead with free space below it drops
    ///        one cell, and beads resting on a falling bead fall with it, so this is the longest
    ///        single fall.
    int ticks{0};
    /// @brief Total cells fallen by all beads.
    long long bead_moves{0};
    /// @brief Number of beads moved at each tick.
    std::vector<long long> moves_per_tick;
};

/// @brief Simulates gravity on the bead grid of @p a, one cell per tick, as an abacus would.
///
/// Beads keep their order within a column, so the \f$k\f$-th bead from the bottom of column
/// \f$j\f$ ends in row \f$n-k\f$. The number of ticks is the longest such fall, at most \f$n-1\f$.
/// @param a Values (rows, top row first).
/// @return Grids before and after, ticks and move counts.
/// @throws std::invalid_argument if the grid would exceed 10^8 cells.
BeadFall bead_sort_gravity(const std::vector<unsigned>& a);

} // namespace datamunge::algorithms
