#pragma once

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

namespace detail {

/// @brief Binary insertion sort of a[lo, hi), given that a[lo, start) is already sorted.
/// @tparam T Element type with `operator<`.
/// @param a Values.
/// @param lo First index.
/// @param hi One past the last index.
/// @param start First index not yet sorted.
template <typename T>
void binary_insertion_sort(std::vector<T>& a, int lo, int hi, int start) {
    if (start == lo) ++start;
    for (; start < hi; ++start) {
        const T   pivot = a[start];
        int       left  = lo, right = start;
        while (left < right) {
            const int mid = left + (right - left) / 2;
            if (pivot < a[mid]) right = mid;
            else left = mid + 1;
        }
        for (int i = start; i > left; --i) a[i] = a[i - 1];
        a[left] = pivot;
    }
}

/// @brief Length of the maximal ascending (or strictly descending, then reversed) run starting at
///        a[lo] within a[lo, hi); the run is left ascending.
/// @tparam T Element type with `operator<`.
/// @param a Values.
/// @param lo First index.
/// @param hi One past the last index.
/// @return Run length, at least 1.
template <typename T>
int count_run_and_make_ascending(std::vector<T>& a, int lo, int hi) {
    int run = lo + 1;
    if (run == hi) return 1;
    if (a[run++] < a[lo]) { // strictly descending
        while (run < hi && a[run] < a[run - 1]) ++run;
        std::reverse(a.begin() + lo, a.begin() + run);
    } else { // ascending
        while (run < hi && !(a[run] < a[run - 1])) ++run;
    }
    return run - lo;
}

/// @brief Timsort's minimum run length for @p n elements: between 32 and 64, chosen so that
///        n / minrun is a power of two or just below one.
/// @param n Number of elements.
/// @return Minimum run length (n itself when n < 64).
inline int timsort_min_run(int n) {
    int r = 0;
    while (n >= 64) { r |= (n & 1); n >>= 1; }
    return n + r;
}

/// @brief Stable merge of the adjacent ascending runs a[lo, mid) and a[mid, hi).
/// @tparam T Element type with `operator<`.
/// @param a Values.
/// @param lo Start of the left run.
/// @param mid Start of the right run.
/// @param hi One past the end of the right run.
/// @param tmp Scratch buffer (resized as needed).
template <typename T>
void timsort_merge(std::vector<T>& a, int lo, int mid, int hi, std::vector<T>& tmp) {
    tmp.assign(a.begin() + lo, a.begin() + mid);
    int i = 0, j = mid, k = lo;
    const int n = mid - lo;
    while (i < n && j < hi) {
        if (!(a[j] < tmp[i])) a[k++] = tmp[i++]; // stable: prefer left on ties
        else a[k++] = a[j++];
    }
    while (i < n) a[k++] = tmp[i++];
    // remaining right-run elements are already in place
}

} // namespace detail

/// @brief Timsort, in place, ascending and stable.
///
/// Detects natural runs, extends short ones to `minrun` with binary insertion sort, and merges
/// runs while keeping the stack size invariants. Galloping mode, a constant-factor optimisation,
/// is omitted. \f$O(n\log n)\f$ worst case and \f$O(n)\f$ on already sorted input.
/// @tparam T Element type with `operator<`.
/// @param a Values to sort.
template <typename T>
void timsort(std::vector<T>& a) {
    const int n = static_cast<int>(a.size());
    if (n < 2) return;
    const int minrun = detail::timsort_min_run(n);

    struct Run { int base, len; };
    std::vector<Run> stack;
    std::vector<T>   tmp;

    auto merge_at = [&](int i) {
        const int base = stack[i].base;
        const int mid  = stack[i].base + stack[i].len;
        const int hi   = stack[i + 1].base + stack[i + 1].len;
        detail::timsort_merge(a, base, mid, hi, tmp);
        stack[i].len += stack[i + 1].len;
        stack.erase(stack.begin() + i + 1);
    };
    auto merge_collapse = [&]() {
        while (stack.size() > 1) {
            const int s = static_cast<int>(stack.size());
            if (s >= 3 && stack[s - 3].len <= stack[s - 2].len + stack[s - 1].len) {
                if (stack[s - 3].len < stack[s - 1].len) merge_at(s - 3);
                else merge_at(s - 2);
            } else if (stack[s - 2].len <= stack[s - 1].len) {
                merge_at(s - 2);
            } else {
                break;
            }
        }
    };

    int lo = 0;
    while (lo < n) {
        int runlen = detail::count_run_and_make_ascending(a, lo, n);
        if (runlen < minrun) {
            const int force = std::min(minrun, n - lo);
            detail::binary_insertion_sort(a, lo, lo + force, lo + runlen);
            runlen = force;
        }
        stack.push_back({lo, runlen});
        merge_collapse();
        lo += runlen;
    }
    while (stack.size() > 1) merge_at(static_cast<int>(stack.size()) - 2);
}

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

namespace detail {
/// @brief Recursive step of stooge_sort() on the inclusive range a[lo..hi].
/// @tparam T Element type with `operator<`.
/// @param a Values.
/// @param lo First index.
/// @param hi Last index (inclusive).
template <typename T>
void stooge(std::vector<T>& a, int lo, int hi) { // inclusive bounds
    if (a[hi] < a[lo]) std::swap(a[lo], a[hi]);
    const int len = hi - lo + 1;
    if (len > 2) {
        const int t = len / 3;
        stooge(a, lo, hi - t);
        stooge(a, lo + t, hi);
        stooge(a, lo, hi - t);
    }
}
} // namespace detail

/// @brief Stooge sort, in place, ascending: sort the first two thirds, the last two thirds, then
///        the first two thirds again.
///
/// Correct but \f$O(n^{\log 3/\log 1.5})\approx O(n^{2.71})\f$; a classic example of an inefficient sort.
/// @tparam T Element type with `operator<`.
/// @param a Values to sort.
template <typename T>
void stooge_sort(std::vector<T>& a) {
    if (a.size() > 1) detail::stooge(a, 0, static_cast<int>(a.size()) - 1);
}

} // namespace datamunge::algorithms
