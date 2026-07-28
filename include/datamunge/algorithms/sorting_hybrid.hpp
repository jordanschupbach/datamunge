#pragma once

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

// Binary insertion sort of a[lo, hi), with a[lo, start) already sorted.
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

// Length of the maximal ascending (or strictly descending, then reversed) run at
// a[lo, hi); the run is left ascending.
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

inline int timsort_min_run(int n) {
    int r = 0;
    while (n >= 64) { r |= (n & 1); n >>= 1; }
    return n + r;
}

// Merge adjacent runs a[lo, mid) and a[mid, hi), both ascending.
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

// Timsort: detect natural runs, extend short ones to `minrun` with binary
// insertion sort, and merge runs while maintaining the size invariants. (Galloping
// mode -- an extra constant-factor optimisation -- is omitted for clarity.)
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

// Bead sort ("gravity" sort) for non-negative integers.
inline void bead_sort(std::vector<unsigned>& a) {
    const std::size_t n = a.size();
    if (n == 0) return;
    unsigned mx = a[0];
    for (unsigned x : a) mx = x > mx ? x : mx;
    if (mx == 0) return; // all zero
    std::vector<std::size_t> counts(mx, 0); // counts[j] = number of beads in column j
    for (unsigned x : a)
        for (unsigned j = 0; j < x; ++j) ++counts[j];
    for (std::size_t r = 0; r < n; ++r) {
        std::size_t beads = 0;
        for (unsigned j = 0; j < mx; ++j)
            if (counts[j] > r) ++beads;
        a[n - 1 - r] = static_cast<unsigned>(beads); // r-th largest
    }
}

namespace detail {
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

// Stooge sort: recursively sort the first 2/3, the last 2/3, then the first 2/3
// again. Correct but O(n^2.71) -- a classic example of a badly inefficient sort.
template <typename T>
void stooge_sort(std::vector<T>& a) {
    if (a.size() > 1) detail::stooge(a, 0, static_cast<int>(a.size()) - 1);
}

} // namespace datamunge::algorithms
