#pragma once

// Uniform binary search: a binary search whose sequence of step sizes is fixed
// in advance -- a precomputed schedule of powers of two -- rather than recomputed
// as a data-dependent midpoint (lo + hi) / 2 at each iteration. Each step is a
// single add and compare, and the same schedule is reused across many searches of
// an array of the given size.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

// Precompute the descending step schedule for a sorted array of length n:
// {2^k, 2^(k-1), ..., 1} with 2^k the largest power of two <= n (empty if n < 1).
inline std::vector<int> uniform_search_table(int n) {
    std::vector<int> steps;
    if (n <= 0) return steps;
    int s = 1;
    while (s * 2 <= n) s *= 2; // largest power of two <= n
    for (; s >= 1; s >>= 1) steps.push_back(s);
    return steps;
}

// Search sorted `a` for `key` using the fixed step schedule `steps` (from
// uniform_search_table(a.size())). Returns an index of a matching element, or
// std::size_t(-1) if not found. Every probe stays in bounds by construction.
template <typename T>
std::size_t uniform_binary_search(const std::vector<T>& a, const T& key, const std::vector<int>& steps) {
    const int n = static_cast<int>(a.size());
    if (n == 0) return static_cast<std::size_t>(-1);
    int i = -1; // largest index with a[i] <= key so far
    for (int step : steps)
        if (i + step < n && a[i + step] <= key) i += step;
    if (i >= 0 && a[i] == key) return static_cast<std::size_t>(i);
    return static_cast<std::size_t>(-1);
}

} // namespace datamunge::algorithms
