#pragma once

// A quartet of sorting algorithms with distinct mechanisms:
//   - bitonic_sort  : a data-oblivious sorting network (Batcher), general n
//   - strand_sort   : repeatedly pull out an increasing run and merge it
//   - patience_sort : deal into piles, then k-way merge the pile tops
//   - flashsort     : distribution sort by value classification + permutation
// Each sorts the vector in place into ascending order.

#include <algorithm>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

template <typename T>
void bitonic_merge(std::vector<T>& a, int lo, int n, bool asc) {
    if (n <= 1) return;
    int m = 1;
    while (m * 2 < n) m *= 2; // greatest power of two < n
    for (int i = lo; i < lo + n - m; ++i)
        if (asc == (a[i] > a[i + m])) std::swap(a[i], a[i + m]);
    bitonic_merge(a, lo, m, asc);
    bitonic_merge(a, lo + m, n - m, asc);
}

template <typename T>
void bitonic_sort_rec(std::vector<T>& a, int lo, int n, bool asc) {
    if (n <= 1) return;
    const int m = n / 2;
    bitonic_sort_rec(a, lo, m, !asc);
    bitonic_sort_rec(a, lo + m, n - m, asc);
    bitonic_merge(a, lo, n, asc);
}

} // namespace detail

// Batcher's bitonic sorting network, generalised to arbitrary n.
template <typename T>
void bitonic_sort(std::vector<T>& a) {
    detail::bitonic_sort_rec(a, 0, static_cast<int>(a.size()), true);
}

// Strand sort: repeatedly extract an increasing subsequence (a "strand") from
// the input and merge it into the growing result.
template <typename T>
void strand_sort(std::vector<T>& a) {
    std::vector<T> input(a.begin(), a.end());
    std::vector<T> result;
    while (!input.empty()) {
        std::vector<T> strand, remaining;
        strand.push_back(input[0]);
        for (std::size_t i = 1; i < input.size(); ++i) {
            if (input[i] >= strand.back()) strand.push_back(input[i]);
            else remaining.push_back(input[i]);
        }
        std::vector<T> merged;
        merged.reserve(result.size() + strand.size());
        std::merge(result.begin(), result.end(), strand.begin(), strand.end(), std::back_inserter(merged));
        result.swap(merged);
        input.swap(remaining);
    }
    a.swap(result);
}

// Patience sort: deal each element onto the leftmost pile whose top is >= it
// (a new pile if none), then repeatedly extract the globally smallest pile top.
template <typename T>
void patience_sort(std::vector<T>& a) {
    std::vector<std::vector<T>> piles;
    for (const T& x : a) {
        int lo = 0, hi = static_cast<int>(piles.size());
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (piles[mid].back() >= x) hi = mid;
            else lo = mid + 1;
        }
        if (lo == static_cast<int>(piles.size())) piles.push_back({x});
        else piles[lo].push_back(x);
    }
    using Item = std::pair<T, int>; // (pile top, pile index)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    for (int i = 0; i < static_cast<int>(piles.size()); ++i) pq.emplace(piles[i].back(), i);
    std::vector<T> out;
    out.reserve(a.size());
    while (!pq.empty()) {
        const auto [val, i] = pq.top();
        pq.pop();
        out.push_back(val);
        piles[i].pop_back();
        if (!piles[i].empty()) pq.emplace(piles[i].back(), i);
    }
    a.swap(out);
}

// Flashsort: classify elements into value buckets, permute them into bucket
// order in place by cycle leaders, then finish with insertion sort.
template <typename T>
void flashsort(std::vector<T>& a) {
    const int n = static_cast<int>(a.size());
    if (n <= 1) return;
    T   minv = a[0];
    int amax = 0;
    for (int i = 1; i < n; ++i) {
        if (a[i] < minv) minv = a[i];
        if (a[i] > a[amax]) amax = i;
    }
    if (!(minv < a[amax])) return; // all equal

    const int        m  = std::max(2, static_cast<int>(0.43 * n));
    const double     c1 = static_cast<double>(m - 1) / static_cast<double>(a[amax] - minv);
    auto             cls = [&](const T& v) {
        int k = static_cast<int>(c1 * static_cast<double>(v - minv));
        if (k < 0) k = 0;
        if (k >= m) k = m - 1;
        return k;
    };
    std::vector<int> L(m, 0);
    for (int i = 0; i < n; ++i) ++L[cls(a[i])];
    for (int k = 1; k < m; ++k) L[k] += L[k - 1]; // L[k] = one past class k's range

    int move = 0, j = 0, k = m - 1;
    while (move < n - 1) {
        while (j > L[k] - 1) { ++j; k = cls(a[j]); }
        T flash = a[j];
        while (j != L[k]) {
            k              = cls(flash);
            const int idx  = --L[k];
            std::swap(flash, a[idx]);
            ++move;
        }
    }
    for (int i = 1; i < n; ++i) { // insertion sort within buckets
        T   key = a[i];
        int p   = i - 1;
        while (p >= 0 && a[p] > key) { a[p + 1] = a[p]; --p; }
        a[p + 1] = key;
    }
}

} // namespace datamunge::algorithms
