#pragma once

// Three distribution / bucket sorts:
//   - samplesort   : pick splitters from a sample, bucket by them, recurse
//   - postman_sort : MSD radix (hierarchical bucketing by digit) for unsigned
//   - burstsort    : cache-friendly string sort via a burst trie of buckets

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <datamunge/algorithms/burstsort.hpp>
#include <string>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

template <typename T>
void insertion_sort_range(std::vector<T>& a, int lo, int hi) {
    for (int i = lo + 1; i < hi; ++i) {
        T   key = a[i];
        int j   = i - 1;
        while (j >= lo && a[j] > key) { a[j + 1] = a[j]; --j; }
        a[j + 1] = key;
    }
}

template <typename T>
void samplesort_rec(std::vector<T>& a, int lo, int hi) {
    const int n = hi - lo;
    if (n <= 16) { insertion_sort_range(a, lo, hi); return; }

    const int k = 8; // buckets
    // Oversample deterministically (evenly spaced), sort the sample, pick k-1 splitters.
    std::vector<T> sample;
    const int      os = 3; // oversampling factor
    for (int i = 1; i <= k * os; ++i) sample.push_back(a[lo + (long long)i * n / (k * os + 1)]);
    std::sort(sample.begin(), sample.end());
    std::vector<T> splitters;
    for (int i = 1; i < k; ++i) splitters.push_back(sample[i * os]);

    // Classify into k buckets by upper_bound on the splitters.
    std::vector<std::vector<T>> buckets(k);
    for (int i = lo; i < hi; ++i) {
        const int b = static_cast<int>(std::upper_bound(splitters.begin(), splitters.end(), a[i]) - splitters.begin());
        buckets[b].push_back(a[i]);
    }
    // Write buckets back contiguously, then sort each in place.
    int pos = lo;
    for (auto& bk : buckets) {
        const int blo = pos;
        for (const T& x : bk) a[pos++] = x;
        // If a bucket swallowed the whole range (splitters failed to split, e.g.
        // many equal keys), the split made no progress -- sort it directly to
        // avoid infinite recursion.
        if (pos - blo == n) std::sort(a.begin() + blo, a.begin() + pos);
        else samplesort_rec(a, blo, pos);
    }
}

inline void msd_radix(std::vector<std::uint32_t>& a, int lo, int hi, int shift) {
    if (hi - lo <= 32 || shift < 0) {
        insertion_sort_range(a, lo, hi);
        return;
    }
    std::array<int, 257> count{};
    for (int i = lo; i < hi; ++i) ++count[((a[i] >> shift) & 0xFFu) + 1];
    for (int c = 1; c < 257; ++c) count[c] += count[c - 1];
    std::vector<std::uint32_t> tmp(hi - lo);
    std::array<int, 256>       cur;
    for (int c = 0; c < 256; ++c) cur[c] = count[c];
    for (int i = lo; i < hi; ++i) tmp[cur[(a[i] >> shift) & 0xFFu]++] = a[i];
    for (int i = lo; i < hi; ++i) a[i] = tmp[i - lo];
    for (int c = 0; c < 256; ++c) {
        const int blo = lo + count[c], bhi = lo + count[c + 1];
        if (bhi - blo > 1) msd_radix(a, blo, bhi, shift - 8);
    }
}

} // namespace detail

// Samplesort: bucket the data by splitters drawn from a sample, then sort each
// bucket recursively. A cache- and parallel-friendly generalisation of quicksort.
template <typename T>
void samplesort(std::vector<T>& a) {
    if (a.size() > 1) detail::samplesort_rec(a, 0, static_cast<int>(a.size()));
}

// Postman sort: most-significant-digit radix sort, bucketing 32-bit unsigned keys
// one byte at a time from the top, recursing per bucket -- the hierarchical
// bucketing a postal service uses (country, then city, then street).
inline void postman_sort(std::vector<std::uint32_t>& a) {
    if (a.size() > 1) detail::msd_radix(a, 0, static_cast<int>(a.size()), 24);
}

} // namespace datamunge::algorithms
