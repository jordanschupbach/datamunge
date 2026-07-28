#pragma once

// Three distribution / bucket sorts:
//   - samplesort   : pick splitters from a sample, bucket by them, recurse
//   - postman_sort : MSD radix (hierarchical bucketing by digit) for unsigned
//   - burstsort    : cache-friendly string sort via a burst trie of buckets

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
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

// ---- Burstsort ----

namespace detail {

struct BurstNode {
    static constexpr int kBurst = 8; // burst threshold
    // For each byte value 0..255: a bucket of suffixes, or a child node.
    std::array<std::vector<std::string>, 256> buckets;
    std::array<std::unique_ptr<BurstNode>, 256> children;
    std::vector<std::string>                    ended; // strings that terminate here
    int                                         count = 0;
};

inline void burst_insert(BurstNode* node, const std::string& s, std::size_t depth) {
    if (depth == s.size()) { node->ended.push_back(s); return; }
    const unsigned char c = static_cast<unsigned char>(s[depth]);
    if (node->children[c]) { burst_insert(node->children[c].get(), s, depth + 1); return; }
    node->buckets[c].push_back(s);
    if (static_cast<int>(node->buckets[c].size()) > BurstNode::kBurst) {
        // Burst: push the bucket down into a new child node.
        auto child = std::make_unique<BurstNode>();
        for (const std::string& t : node->buckets[c]) burst_insert(child.get(), t, depth + 1);
        node->buckets[c].clear();
        node->children[c] = std::move(child);
    }
}

inline void burst_collect(BurstNode* node, std::vector<std::string>& out) {
    std::sort(node->ended.begin(), node->ended.end());
    for (auto& s : node->ended) out.push_back(s);
    for (int c = 0; c < 256; ++c) {
        if (!node->buckets[c].empty()) {
            std::sort(node->buckets[c].begin(), node->buckets[c].end());
            for (auto& s : node->buckets[c]) out.push_back(s);
        } else if (node->children[c]) {
            burst_collect(node->children[c].get(), out);
        }
    }
}

} // namespace detail

// Burstsort: insert strings into a burst trie -- shallow buckets that "burst"
// into deeper nodes once they overflow -- then traverse the trie in order.
inline void burstsort(std::vector<std::string>& a) {
    if (a.size() < 2) return;
    detail::BurstNode root;
    for (const std::string& s : a) detail::burst_insert(&root, s, 0);
    std::vector<std::string> out;
    out.reserve(a.size());
    detail::burst_collect(&root, out);
    a.swap(out);
}

} // namespace datamunge::algorithms
