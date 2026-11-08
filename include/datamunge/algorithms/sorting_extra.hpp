#pragma once
#include <datamunge/algorithms/bitonic_sort.hpp>
#include <datamunge/algorithms/flashsort.hpp>

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

} // namespace datamunge::algorithms
