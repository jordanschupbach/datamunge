#include <datamunge/algorithms/exchange_sorts.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

std::vector<int> gnome_sort(std::vector<int> a) {
    const std::size_t n = a.size();
    std::size_t       i = 0;
    while (i < n) {
        if (i == 0 || a[i - 1] <= a[i]) {
            ++i; // in order: step forward
        } else {
            std::swap(a[i - 1], a[i]);
            --i; // out of order: swap and step back
        }
    }
    return a;
}

std::vector<int> cocktail_shaker_sort(std::vector<int> a) {
    const std::size_t n = a.size();
    if (n < 2) return a;
    std::size_t lo      = 0;
    std::size_t hi      = n - 1;
    bool        swapped = true;
    while (swapped) {
        swapped = false;
        for (std::size_t i = lo; i < hi; ++i) // forward: push the largest to hi
            if (a[i] > a[i + 1]) { std::swap(a[i], a[i + 1]); swapped = true; }
        if (!swapped) break;
        --hi;
        swapped = false;
        for (std::size_t i = hi; i > lo; --i) // backward: push the smallest to lo
            if (a[i - 1] > a[i]) { std::swap(a[i - 1], a[i]); swapped = true; }
        ++lo;
    }
    return a;
}

std::vector<int> odd_even_sort(std::vector<int> a) {
    const std::size_t n = a.size();
    bool              sorted = false;
    while (!sorted) {
        sorted = true;
        for (std::size_t i = 1; i + 1 < n; i += 2) // (odd, even) pairs
            if (a[i] > a[i + 1]) { std::swap(a[i], a[i + 1]); sorted = false; }
        for (std::size_t i = 0; i + 1 < n; i += 2) // (even, odd) pairs
            if (a[i] > a[i + 1]) { std::swap(a[i], a[i + 1]); sorted = false; }
    }
    return a;
}

PancakeResult pancake_sort(std::vector<int> a) {
    PancakeResult     out;
    const std::size_t n = a.size();
    auto flip = [&](std::size_t k) { // reverse the prefix of length k
        std::reverse(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(k));
        out.flips.push_back(k);
    };
    for (std::size_t size = n; size > 1; --size) {
        // Index of the maximum within the unsorted prefix a[0..size-1].
        std::size_t maxi = 0;
        for (std::size_t i = 1; i < size; ++i)
            if (a[i] > a[maxi]) maxi = i;
        if (maxi == size - 1) continue;   // already at the bottom of the unsorted region
        if (maxi != 0) flip(maxi + 1);     // bring the max to the top
        flip(size);                        // flip it down to its final position
    }
    out.sorted = std::move(a);
    return out;
}

} // namespace datamunge::algorithms
