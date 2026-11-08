#include <datamunge/algorithms/exchange_sorts.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

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
