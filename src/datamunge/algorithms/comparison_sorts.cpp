#include <datamunge/algorithms/comparison_sorts.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

std::vector<int> insertion_sort(std::vector<int> a) {
    for (std::size_t i = 1; i < a.size(); ++i) {
        const int   key = a[i];
        std::size_t j   = i;
        while (j > 0 && a[j - 1] > key) { // shift larger elements one place right
            a[j] = a[j - 1];
            --j;
        }
        a[j] = key;
    }
    return a;
}

std::vector<int> selection_sort(std::vector<int> a) {
    const std::size_t n = a.size();
    for (std::size_t i = 0; i + 1 < n; ++i) {
        std::size_t min_idx = i;
        for (std::size_t j = i + 1; j < n; ++j)
            if (a[j] < a[min_idx]) min_idx = j;
        if (min_idx != i) std::swap(a[i], a[min_idx]); // exactly one swap per position (at most n-1)
    }
    return a;
}

std::vector<int> shell_sort(std::vector<int> a) {
    const std::size_t n = a.size();
    // Shell's original halving gap sequence: n/2, n/4, ..., 1.
    for (std::size_t gap = n / 2; gap > 0; gap /= 2) {
        // Gapped insertion sort: insertion sort on each interleaved subsequence.
        for (std::size_t i = gap; i < n; ++i) {
            const int   key = a[i];
            std::size_t j   = i;
            while (j >= gap && a[j - gap] > key) {
                a[j] = a[j - gap];
                j -= gap;
            }
            a[j] = key;
        }
    }
    return a;
}

std::vector<int> comb_sort(std::vector<int> a) {
    const std::size_t n = a.size();
    if (n < 2) return a;
    std::size_t gap     = n;
    bool        swapped = true;
    while (gap > 1 || swapped) {
        // Shrink the gap by the factor 1.3 (until it reaches 1).
        gap = (gap * 10) / 13;
        if (gap < 1) gap = 1;
        swapped = false;
        for (std::size_t i = 0; i + gap < n; ++i)
            if (a[i] > a[i + gap]) {
                std::swap(a[i], a[i + gap]);
                swapped = true;
            }
    }
    return a;
}

} // namespace datamunge::algorithms
