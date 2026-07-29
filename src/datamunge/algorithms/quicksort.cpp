#include <datamunge/algorithms/quicksort.hpp>

#include <cstddef>
#include <utility>

namespace datamunge::algorithms {

namespace {

// Ranges of at most this many elements are finished with insertion sort rather than partitioned:
// on tiny inputs the lower constant factor of insertion sort beats recursion and pivot selection.
constexpr std::ptrdiff_t kInsertionCutoff = 16;

// Sort a[lo..hi] (inclusive) with insertion sort. Used for the small-subarray cutoff; O(k^2) on a
// range of length k but with a tiny constant, and O(k) on nearly-sorted ranges.
void insertion_sort(std::int64_t* a, std::ptrdiff_t lo, std::ptrdiff_t hi, std::uint64_t& comparisons) {
    for (std::ptrdiff_t i = lo + 1; i <= hi; ++i) {
        const std::int64_t key = a[i];
        std::ptrdiff_t j = i - 1;
        while (j >= lo) {
            ++comparisons;
            if (a[j] <= key) break; // key belongs just after a[j]
            a[j + 1] = a[j];        // shift the larger element up one slot
            --j;
        }
        a[j + 1] = key;
    }
}

// Median-of-three pivot selection: order a[lo] <= a[mid] <= a[hi] with three comparisons, leaving
// the median of the three at a[mid]. Returns mid. Sampling the ends and centre makes an
// already-sorted or reverse-sorted range partition evenly (its true median is picked), which is
// what defuses Quicksort's O(n^2) worst case on those adversarial inputs.
std::ptrdiff_t median_of_three(std::int64_t* a, std::ptrdiff_t lo, std::ptrdiff_t hi, std::uint64_t& comparisons) {
    const std::ptrdiff_t mid = lo + (hi - lo) / 2;
    ++comparisons;
    if (a[mid] < a[lo]) std::swap(a[mid], a[lo]);
    ++comparisons;
    if (a[hi] < a[lo]) std::swap(a[hi], a[lo]);
    ++comparisons;
    if (a[hi] < a[mid]) std::swap(a[hi], a[mid]);
    return mid; // a[lo] <= a[mid] <= a[hi]; the median sits at mid
}

// Lomuto partition of a[lo..hi] around the median-of-three pivot. The pivot is parked at a[hi],
// then every element strictly less than it is swept to the front; finally the pivot is dropped into
// the gap. Returns the pivot's final index p, with a[lo..p-1] < pivot and a[p+1..hi] >= pivot.
std::ptrdiff_t partition(std::int64_t* a, std::ptrdiff_t lo, std::ptrdiff_t hi, std::uint64_t& comparisons) {
    const std::ptrdiff_t mid = median_of_three(a, lo, hi, comparisons);
    std::swap(a[mid], a[hi]); // move the chosen pivot out of the way, to the end of the range
    const std::int64_t pivot = a[hi];
    std::ptrdiff_t store = lo; // a[lo..store-1] hold the elements known to be < pivot
    for (std::ptrdiff_t j = lo; j < hi; ++j) {
        ++comparisons;
        if (a[j] < pivot) {
            std::swap(a[store], a[j]);
            ++store;
        }
    }
    std::swap(a[store], a[hi]); // pivot into its final resting place
    return store;
}

// In-place Quicksort of a[lo..hi] with tail-call elimination: recurse on the smaller partition and
// loop on the larger, so the live recursion depth stays O(log n) regardless of how the partitions
// fall. Ranges at or below the cutoff are handed to insertion sort.
void quicksort_range(std::int64_t* a, std::ptrdiff_t lo, std::ptrdiff_t hi, std::uint64_t& comparisons) {
    while (hi - lo + 1 > kInsertionCutoff) {
        const std::ptrdiff_t p = partition(a, lo, hi, comparisons);
        if (p - lo < hi - p) { // left side smaller: recurse left, iterate on the right
            quicksort_range(a, lo, p - 1, comparisons);
            lo = p + 1;
        } else { // right side smaller (or equal): recurse right, iterate on the left
            quicksort_range(a, p + 1, hi, comparisons);
            hi = p - 1;
        }
    }
    insertion_sort(a, lo, hi, comparisons);
}

} // namespace

void quicksort(std::vector<std::int64_t>& data, std::uint64_t& comparisons) {
    comparisons = 0;
    if (data.size() < 2) return; // empty or singleton is already sorted
    quicksort_range(data.data(), 0, static_cast<std::ptrdiff_t>(data.size()) - 1, comparisons);
}

void quicksort(std::vector<std::int64_t>& data) {
    std::uint64_t ignored = 0;
    quicksort(data, ignored);
}

} // namespace datamunge::algorithms
