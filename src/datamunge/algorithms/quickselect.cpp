#include <datamunge/algorithms/quickselect.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {

namespace {

// Lomuto partition of data[lo..hi] (inclusive) around the value at pivot_index. The pivot is moved
// to the end, every element strictly smaller than it is swept to the front, and the pivot is then
// dropped into the gap. Returns the pivot's final index: everything before it is < pivot, and
// everything after is >= pivot. Values equal to the pivot (duplicates) end up in the right part,
// which is fine -- the pivot itself lands at its true sorted position.
std::size_t partition(std::vector<std::int64_t>& data, std::size_t lo, std::size_t hi,
                      std::size_t pivot_index) {
    const std::int64_t pivot_value = data[pivot_index];
    std::swap(data[pivot_index], data[hi]); // park the pivot at the end
    std::size_t store = lo;
    for (std::size_t i = lo; i < hi; ++i)
        if (data[i] < pivot_value) {
            std::swap(data[store], data[i]);
            ++store;
        }
    std::swap(data[hi], data[store]); // move the pivot into its final place
    return store;
}

// median-of-three pivot: the index (among lo, mid, hi) whose value is the median of the three.
// Cheap protection against the sorted/reverse-sorted worst case for plain Quickselect.
std::size_t median_of_three(const std::vector<std::int64_t>& data, std::size_t lo, std::size_t hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    const std::int64_t a = data[lo], b = data[mid], c = data[hi];
    if ((a <= b && b <= c) || (c <= b && b <= a)) return mid;
    if ((b <= a && a <= c) || (c <= a && a <= b)) return lo;
    return hi;
}

// Insertion-sort the tiny range data[lo..hi] (at most five elements) and return the index of its
// median element (lo + (hi-lo)/2 -- the lower median for an even-sized group).
std::size_t median_index_small(std::vector<std::int64_t>& data, std::size_t lo, std::size_t hi) {
    for (std::size_t i = lo + 1; i <= hi; ++i) {
        const std::int64_t key = data[i];
        std::size_t j = i;
        while (j > lo && data[j - 1] > key) {
            data[j] = data[j - 1];
            --j;
        }
        data[j] = key;
    }
    return lo + (hi - lo) / 2;
}

// Forward declaration: the two functions below are mutually recursive (a good pivot needs a
// selection, and selection needs a good pivot).
std::size_t mom_select_index(std::vector<std::int64_t>& data, std::size_t lo, std::size_t hi,
                             std::size_t k);

// The BFPRT pivot: split data[lo..hi] into groups of five, pack each group's median into the front
// of the range, then recursively select the median *of those medians* and return its index. A
// range of five or fewer elements is its own base case.
std::size_t median_of_medians(std::vector<std::int64_t>& data, std::size_t lo, std::size_t hi) {
    if (hi - lo < 5) return median_index_small(data, lo, hi);
    std::size_t num_medians = 0;
    for (std::size_t g = lo; g <= hi; g += 5) {
        const std::size_t group_hi = std::min(g + 4, hi);
        const std::size_t mi = median_index_small(data, g, group_hi);
        std::swap(data[lo + num_medians], data[mi]); // collect medians at data[lo..lo+num_medians)
        ++num_medians;
    }
    const std::size_t mid = lo + (num_medians - 1) / 2; // median position among the packed medians
    return mom_select_index(data, lo, lo + num_medians - 1, mid);
}

// Shared selection loop, parameterised by the pivot strategy. Partition, then recurse into the one
// side that contains rank k (or stop when the pivot lands exactly on k).
std::size_t select_index(std::vector<std::int64_t>& data, std::size_t k, bool worst_case_linear) {
    std::size_t lo = 0, hi = data.size() - 1;
    for (;;) {
        if (lo == hi) return lo;
        const std::size_t pivot = worst_case_linear ? median_of_medians(data, lo, hi)
                                                     : median_of_three(data, lo, hi);
        const std::size_t p = partition(data, lo, hi, pivot);
        if (k == p) return p;
        if (k < p)
            hi = p - 1; // rank k is in the left part
        else
            lo = p + 1; // rank k is in the right part
    }
}

// median-of-medians selection, exposed to median_of_medians for its recursive pivot search.
std::size_t mom_select_index(std::vector<std::int64_t>& data, std::size_t lo, std::size_t hi,
                             std::size_t k) {
    for (;;) {
        if (lo == hi) return lo;
        const std::size_t p = partition(data, lo, hi, median_of_medians(data, lo, hi));
        if (k == p) return p;
        if (k < p)
            hi = p - 1;
        else
            lo = p + 1;
    }
}

void validate(const std::vector<std::int64_t>& data, std::size_t k) {
    if (data.empty())
        throw std::invalid_argument("quickselect: cannot select from an empty array");
    if (k >= data.size())
        throw std::invalid_argument("quickselect: rank k is out of range (k >= data.size())");
}

} // namespace

std::int64_t quickselect(std::vector<std::int64_t> data, std::size_t k) {
    validate(data, k);
    const std::size_t idx = select_index(data, k, /*worst_case_linear=*/false);
    return data[idx];
}

std::int64_t quickselect_mom(std::vector<std::int64_t> data, std::size_t k) {
    validate(data, k);
    const std::size_t idx = select_index(data, k, /*worst_case_linear=*/true);
    return data[idx];
}

} // namespace datamunge::algorithms
