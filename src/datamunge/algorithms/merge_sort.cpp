#include <datamunge/algorithms/merge_sort.hpp>

#include <cstddef>

namespace datamunge::algorithms {

namespace {

// Stable top-down merge sort of data[lo, hi) using `buf` as O(n) scratch space. `key(x)` extracts
// the value the ordering is on; the same routine drives both the int64 and the keyed-pair overload.
template <typename T, typename Key>
void merge_sort_impl(std::vector<T>& data, std::vector<T>& buf, std::size_t lo, std::size_t hi,
                     Key key) {
    if (hi - lo <= 1) return; // 0 or 1 element: already sorted (base case)

    const std::size_t mid = lo + (hi - lo) / 2; // split point, avoiding overflow
    merge_sort_impl(data, buf, lo, mid, key);    // sort the left run [lo, mid)
    merge_sort_impl(data, buf, mid, hi, key);    // sort the right run [mid, hi)

    // Merge the two sorted runs into buf[lo, hi), then copy back. Each iteration of the first loop
    // consumes exactly one element with one key comparison, so the whole merge is O(hi - lo).
    std::size_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        // Take from the RIGHT run only when it is strictly smaller; on a tie the LEFT front wins,
        // which is exactly what keeps the sort STABLE (left-run elements came first in the input).
        if (key(data[j]) < key(data[i]))
            buf[k++] = data[j++];
        else
            buf[k++] = data[i++];
    }
    while (i < mid) buf[k++] = data[i++]; // drain whichever run is left
    while (j < hi) buf[k++] = data[j++];
    for (std::size_t t = lo; t < hi; ++t) data[t] = buf[t];
}

} // namespace

void merge_sort(std::vector<std::int64_t>& data) {
    if (data.size() <= 1) return;
    std::vector<std::int64_t> buf(data.size());
    merge_sort_impl(data, buf, 0, data.size(), [](std::int64_t v) { return v; });
}

void merge_sort_pairs(std::vector<std::pair<int, int>>& data) {
    if (data.size() <= 1) return;
    std::vector<std::pair<int, int>> buf(data.size());
    merge_sort_impl(data, buf, 0, data.size(), [](const std::pair<int, int>& p) { return p.first; });
}

} // namespace datamunge::algorithms
