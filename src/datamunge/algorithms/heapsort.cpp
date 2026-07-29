#include <datamunge/algorithms/heapsort.hpp>

#include <cstddef>
#include <utility>

namespace datamunge::algorithms {

namespace {

// Restore the max-heap property within data[0..heap_size) assuming the subtrees rooted at the
// children of `root` are already valid heaps. The (possibly too-small) value at `root` is sifted
// down along the path of larger children until it sits above both of its children.
void sift_down(std::vector<std::int64_t>& data, std::size_t root, std::size_t heap_size) {
    for (;;) {
        const std::size_t left = 2 * root + 1;
        const std::size_t right = 2 * root + 2;
        std::size_t largest = root;
        if (left < heap_size && data[left] > data[largest]) largest = left;
        if (right < heap_size && data[right] > data[largest]) largest = right;
        if (largest == root) return; // heap property holds here; done
        std::swap(data[root], data[largest]);
        root = largest; // follow the value down and continue
    }
}

} // namespace

void build_max_heap(std::vector<std::int64_t>& data) {
    const std::size_t n = data.size();
    if (n < 2) return; // empty or singleton is already a heap
    // The last internal node is the parent of the last element: index n/2 - 1. Sift down every
    // internal node, deepest first, so each call's child subtrees are already valid heaps.
    for (std::size_t i = n / 2; i-- > 0;) sift_down(data, i, n);
}

bool is_max_heap(const std::vector<std::int64_t>& data) {
    const std::size_t n = data.size();
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t left = 2 * i + 1;
        const std::size_t right = 2 * i + 2;
        if (left < n && data[left] > data[i]) return false;
        if (right < n && data[right] > data[i]) return false;
    }
    return true;
}

void heapsort(std::vector<std::int64_t>& data) {
    const std::size_t n = data.size();
    if (n < 2) return; // nothing to do

    build_max_heap(data); // O(n): array is now a max-heap

    // Extract-max: swap the root (largest) to the end of the active heap, shrink, and re-heapify.
    for (std::size_t end = n - 1; end > 0; --end) {
        std::swap(data[0], data[end]); // largest remaining value moves into its final position
        sift_down(data, 0, end);       // restore the heap over the reduced range data[0..end)
    }
}

} // namespace datamunge::algorithms
