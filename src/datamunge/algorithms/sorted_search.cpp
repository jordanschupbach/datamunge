#include <datamunge/algorithms/sorted_search.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

SearchResult binary_search(const std::vector<int>& a, int key) {
    SearchResult   r;
    std::ptrdiff_t lo = 0;
    std::ptrdiff_t hi = static_cast<std::ptrdiff_t>(a.size()) - 1;
    while (lo <= hi) {
        const std::ptrdiff_t mid = lo + (hi - lo) / 2;
        ++r.probes;
        if (a[static_cast<std::size_t>(mid)] == key) {
            r.found = true;
            r.index = mid;
            return r;
        }
        if (a[static_cast<std::size_t>(mid)] < key)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return r;
}

SearchResult jump_search(const std::vector<int>& a, int key) {
    SearchResult      r;
    const std::size_t n = a.size();
    if (n == 0) return r;

    const std::size_t step = std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(static_cast<double>(n))));
    // Advance block by block until the block's last element is >= key (or we run off the end).
    std::size_t block_end = step; // one past the current block's last index (exclusive upper)
    std::size_t prev      = 0;
    while (prev < n) {
        const std::size_t last = std::min(block_end, n) - 1;
        ++r.probes;
        if (a[last] >= key) break;   // key, if present, is in [prev, last]
        prev      = block_end;
        block_end += step;
        if (prev >= n) return r;     // key is larger than every element
    }
    // Linear scan within the identified block.
    const std::size_t hi = std::min(block_end, n);
    for (std::size_t i = prev; i < hi; ++i) {
        ++r.probes;
        if (a[i] == key) {
            r.found = true;
            r.index = static_cast<std::ptrdiff_t>(i);
            return r;
        }
        if (a[i] > key) break;       // passed where key would be
    }
    return r;
}

SearchResult interpolation_search(const std::vector<int>& a, int key) {
    SearchResult   r;
    std::ptrdiff_t lo = 0;
    std::ptrdiff_t hi = static_cast<std::ptrdiff_t>(a.size()) - 1;
    while (lo <= hi && key >= a[static_cast<std::size_t>(lo)] && key <= a[static_cast<std::size_t>(hi)]) {
        const int lo_val = a[static_cast<std::size_t>(lo)];
        const int hi_val = a[static_cast<std::size_t>(hi)];
        if (lo_val == hi_val) {              // flat range: the guard above guarantees key == lo_val
            ++r.probes;
            r.found = true;
            r.index = lo;
            return r;
        }
        // Linear interpolation of the key's likely position, clamped into [lo, hi].
        const long double frac = static_cast<long double>(key - lo_val) / static_cast<long double>(hi_val - lo_val);
        std::ptrdiff_t    pos  = lo + static_cast<std::ptrdiff_t>(frac * static_cast<long double>(hi - lo));
        pos                    = std::min(hi, std::max(lo, pos));
        ++r.probes;
        if (a[static_cast<std::size_t>(pos)] == key) {
            r.found = true;
            r.index = pos;
            return r;
        }
        if (a[static_cast<std::size_t>(pos)] < key)
            lo = pos + 1;
        else
            hi = pos - 1;
    }
    return r;
}

SearchResult fibonacci_search(const std::vector<int>& a, int key) {
    SearchResult      r;
    const std::size_t n = a.size();
    if (n == 0) return r;

    // Smallest Fibonacci number >= n, with its two predecessors.
    std::size_t fib2 = 0; // F(k-2)
    std::size_t fib1 = 1; // F(k-1)
    std::size_t fib  = 1; // F(k)
    while (fib < n) {
        fib2 = fib1;
        fib1 = fib;
        fib  = fib1 + fib2;
    }

    std::ptrdiff_t offset = -1; // elements up to `offset` are already eliminated
    while (fib > 1) {
        const std::size_t i =
            std::min<std::size_t>(static_cast<std::size_t>(offset) + fib2, n - 1);
        ++r.probes;
        if (a[i] < key) {          // key is in the upper (larger) two-thirds
            fib    = fib1;
            fib1   = fib2;
            fib2   = fib - fib1;
            offset = static_cast<std::ptrdiff_t>(i);
        } else if (a[i] > key) {   // key is in the lower one-third
            fib  = fib2;
            fib1 = fib1 - fib2;
            fib2 = fib - fib1;
        } else {
            r.found = true;
            r.index = static_cast<std::ptrdiff_t>(i);
            return r;
        }
    }
    // One element may remain to be checked.
    if (fib1 == 1 && offset + 1 < static_cast<std::ptrdiff_t>(n)) {
        ++r.probes;
        if (a[static_cast<std::size_t>(offset + 1)] == key) {
            r.found = true;
            r.index = offset + 1;
        }
    }
    return r;
}

namespace {

// In-order traversal of the implicit complete tree, writing successive sorted values into the
// BFS-ordered `out` array: node k (1-based) lands at out[k-1], its subtrees at 2k and 2k+1.
void fill_eytzinger(const std::vector<int>& sorted, std::vector<int>& out, std::size_t k, std::size_t& idx) {
    const std::size_t n = out.size();
    if (k > n) return;
    fill_eytzinger(sorted, out, 2 * k, idx);
    out[k - 1] = sorted[idx++];
    fill_eytzinger(sorted, out, 2 * k + 1, idx);
}

} // namespace

std::vector<int> eytzinger_layout(const std::vector<int>& sorted) {
    std::vector<int> out(sorted.size());
    std::size_t      idx = 0;
    if (!sorted.empty()) fill_eytzinger(sorted, out, 1, idx);
    return out;
}

SearchResult eytzinger_search(const std::vector<int>& layout, int key) {
    SearchResult      r;
    const std::size_t n = layout.size();
    std::size_t       k = 1; // 1-based node index; array position is k-1
    while (k <= n) {
        const int v = layout[k - 1];
        ++r.probes;
        if (v == key) {
            r.found = true;
            r.index = static_cast<std::ptrdiff_t>(k - 1);
            return r;
        }
        k = 2 * k + (v < key ? 1 : 0); // right child if the node is too small, else left child
    }
    return r;
}

} // namespace datamunge::algorithms
