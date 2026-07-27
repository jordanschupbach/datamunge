#include <datamunge/algorithms/distribution_sorts.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

namespace {
// Small in-place insertion sort used to order individual buckets.
void insertion_inplace(std::vector<int>& v) {
    for (std::size_t i = 1; i < v.size(); ++i) {
        const int   key = v[i];
        std::size_t j   = i;
        while (j > 0 && v[j - 1] > key) { v[j] = v[j - 1]; --j; }
        v[j] = key;
    }
}
} // namespace

std::vector<int> bucket_sort(std::vector<int> a) {
    const std::size_t n = a.size();
    if (n <= 1) return a;
    const int mn = *std::min_element(a.begin(), a.end());
    const int mx = *std::max_element(a.begin(), a.end());
    if (mn == mx) return a; // all identical

    const std::size_t              k = n; // use n buckets spanning [mn, mx]
    std::vector<std::vector<int>>  buckets(k);
    const long long                span = static_cast<long long>(mx) - mn + 1;
    for (int x : a) {
        // Map value to a bucket index in [0, k-1].
        const std::size_t idx = static_cast<std::size_t>(static_cast<long long>(x - mn) * static_cast<long long>(k) / span);
        buckets[idx].push_back(x);
    }
    std::vector<int> out;
    out.reserve(n);
    for (auto& b : buckets) {
        insertion_inplace(b);
        out.insert(out.end(), b.begin(), b.end());
    }
    return out;
}

std::vector<int> pigeonhole_sort(std::vector<int> a) {
    const std::size_t n = a.size();
    if (n <= 1) return a;
    const int         mn    = *std::min_element(a.begin(), a.end());
    const int         mx    = *std::max_element(a.begin(), a.end());
    const std::size_t range = static_cast<std::size_t>(static_cast<long long>(mx) - mn + 1);

    std::vector<std::vector<int>> holes(range); // one hole per value in [mn, mx]
    for (int x : a) holes[static_cast<std::size_t>(x - mn)].push_back(x);

    std::vector<int> out;
    out.reserve(n);
    for (auto& h : holes)
        for (int v : h) out.push_back(v);
    return out;
}

std::vector<int> cycle_sort(std::vector<int> a) {
    const std::size_t n = a.size();
    for (std::size_t start = 0; start + 1 < n; ++start) {
        int         item = a[start];
        std::size_t pos  = start;
        // Count elements smaller than `item` to find its sorted position.
        for (std::size_t i = start + 1; i < n; ++i)
            if (a[i] < item) ++pos;
        if (pos == start) continue;                 // already in place
        while (item == a[pos]) ++pos;               // skip past equal elements
        std::swap(item, a[pos]);
        // Rotate the rest of the cycle into place.
        while (pos != start) {
            pos = start;
            for (std::size_t i = start + 1; i < n; ++i)
                if (a[i] < item) ++pos;
            while (item == a[pos]) ++pos;
            std::swap(item, a[pos]);
        }
    }
    return a;
}

std::vector<int> tree_sort(std::vector<int> a) {
    const std::size_t n = a.size();
    if (n <= 1) return a;

    std::vector<int> val, left, right;
    val.reserve(n);
    left.reserve(n);
    right.reserve(n);
    auto make_node = [&](int x) {
        val.push_back(x);
        left.push_back(-1);
        right.push_back(-1);
        return static_cast<int>(val.size()) - 1;
    };

    const int root = make_node(a[0]);
    for (std::size_t i = 1; i < n; ++i) {
        const int x   = a[i];
        int       cur = root;
        while (true) {
            if (x < val[static_cast<std::size_t>(cur)]) {
                if (left[static_cast<std::size_t>(cur)] == -1) { left[static_cast<std::size_t>(cur)] = make_node(x); break; }
                cur = left[static_cast<std::size_t>(cur)];
            } else { // duplicates go right, keeping the sort stable by value
                if (right[static_cast<std::size_t>(cur)] == -1) { right[static_cast<std::size_t>(cur)] = make_node(x); break; }
                cur = right[static_cast<std::size_t>(cur)];
            }
        }
    }

    // Iterative in-order traversal (avoids deep recursion on a degenerate tree).
    std::vector<int> out;
    out.reserve(n);
    std::vector<int> stack;
    int              cur = root;
    while (cur != -1 || !stack.empty()) {
        while (cur != -1) {
            stack.push_back(cur);
            cur = left[static_cast<std::size_t>(cur)];
        }
        cur = stack.back();
        stack.pop_back();
        out.push_back(val[static_cast<std::size_t>(cur)]);
        cur = right[static_cast<std::size_t>(cur)];
    }
    return out;
}

} // namespace datamunge::algorithms
