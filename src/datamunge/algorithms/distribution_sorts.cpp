#include <datamunge/algorithms/distribution_sorts.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

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
