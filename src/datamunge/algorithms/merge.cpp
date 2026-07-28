#include <datamunge/algorithms/merge.hpp>

#include <cstddef>
#include <queue>
#include <tuple>
#include <vector>

namespace datamunge::algorithms {

std::vector<int> simple_merge(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out;
    out.reserve(a.size() + b.size());
    std::size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (b[j] < a[i]) // strict: equal elements take a before b, so the merge is stable
            out.push_back(b[j++]);
        else
            out.push_back(a[i++]);
    }
    while (i < a.size()) out.push_back(a[i++]);
    while (j < b.size()) out.push_back(b[j++]);
    return out;
}

std::vector<int> k_way_merge(const std::vector<std::vector<int>>& lists) {
    // Heap entry: (value, list index, element index) with the smallest value on top.
    using Node = std::tuple<int, std::size_t, std::size_t>;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> heap;

    std::size_t total = 0;
    for (std::size_t l = 0; l < lists.size(); ++l) {
        total += lists[l].size();
        if (!lists[l].empty()) heap.emplace(lists[l][0], l, 0);
    }

    std::vector<int> out;
    out.reserve(total);
    while (!heap.empty()) {
        const auto [val, l, e] = heap.top();
        heap.pop();
        out.push_back(val);
        if (e + 1 < lists[l].size()) heap.emplace(lists[l][e + 1], l, e + 1);
    }
    return out;
}

std::vector<int> union_merge(const std::vector<std::vector<int>>& lists) {
    using Node = std::tuple<int, std::size_t, std::size_t>;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> heap;

    for (std::size_t l = 0; l < lists.size(); ++l)
        if (!lists[l].empty()) heap.emplace(lists[l][0], l, 0);

    std::vector<int> out;
    bool             have_last = false;
    int              last      = 0;
    while (!heap.empty()) {
        const auto [val, l, e] = heap.top();
        heap.pop();
        if (!have_last || val != last) { // skip a value equal to the one just emitted
            out.push_back(val);
            last      = val;
            have_last = true;
        }
        if (e + 1 < lists[l].size()) heap.emplace(lists[l][e + 1], l, e + 1);
    }
    return out;
}

} // namespace datamunge::algorithms
