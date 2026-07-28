#include <datamunge/algorithms/branch_and_bound.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <vector>

namespace datamunge::algorithms {

namespace {

struct Item {
    int         value;
    int         weight;
    std::size_t original; // index in the caller's input
};

// Fractional-knapsack upper bound from item `level` onward, starting with `curr_value`/`curr_weight`.
double bound(const std::vector<Item>& items, std::size_t level, int curr_weight, int curr_value, int capacity) {
    double b   = curr_value;
    int    w   = curr_weight;
    for (std::size_t i = level; i < items.size(); ++i) {
        if (w + items[i].weight <= capacity) {
            w += items[i].weight;
            b += items[i].value;
        } else {
            const int room = capacity - w;
            b += static_cast<double>(items[i].value) * room / items[i].weight; // take a fraction
            break;
        }
    }
    return b;
}

} // namespace

KnapsackResult knapsack_branch_and_bound(const std::vector<int>& values, const std::vector<int>& weights,
                                         int capacity) {
    const std::size_t n = values.size();
    KnapsackResult    result;
    result.chosen.assign(n, 0);
    if (n == 0) return result;

    // Sort items by value density (value/weight) descending -- the order the bound assumes.
    std::vector<Item> items(n);
    for (std::size_t i = 0; i < n; ++i) items[i] = Item{values[i], weights[i], i};
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return static_cast<double>(a.value) * b.weight > static_cast<double>(b.value) * a.weight;
    });

    int              best_value = 0;
    std::vector<int> best_take(n, 0); // take[k] over the sorted order
    std::vector<int> take(n, 0);

    // DFS over the sorted items via an explicit recursive lambda.
    std::function<void(std::size_t, int, int)> dfs = [&](std::size_t level, int weight, int value) {
        ++result.nodes_explored;
        if (value > best_value) {
            best_value = value;
            best_take  = take;
        }
        if (level == n) return;
        if (bound(items, level, weight, value, capacity) <= best_value) return; // prune

        // Branch 1: take item `level` if it fits.
        if (weight + items[level].weight <= capacity) {
            take[level] = 1;
            dfs(level + 1, weight + items[level].weight, value + items[level].value);
            take[level] = 0;
        }
        // Branch 2: skip item `level`.
        dfs(level + 1, weight, value);
    };
    dfs(0, 0, 0);

    result.value = best_value;
    for (std::size_t k = 0; k < n; ++k)
        if (best_take[k]) result.chosen[items[k].original] = 1;
    return result;
}

} // namespace datamunge::algorithms
