#include <datamunge/algorithms/permutations.hpp>

#include <datamunge/random/random.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

std::vector<int> fisher_yates_shuffle(const std::vector<int>& items, std::uint64_t seed) {
    std::vector<int>             a = items;
    datamunge::random::SplitMix64 rng(seed);
    // Durstenfeld's in-place form: for i from n-1 down to 1, swap a[i] with a uniform j in [0, i].
    for (std::size_t i = a.size(); i-- > 1;) {
        const std::uint64_t j = rng.uniform_u64(0, i); // inclusive of both ends
        std::swap(a[i], a[static_cast<std::size_t>(j)]);
    }
    return a;
}

namespace {

// Recursive core of Heap's algorithm: emit every permutation of a[0..k-1] (the tail a[k..] fixed).
void heap_generate(std::vector<int>& a, std::size_t k, std::vector<std::vector<int>>& out) {
    if (k == 1) {
        out.push_back(a);
        return;
    }
    for (std::size_t i = 0; i + 1 < k; ++i) {
        heap_generate(a, k - 1, out);
        if (k % 2 == 0)
            std::swap(a[i], a[k - 1]);   // even k: swap the i-th and last
        else
            std::swap(a[0], a[k - 1]);   // odd k: swap the first and last
    }
    heap_generate(a, k - 1, out);
}

} // namespace

std::vector<std::vector<int>> heap_permutations(const std::vector<int>& items) {
    std::vector<std::vector<int>> out;
    std::vector<int>              a = items;
    if (a.size() <= 1) {
        out.push_back(a); // 0! = 1! = 1 permutation (possibly empty)
        return out;
    }
    heap_generate(a, a.size(), out);
    return out;
}

std::vector<std::vector<int>> sjt_permutations(std::size_t n) {
    std::vector<std::vector<int>> out;
    if (n == 0) {
        out.emplace_back(); // the single empty permutation
        return out;
    }

    std::vector<int> perm(n);
    for (std::size_t i = 0; i < n; ++i) perm[i] = static_cast<int>(i + 1);
    // Direction of each *value* 1..n: -1 points left, +1 points right. All start pointing left.
    std::vector<int> dir(n + 1, -1);
    out.push_back(perm);

    while (true) {
        // Find the largest mobile element: one whose direction points at a smaller neighbour.
        int largest = 0;
        int at      = -1;
        for (int i = 0; i < static_cast<int>(n); ++i) {
            const int v = perm[static_cast<std::size_t>(i)];
            const int j = i + dir[static_cast<std::size_t>(v)];
            if (j >= 0 && j < static_cast<int>(n) && perm[static_cast<std::size_t>(j)] < v && v > largest) {
                largest = v;
                at      = i;
            }
        }
        if (largest == 0) break; // no mobile element: all permutations generated

        const int j = at + dir[static_cast<std::size_t>(largest)];
        std::swap(perm[static_cast<std::size_t>(at)], perm[static_cast<std::size_t>(j)]);
        // Reverse the direction of every element greater than the one just moved.
        for (int v = largest + 1; v <= static_cast<int>(n); ++v)
            dir[static_cast<std::size_t>(v)] = -dir[static_cast<std::size_t>(v)];
        out.push_back(perm);
    }
    return out;
}

YoungTableaux rsk_insert(const std::vector<int>& permutation) {
    YoungTableaux out;
    for (std::size_t step = 0; step < permutation.size(); ++step) {
        int         x   = permutation[step];
        std::size_t row = 0;
        while (true) {
            if (row == out.p.size()) {              // fell off the bottom: start a new row
                out.p.push_back({x});
                out.q.push_back({static_cast<int>(step + 1)});
                break;
            }
            std::vector<int>& r = out.p[row];
            // Find the leftmost entry strictly greater than x (the row is increasing).
            std::size_t idx = 0;
            while (idx < r.size() && r[idx] <= x) ++idx;
            if (idx == r.size()) {                  // no larger entry: append and record the new box
                r.push_back(x);
                out.q[row].push_back(static_cast<int>(step + 1));
                break;
            }
            std::swap(r[idx], x);                    // bump: x replaces r[idx], the old value moves down
            ++row;
        }
    }
    return out;
}

} // namespace datamunge::algorithms
