#pragma once

// Subset-sum by meet-in-the-middle (Horowitz-Sahni): decide whether some subset
// of `nums` sums to `target`, and return one such subset. Splitting the n items
// in half and matching subset sums across the halves costs O(2^(n/2)) time
// instead of the O(2^n) of exhaustive search. Works with negative values.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

struct SubsetSumResult {
    bool                     found{false};
    std::vector<std::size_t> indices; // indices of a subset summing to target
};

inline SubsetSumResult subset_sum(const std::vector<long long>& nums, long long target) {
    const int  n    = static_cast<int>(nums.size());
    const int  na   = n / 2;
    const int  nb   = n - na;

    // All subset sums of the second half, paired with their bitmask, sorted.
    std::vector<std::pair<long long, std::uint32_t>> sb;
    sb.reserve(std::size_t{1} << nb);
    for (std::uint32_t m = 0; m < (std::uint32_t{1} << nb); ++m) {
        long long s = 0;
        for (int j = 0; j < nb; ++j)
            if (m & (std::uint32_t{1} << j)) s += nums[na + j];
        sb.emplace_back(s, m);
    }
    std::sort(sb.begin(), sb.end(),
              [](const auto& x, const auto& y) { return x.first < y.first; });

    for (std::uint32_t ma = 0; ma < (std::uint32_t{1} << na); ++ma) {
        long long sa = 0;
        for (int j = 0; j < na; ++j)
            if (ma & (std::uint32_t{1} << j)) sa += nums[j];
        const long long need = target - sa;

        auto it = std::lower_bound(sb.begin(), sb.end(), need,
                                   [](const auto& pr, long long v) { return pr.first < v; });
        if (it != sb.end() && it->first == need) {
            SubsetSumResult r;
            r.found = true;
            for (int j = 0; j < na; ++j)
                if (ma & (std::uint32_t{1} << j)) r.indices.push_back(static_cast<std::size_t>(j));
            for (int j = 0; j < nb; ++j)
                if (it->second & (std::uint32_t{1} << j)) r.indices.push_back(static_cast<std::size_t>(na + j));
            return r;
        }
    }
    return {};
}

} // namespace datamunge::algorithms
