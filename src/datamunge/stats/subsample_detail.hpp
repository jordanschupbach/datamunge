#pragma once

/// @file subsample_detail.hpp
/// @brief Shared uniform sampling without replacement.

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

/// @brief Internal row subsampling shared by statistical estimators.
namespace datamunge::stats::subsample_detail {

/// @brief Uniformly selects a subset without replacement and restores input row order.
inline void draw_rows(std::vector<std::size_t>& indices, std::size_t size, std::mt19937_64& random) {
    std::iota(indices.begin(),indices.end(),0);
    if (size == indices.size())
        return;
    for (std::size_t i = 0; i < size; ++i) {
        const auto j = std::uniform_int_distribution<std::size_t>(i,indices.size()-1)(random);
        std::swap(indices[i],indices[j]);
    }
    std::sort(indices.begin(), indices.begin()+static_cast<std::ptrdiff_t>(size));
}

} // namespace datamunge::stats::subsample_detail
