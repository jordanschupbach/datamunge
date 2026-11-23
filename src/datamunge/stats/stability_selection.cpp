/// @file stability_selection.cpp
/// @brief Paired subsampling across a grid of regularization levels.

#include <datamunge/stats/stability_selection.hpp>

#include "subsample_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace datamunge::stats {

StabilitySelectionResult stability_selection(
    const linalg::DenseMatrix<double>& predictors, const std::vector<double>& response,
    const std::vector<double>& levels, const StabilitySelector& select,
    const StabilitySelectionOptions& options) {
    const auto n = predictors.rows(), p = predictors.cols();
    if (n == 0 || p == 0 || n != response.size() || levels.empty() || !select ||
        options.subsamples == 0 || !(options.sample_fraction > 0 && options.sample_fraction <= 1) ||
        !(options.selection_threshold > .5 && options.selection_threshold <= 1) ||
        !std::ranges::all_of(predictors.data(),[](double x) { return std::isfinite(x); }) ||
        !std::ranges::all_of(response,[](double x) { return std::isfinite(x); }) ||
        !std::ranges::all_of(levels,[](double x) { return x >= 0 && std::isfinite(x); }))
        throw std::invalid_argument("stability selection: invalid data, grid or sampling controls");
    auto distinct = levels;
    std::ranges::sort(distinct);
    if (std::ranges::adjacent_find(distinct) != distinct.end())
        throw std::invalid_argument("stability selection: regularization levels must be distinct");

    const auto size = options.sample_fraction == 1 ? n
        : static_cast<std::size_t>(static_cast<double>(n)*options.sample_fraction);
    if (size == 0)
        throw std::invalid_argument("stability selection: sample fraction retains no rows");
    if (levels.size() > std::vector<std::uint64_t>{}.max_size()/p ||
        levels.size() > std::vector<double>{}.max_size()/p)
        throw std::length_error("stability selection: grid matrices exceed storage capacity");

    StabilitySelectionResult result{levels,linalg::DenseMatrix<std::uint64_t>(levels.size(),p),
        linalg::DenseMatrix<double>(levels.size(),p),std::vector<double>(p),{},size};
    std::vector<std::size_t> indices(n);
    linalg::DenseMatrix<double> x(size,p);
    std::vector<double> y(size);
    std::mt19937_64 random(options.seed);
    for (std::size_t b = 0; b < options.subsamples; ++b) {
        subsample_detail::draw_rows(indices,size,random);
        for (std::size_t i = 0; i < size; ++i) {
            const auto row = indices[i];
            y[i] = response[row];
            for (std::size_t j = 0; j < p; ++j)
                x(i,j) = predictors(row,j);
        }
        for (std::size_t k = 0; k < levels.size(); ++k) {
            auto selected = select(x,y,levels[k]);
            std::ranges::sort(selected);
            if ((!selected.empty() && selected.back() >= p) ||
                std::ranges::adjacent_find(selected) != selected.end())
                throw std::invalid_argument("stability selection: invalid or repeated selected feature");
            for (const auto j : selected)
                ++result.counts(k,j);
        }
    }
    for (std::size_t j = 0; j < p; ++j) {
        double maximum = 0;
        for (std::size_t k = 0; k < levels.size(); ++k) {
            const auto frequency = static_cast<double>(static_cast<long double>(result.counts(k,j))/options.subsamples);
            result.frequencies(k,j) = frequency;
            maximum = std::max(maximum,frequency);
        }
        result.maximum_frequency[j] = maximum;
        if (maximum >= options.selection_threshold)
            result.stable_features.push_back(j);
    }
    return result;
}

} // namespace datamunge::stats
