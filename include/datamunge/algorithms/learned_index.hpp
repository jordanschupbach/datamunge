#pragma once

/// @file learned_index.hpp
/// @brief Piecewise linear rank prediction with certified integer search bounds.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief An owning lower-bound index for sorted signed 32-bit integer keys.
/// @par Example
/// @code
/// const dm::algorithms::LearnedIndex index({1, 1, 4, 8}, 4);
/// std::print("{}\n", index.lower_bound(4)); // 2
/// @endcode
class LearnedIndex {
public:
    /// @brief Fits one least-squares rank predictor per fixed-width integer interval.
    /// @param keys Nonempty sorted keys, including any duplicates.
    /// @param width Positive interval size in key units. Intervals start at the first key.
    /// @throws std::invalid_argument Invalid keys or width.
    /// @throws std::length_error The key count exceeds the signed 32-bit rank range
    /// or more than one million intervals would be needed.
    /// @par Example
    /// See the class example.
    LearnedIndex(std::vector<std::int32_t> keys, std::int64_t width)
        : keys_(std::move(keys)), width_(width) {
        if (keys_.empty() || width_ <= 0 || !std::ranges::is_sorted(keys_))
            throw std::invalid_argument("LearnedIndex requires sorted keys and positive width");
        if (keys_.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::length_error("LearnedIndex rank exceeds supported range");
        const auto count = (std::int64_t(keys_.back()) - keys_.front()) / width_ + 1;
        if (count > 1000000) throw std::length_error("LearnedIndex interval count exceeds one million");
        segments_.reserve(static_cast<std::size_t>(count));
        const auto limit = std::int64_t(keys_.back());
        auto first = keys_.begin();
        for (std::int64_t i = 0; i < count; ++i) {
            const auto start = keys_.front() + i * width_;
            const auto end = start + std::min(width_ - 1, limit - start);
            const auto last = std::upper_bound(first, keys_.end(), end);
            Segment model;
            double mean_x = 0, mean_y = 0, covariance = 0, variance = 0;
            std::int64_t n = 0;
            for (auto run = first; run != last;) {
                const auto next = std::upper_bound(run, last, *run);
                const auto weight = next - run;
                const double x = double(*run - start) / double(width_);
                const double y = double(run - keys_.begin());
                const double dx = x - mean_x, dy = y - mean_y;
                const double factor = double(n) * double(weight) / double(n + weight);
                covariance += factor * dx * dy;
                variance += factor * dx * dx;
                n += weight;
                mean_x += dx * double(weight) / double(n);
                mean_y += dy * double(weight) / double(n);
                run = next;
            }
            model.slope = variance > 0 ? covariance / variance : 0;
            model.intercept = n ? mean_y - model.slope * mean_x : double(first - keys_.begin());

            const auto include_error = [&](std::int64_t q, std::int64_t rank) {
                const auto error = rank - estimate(model, q - start);
                model.min_error = std::min(model.min_error, error);
                model.max_error = std::max(model.max_error, error);
            };
            auto run = first;
            auto cursor = start;
            while (run != last) {
                const auto key = std::int64_t(*run);
                const auto rank = run - keys_.begin();
                include_error(cursor, rank);
                include_error(key, rank);
                cursor = key + 1;
                run = std::upper_bound(run, last, *run);
            }
            if (cursor <= end) {
                include_error(cursor, last - keys_.begin());
                include_error(end, last - keys_.begin());
            }
            segments_.push_back(model);
            first = last;
        }
    }

    /// @brief Returns the inclusive interval that contains the exact insertion rank.
    /// @param key Query key.
    /// @return Two bounds in [0, size]. The upper bound can be returned by a half-open search.
    /// @par Example
    /// @code
    /// const auto bounds = dm::algorithms::LearnedIndex({1, 3}, 4).rank_bounds(2);
    /// @endcode
    /// @see lower_bound()
    [[nodiscard]] std::pair<std::size_t, std::size_t> rank_bounds(std::int32_t key) const {
        if (key < keys_.front()) return {0, 0};
        if (key > keys_.back()) return {keys_.size(), keys_.size()};
        const auto delta = std::int64_t(key) - keys_.front();
        const auto& model = segments_[static_cast<std::size_t>(delta / width_)];
        const auto rank = estimate(model, delta % width_);
        const auto clamp = [&](std::int64_t value) {
            return static_cast<std::size_t>(std::clamp(value, std::int64_t{0}, std::int64_t(keys_.size())));
        };
        return {clamp(rank + model.min_error), clamp(rank + model.max_error)};
    }

    /// @brief Finds the first key greater than or equal to the query by bounded binary search.
    /// @param key Query key.
    /// @return Zero-based insertion rank, or key count if all keys are smaller.
    /// @par Example
    /// See the class example.
    /// @see rank_bounds()
    [[nodiscard]] std::size_t lower_bound(std::int32_t key) const {
        const auto [lo, hi] = rank_bounds(key);
        return static_cast<std::size_t>(std::lower_bound(keys_.begin() + lo, keys_.begin() + hi, key) - keys_.begin());
    }

private:
    /// @brief Fitted rank coefficients and extremal signed residuals.
    struct Segment {
        double intercept{}; ///< Rank at the interval origin before flooring.
        double slope{}; ///< Rank change over one full interval.
        std::int64_t min_error{}; ///< Lower bound on true rank minus predicted rank.
        std::int64_t max_error{}; ///< Upper bound on true rank minus predicted rank.
    };
    /// @brief Evaluates the integer rank predictor at a nonnegative interval offset.
    [[nodiscard]] std::int64_t estimate(const Segment& model, std::int64_t offset) const {
        const double predicted = model.intercept + model.slope * (double(offset) / double(width_));
        return static_cast<std::int64_t>(std::clamp(std::floor(predicted), 0.0, double(keys_.size())));
    }

    std::vector<std::int32_t> keys_; ///< Sorted owned keys.
    std::int64_t width_; ///< Interval width.
    std::vector<Segment> segments_; ///< Interval models in ascending key order.
};

} // namespace datamunge::algorithms
