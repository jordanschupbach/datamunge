/// @file isotonic_regression.cpp
/// @brief Weighted pool-adjacent-violators fit over sorted predictor ties.

#include <datamunge/stats/isotonic_regression.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Working blocks with local scales for summing weights.
namespace {
/// @brief Contiguous range of sorted observations with an inclusive start and exclusive end.
struct Pool {
    std::size_t first; ///< First position in the ordering permutation.
    std::size_t end;   ///< One past the last position in the ordering permutation.
    long double scale;  ///< Largest case weight in the block.
    long double weight; ///< Sum of case weights divided by scale.
    long double level;  ///< Weighted block mean.
};

/// @brief Pools adjacent ranges using a common local weight scale.
Pool merge(const Pool& left, const Pool& right) {
    const auto scale = std::max(left.scale, right.scale);
    const auto a = left.weight * (left.scale / scale), b = right.weight * (right.scale / scale);
    const auto sum = a + b;
    const auto level = a >= b ? std::lerp(left.level, right.level, b / sum)
                              : std::lerp(right.level, left.level, a / sum);
    return {left.first, right.end, scale, sum, level};
}
} // namespace

IsotonicModel fit_isotonic(const std::vector<double>& x, const std::vector<double>& response,
    const IsotonicOptions& options) {
    const auto n = x.size();
    const bool increasing = options.direction == IsotonicDirection::Increasing;
    const auto finite = [](double value) { return std::isfinite(value); };
    if (n == 0 || response.size() != n || (!options.weights.empty() && options.weights.size() != n) ||
        (!increasing && options.direction != IsotonicDirection::Decreasing) ||
        !std::ranges::all_of(x, finite) || !std::ranges::all_of(response, finite) ||
        !std::ranges::all_of(options.weights, [&](double w) { return w > 0 && finite(w); }))
        throw std::invalid_argument("isotonic regression: invalid data, weights or direction");
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::ranges::stable_sort(order, {}, [&](std::size_t i) { return x[i]; });
    std::vector<Pool> pools;
    pools.reserve(n);
    /// @brief Initializes one sorted observation with its original response and weight.
    const auto singleton = [&](std::size_t i) {
        const auto row = order[i];
        return Pool{i, i + 1, options.weights.empty() ? 1 : options.weights[row], 1, response[row]};
    };
    for (std::size_t i = 0; i < n;) {
        auto pool = singleton(i++);
        while (i < n && x[order[i]] == x[order[pool.first]]) {
            pool = merge(pool, singleton(i));
            ++i;
        }
        while (!pools.empty()) {
            const auto& left = pools.back();
            if (increasing ? left.level < pool.level : left.level > pool.level)
                break;
            pool = merge(left, pool);
            pools.pop_back();
        }
        pools.push_back(pool);
    }

    IsotonicModel result;
    result.fitted_.resize(n);
    result.blocks_.reserve(pools.size());
    for (const auto& pool : pools) {
        const auto level = static_cast<double>(pool.level);
        const auto first = x[order[pool.first]], last = x[order[pool.end - 1]];
        if (!result.blocks_.empty() && result.blocks_.back().level == level) {
            result.blocks_.back().last_x = last;
            result.blocks_.back().observations += pool.end - pool.first;
        } else {
            result.blocks_.push_back({first, last, level, pool.end - pool.first});
        }
        for (std::size_t j = pool.first; j < pool.end; ++j)
            result.fitted_[order[j]] = level;
    }
    return result;
}

double IsotonicModel::predict(double x) const {
    if (!std::isfinite(x))
        throw std::invalid_argument("isotonic regression: query must be finite");
    if (blocks_.empty())
        throw std::logic_error("isotonic regression: model has no fitted blocks");
    const auto next = std::ranges::upper_bound(blocks_, x, {}, &IsotonicBlock::first_x);
    return (next == blocks_.begin() ? next : std::prev(next))->level;
}

std::vector<double> IsotonicModel::predict(const std::vector<double>& x) const {
    std::vector<double> result;
    result.reserve(x.size());
    for (double value : x)
        result.push_back(predict(value));
    return result;
}

} // namespace datamunge::stats
