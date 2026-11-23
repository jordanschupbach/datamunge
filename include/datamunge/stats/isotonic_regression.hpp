#pragma once

/// @file isotonic_regression.hpp
/// @brief Weighted monotone least squares and step predictions.

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Direction of the fitted step function as predictors increase.
/// @par Example
/// @code
/// const auto direction = dm::stats::IsotonicDirection::Decreasing;
/// std::println("{}", direction == dm::stats::IsotonicDirection::Decreasing); // true
/// @endcode
enum class IsotonicDirection {
    Increasing, ///< Nondecreasing fitted levels.
    Decreasing  ///< Nonincreasing fitted levels.
};

/// @brief Case weights and monotonicity for isotonic regression.
/// @par Example
/// @code
/// const dm::stats::IsotonicOptions options{.weights = {1, 2, 1}};
/// std::println("{}", options.weights.size()); // 3
/// @endcode
struct IsotonicOptions {
    IsotonicDirection direction{IsotonicDirection::Increasing}; ///< Required order of the fitted levels.
    std::vector<double> weights{}; ///< Positive finite weights in input row order, or empty for unit weights.
};

/// @brief Training interval fitted to one constant level.
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_isotonic({1,2,3}, {1,4,2});
/// std::println("{:.1f}", fit.blocks().back().level); // 3.0
/// @endcode
struct IsotonicBlock {
    double first_x;           ///< Smallest observed predictor pooled in the block.
    double last_x;            ///< Largest observed predictor pooled in the block.
    double level;             ///< Weighted mean of the pooled responses.
    std::size_t observations; ///< Number of input rows in the block, including tied predictors.
};

/// @brief Owns the monotone fit, pooled blocks, and right-continuous predictor.
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_isotonic({1,2,3}, {1,4,2});
/// std::println("{}", fit.blocks().size()); // 2
/// @endcode
class IsotonicModel {
public:
    /// @brief Borrows pooled blocks sorted by their smallest predictor.
    /// @return Reference valid while the fitted model lives.
    [[nodiscard]] const std::vector<IsotonicBlock>& blocks() const noexcept { return blocks_; }

    /// @brief Borrows fitted levels in the original row order.
    /// @return Reference valid while the fitted model lives.
    [[nodiscard]] const std::vector<double>& fitted() const noexcept { return fitted_; }

    /// @brief Evaluates the right-continuous step function at a finite query.
    ///
    /// Uses the level whose block starts at the latest predictor at or below x.
    /// At either end of the training range, the nearest endpoint level is used.
    /// @param x Finite query predictor.
    /// @return Fitted level at x.
    /// @throws std::invalid_argument A nonfinite query was supplied.
    /// @throws std::logic_error The model has no blocks, for example after moving from it.
    /// @par Example
    /// @code
    /// const auto fit = dm::stats::fit_isotonic({1,2,3}, {1,4,2});
    /// std::println("{:.1f}", fit.predict(1.9)); // 1.0
    /// @endcode
    [[nodiscard]] double predict(double x) const;

    /// @brief Evaluates the step function at several finite queries.
    /// @param x Query predictors. Empty input returns an empty vector.
    /// @return Fitted levels in query order.
    /// @throws std::invalid_argument A nonfinite query was supplied.
    /// @throws std::logic_error A nonempty query was evaluated on a model without blocks.
    /// @par Example
    /// @code
    /// const auto fit = dm::stats::fit_isotonic({1,2,3}, {1,4,2});
    /// std::println("{:.1f}", fit.predict(std::vector<double>{1.9,2.1})[1]); // 3.0
    /// @endcode
    [[nodiscard]] std::vector<double> predict(const std::vector<double>& x) const;

private:
    /// @brief Creates storage to be populated by fit_isotonic().
    IsotonicModel() = default;
    std::vector<IsotonicBlock> blocks_; ///< Final blocks in ascending predictor order.
    std::vector<double> fitted_;       ///< Level assigned to every input observation.

    friend IsotonicModel fit_isotonic(const std::vector<double>&, const std::vector<double>&,
                                       const IsotonicOptions&);
};

/// @brief Fits weighted isotonic least squares by pooling adjacent violators.
///
/// Sorts by predictor, pools tied predictors into a single level, and then
/// enforces the requested order. The pooling pass is linear after sorting.
/// @param x Nonempty finite predictor values, in any order.
/// @param response Finite observed responses aligned with the predictors.
/// @param options Monotonicity direction and optional positive case weights.
/// @return Pooled blocks, original-order fitted levels, and a step predictor.
/// @throws std::invalid_argument Empty input, incompatible lengths, or invalid values or options.
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_isotonic({1,2,3}, {1,4,2}, {.weights = {1,1,3}});
/// std::println("{:.1f}", fit.predict(3.0)); // 2.5
/// @endcode
/// @par References
/// - Ayer et al. (1955), https://doi.org/10.1214/aoms/1177728423
/// - Best and Chakravarti (1990), https://doi.org/10.1007/BF01580873
[[nodiscard]] IsotonicModel fit_isotonic(const std::vector<double>& x, const std::vector<double>& response,
    const IsotonicOptions& options = {});

} // namespace datamunge::stats
