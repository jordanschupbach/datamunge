#pragma once

/// @file
/// @brief Shared validation, scaling and accounting for boxed optimizers.

#include <datamunge/optim/function_types.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::optim::detail {

/// @brief Check the dimension, order and finiteness of the box and the work budget.
/// @param dimension Coordinate vector size.
/// @param lower Lower bounds.
/// @param upper Upper bounds.
/// @param budget Objective-call budget.
/// @throws std::invalid_argument The bounds, dimension or budget are invalid.
inline void validate_box(std::size_t dimension, const std::vector<double>& lower,
                  const std::vector<double>& upper, std::size_t budget) {
  if (dimension == 0 || lower.size() != dimension || upper.size() != dimension ||
      budget == 0) {
    throw std::invalid_argument("Boxed search requires a nonempty box and positive budget");
  }
  for (std::size_t axis = 0; axis < dimension; ++axis) {
    if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) || lower[axis] > upper[axis]) {
      throw std::invalid_argument("Boxed search requires finite ordered bounds");
    }
  }
}

/// @brief Clamp a finite start and express it in unit-box coordinates.
/// @param start Finite start, clamped in place.
/// @param lower Valid lower bounds.
/// @param upper Valid upper bounds.
/// @return Clamped point in the unit box, zero on fixed axes.
/// @throws std::invalid_argument Nonfinite start.
inline std::vector<double> normalize_start(std::vector<double>& start, const std::vector<double>& lower,
                                    const std::vector<double>& upper) {
  std::vector<double> unit(start.size());
  for (std::size_t axis = 0; axis < start.size(); ++axis) {
    if (!std::isfinite(start[axis])) throw std::invalid_argument("Boxed search requires a finite start");
    start[axis] = std::clamp(start[axis], lower[axis], upper[axis]);
    const double width = upper[axis] - lower[axis];
    if (std::isinf(width)) {
      // Halve both differences before subtraction to normalize very wide boxes.
      unit[axis] = (start[axis] / 2 - lower[axis] / 2) / (upper[axis] / 2 - lower[axis] / 2);
    } else if (width > 0) {
      unit[axis] = (start[axis] - lower[axis]) / width;
    }
  }
  return unit;
}

/// @brief Score one trial and require a finite value.
/// @param function Borrowed objective.
/// @param point Feasible trial coordinates.
/// @return Finite observed value.
/// @throws std::domain_error The objective returns a nonfinite value.
inline double score(ArbitraryFunction& function, const std::vector<double>& point) {
  const double value = function.evaluate(point);
  if (!std::isfinite(value)) {
    throw std::domain_error("Boxed search requires finite objective values");
  }
  return value;
}


/// @brief Count all real nested objective calls and remember the best trial.
class CountingObjective final : public ArbitraryFunction {
public:
  /// @brief Borrow the real objective for one optimizer run.
  /// @param function Objective to score.
  explicit CountingObjective(ArbitraryFunction& function) : function_(function) {}

  /// @brief Score and retain the trial under a caller-managed budget.
  /// @param coordinates Feasible trial.
  /// @return Finite observed value.
  double evaluate(const std::vector<double>& coordinates) override {
    const auto value = score(function_, coordinates);
    ++calls;
    if (value < best_value) {
      best = coordinates;
      best_value = value;
    }
    return value;
  }

  std::size_t calls{}; ///< Completed objective calls.
  std::vector<double> best; ///< Earliest best point.
  double best_value{std::numeric_limits<double>::infinity()}; ///< Best observed loss.

private:
  ArbitraryFunction& function_; ///< Borrowed objective.
};

} // namespace datamunge::optim::detail
