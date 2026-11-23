#pragma once
/// @file discovery_bounds.hpp
/// @brief Adaptive q-values and simultaneous lower bounds on true discoveries.

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Fixed-threshold Storey estimate of the null fraction and ordered q-values.
/// @par Example
/// @code
/// const auto fit = dm::stats::storey_qvalues({.01, .03, .9}, {0});
/// std::print("{:.2f}\n", fit.front().null_fraction); // 1.00
/// @endcode
/// @see storey_qvalues()
struct StoreyQValueFit {
  double threshold;             ///< Lambda used for this estimate.
  double null_fraction;         ///< Estimated fraction of true nulls, capped at one.
  std::vector<double> q_values;  ///< Monotone in p-value rank, stored in original hypothesis order.
};

/// @brief Estimates Storey q-values at each supplied tuning threshold.
///
/// Estimates the null fraction from the count of p-values strictly above
/// lambda divided by m times (1-lambda), capped at one. The optional add-one
/// correction is enabled by default. Without it a tail with no observations
/// estimates zero nulls and yields zero q-values. Lambda zero uses the BH
/// baseline. Each row is a sensitivity analysis at a fixed threshold.
/// @param p Nonempty vector of finite p-values in [0,1].
/// @param thresholds Nonempty tuning thresholds in [0,1), default {.5}.
/// @param add_one Add one to the tail count when true, default true.
/// @return One null estimate and q-value vector per threshold, in supplied order.
/// @throws std::invalid_argument Empty input or invalid probability.
/// @par Example
/// @code
/// const auto fit = dm::stats::storey_qvalues({.001, .03, .06, .8}, {.5}, false);
/// std::print("{:.3f}\n", fit.front().q_values[0]); // 0.002
/// @endcode
/// @par References
/// - J. D. Storey (2002), https://doi.org/10.1111/1467-9868.00346
/// @see SimesDiscoveryBounds, p_adjust()
std::vector<StoreyQValueFit> storey_qvalues(const std::vector<double>& p,
                                           const std::vector<double>& thresholds = {.5},
                                           bool add_one = true);

/// @brief Simultaneous lower confidence bounds on true discoveries from closed Simes tests.
///
/// Construct the family once, then query any subset of hypotheses at the
/// same confidence level, including subsets chosen after inspecting the data.
/// Coverage requires the Simes inequality for the true-null intersections,
/// as under independent or suitably positively dependent tests.
/// @par Example
/// @code
/// const dm::stats::SimesDiscoveryBounds fit({.001, .01, .7, .8});
/// std::print("{}\n", fit.lower_bound({0, 1, 2})); // 2
/// @endcode
/// @par References
/// - J. J. Goeman, R. J. Meijer, T. J. P. Krebs and A. Solari (2019), https://doi.org/10.1093/biomet/asz041
/// @see storey_qvalues()
class SimesDiscoveryBounds {
public:
  /// @brief Builds the fixed closed-testing family and finds its largest locally nonrejected intersection size.
  /// @param p Finite p-values in [0,1], possibly empty.
  /// @param alpha Simultaneous miscoverage probability in (0,1), default .05.
  /// @throws std::invalid_argument Invalid p-value or alpha.
  /// @par Example
  /// @code
  /// const dm::stats::SimesDiscoveryBounds fit({.5, .7});
  /// std::print("{}\n", fit.lower_bound({0})); // 0
  /// @endcode
  /// @see lower_bound()
  explicit SimesDiscoveryBounds(std::vector<double> p, double alpha = .05);

  /// @brief Computes a simultaneous lower bound on the number of false nulls in the selected subset.
  /// @param indices Distinct zero-based indices into the original family, in any order.
  /// @return Bound from zero through subset size. An empty subset gives zero.
  /// @throws std::invalid_argument Duplicate or out-of-range index.
  /// @par Example
  /// @code
  /// const dm::stats::SimesDiscoveryBounds fit({.001, .7, .8});
  /// std::print("{}\n", fit.lower_bound({2, 0})); // 1
  /// @endcode
  /// @see SimesDiscoveryBounds()
  std::size_t lower_bound(const std::vector<std::size_t>& indices) const;

private:
  std::vector<double> p_;          ///< Original p-values in hypothesis order.
  double alpha_;                   ///< Fixed simultaneous error probability.
  std::size_t nonrejected_size_;  ///< Largest intersection that passes its local Simes test.
};

} // namespace datamunge::stats
