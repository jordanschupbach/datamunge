#pragma once

/// @file kernel_density.hpp
/// @brief Numerical Gaussian kernel density, reflection and bandwidth selection.

#include <optional>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Finite endpoint of the known half-line support for a reflected density.
/// @par Example
/// @code
/// const dm::stats::KdeBoundary boundary{0, true};
/// std::print("{}\n", boundary.lower); // true
/// @endcode
/// @see GaussianKernelDensity
struct KdeBoundary {
  double location;  ///< Endpoint in observation units.
  bool lower{true}; ///< True for support [location, infinity), false for an upper endpoint.
};

/// @brief Equal-weight Gaussian kernel density with optional one-sided reflection.
///
/// The model owns its sample and uses the given bandwidth as the standard
/// deviation of each Gaussian kernel. Read-only evaluation is thread-safe.
/// @par Example
/// @code
/// const dm::stats::GaussianKernelDensity model({0}, 1);
/// std::print("{:.3f}\n", model.density(0)); // 0.399
/// @endcode
/// @par References
/// - E. F. Schuster (1985), https://doi.org/10.1080/03610928508828965
/// @see kde_sheather_jones_bandwidth(), kde_lscv_bandwidth()
class GaussianKernelDensity {
public:
  /// @brief Builds an ordinary or boundary-reflected density from a nonempty sample.
  /// @param sample Finite observations, all on the given support if a boundary is supplied.
  /// @param bandwidth Positive finite kernel standard deviation, in sample units.
  /// @param boundary Optional known lower or upper endpoint, default none.
  /// @throws std::invalid_argument Empty or nonfinite sample, invalid bandwidth or support.
  /// @par Example
  /// @code
  /// const dm::stats::GaussianKernelDensity model({0}, 1, dm::stats::KdeBoundary{0});
  /// std::print("{:.3f}\n", model.density(0)); // 0.798
  /// @endcode
  /// @par References
  /// - E. F. Schuster (1985), https://doi.org/10.1080/03610928508828965
  /// @see density(), probability()
  GaussianKernelDensity(std::vector<double> sample, double bandwidth,
                        std::optional<KdeBoundary> boundary = std::nullopt);

  /// @brief Evaluates the density in O(n) time.
  /// @param value Evaluation point. Infinite points and points outside the support give zero.
  /// @return Nonnegative density, possibly infinite if it exceeds double range.
  /// @throws std::invalid_argument NaN coordinate.
  /// @par Example
  /// @code
  /// const dm::stats::GaussianKernelDensity model({1}, 1);
  /// std::print("{:.3f}\n", model.density(1)); // 0.399
  /// @endcode
  /// @see probability()
  double density(double value) const;

  /// @brief Integrates kernel probability over an interval using Gaussian tails in O(n) time.
  /// @param lower Lower endpoint, possibly infinite.
  /// @param upper Upper endpoint at least as large as lower, possibly infinite.
  /// @return Probability in [0,1]. Equal endpoints give zero and intervals are clipped to the support.
  /// @throws std::invalid_argument NaN or reversed endpoints.
  /// @par Example
  /// @code
  /// const dm::stats::GaussianKernelDensity model({0}, 1);
  /// std::print("{:.3f}\n", model.probability(-1, 1)); // 0.683
  /// @endcode
  /// @see density()
  double probability(double lower, double upper) const;

private:
  std::vector<double> sample_;          ///< Owned empirical kernel centers.
  double bandwidth_;                   ///< Kernel standard deviation.
  std::optional<KdeBoundary> boundary_; ///< Known support endpoint if reflection was requested.
};

/// @brief Estimates Gaussian bandwidth using Sheather–Jones direct plug-in.
///
/// Uses exact pair differences for the fourth- and sixth-derivative pilot
/// estimates. Pilot scale is min(SD, IQR/1.349) with SD used if IQR is zero.
/// Work is O(n squared). The method corresponds to R's bw.SJ method "dpi"
/// with unbinned rather than binned distances.
/// @param sample At least two finite, nonconstant observations.
/// @return Positive finite bandwidth in sample units.
/// @throws std::invalid_argument Too few values, nonfinite or constant observations, or invalid pilots.
/// @throws std::overflow_error Bandwidth outside the positive finite double range.
/// @par Example
/// @code
/// const auto h = dm::stats::kde_sheather_jones_bandwidth({0, 1, 2, 3, 4});
/// std::print("{}\n", h > 0); // true
/// @endcode
/// @par References
/// - S. J. Sheather and M. C. Jones (1991), https://doi.org/10.1111/j.2517-6161.1991.tb01857.x
/// @see kde_lscv_bandwidth(), GaussianKernelDensity
double kde_sheather_jones_bandwidth(const std::vector<double>& sample);

/// @brief Selected bandwidth and objective for Gaussian least-squares cross-validation.
/// @par Example
/// @code
/// const auto fit = dm::stats::kde_lscv_bandwidth({0, 1, 2, 3, 4});
/// std::print("{}\n", fit.bandwidth > 0); // true
/// @endcode
/// @see kde_lscv_bandwidth()
struct KdeLscvResult {
  double bandwidth;   ///< Selected kernel standard deviation, in sample units.
  double objective;   ///< Integrated squared density minus twice the leave-one-out mean, possibly infinite.
  bool at_boundary;   ///< Whether selection is at one of the search endpoints.
};

/// @brief Selects Gaussian bandwidth by least-squares leave-one-out cross-validation.
///
/// Uses the exact Gaussian convolution and optimizes over log bandwidth.
/// A grid locates candidate minima and bounded searches refine them.
/// Work per objective evaluation is O(n squared).
/// @param sample At least two finite observations. Constant input requires an explicit interval.
/// @param range Optional pair of positive finite ordered endpoints. The
/// default is [0.1 h_max, h_max], with h_max = 1.144 SD n^(-1/5).
/// @return Selected bandwidth, its objective and an endpoint diagnostic.
/// @throws std::invalid_argument Invalid sample or explicit interval.
/// @throws std::overflow_error Selected bandwidth outside the positive finite double range.
/// @par Example
/// @code
/// const auto fit = dm::stats::kde_lscv_bandwidth({0, 1, 2, 3, 4}, std::pair{.5, 2.0});
/// std::print("{}\n", fit.bandwidth >= .5 && fit.bandwidth <= 2); // true
/// @endcode
/// @par References
/// - A. W. Bowman (1984), https://doi.org/10.1093/biomet/71.2.353
/// @see kde_sheather_jones_bandwidth(), GaussianKernelDensity
KdeLscvResult kde_lscv_bandwidth(const std::vector<double>& sample,
                                std::optional<std::pair<double, double>> range = std::nullopt);

} // namespace datamunge::stats
