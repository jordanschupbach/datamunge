#pragma once

/// @file stl.hpp
/// @brief Seasonal-trend decomposition by loess (STL) with optional robustness iterations.

#include <cstddef>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Options for stl_decomposition(); unset values take the defaults of R's `stats::stl`.
/// @par Example
/// @code
/// const dm::stats::StlOptions options{.period = 12, .seasonal_window = 7};
/// std::println("{} {}", options.period, options.robust); // 12 false
/// @endcode
struct StlOptions {
    std::size_t period{};  ///< Observations per seasonal cycle (at least 2).

    /// Loess span (odd, at least 7 recommended) for the cycle-subseries smoother; unset means
    /// "periodic": a constant seasonal pattern equal to the cycle-subseries means.
    std::optional<std::size_t> seasonal_window;
    int seasonal_degree{0}; ///< Degree (0 or 1) of the seasonal loess; forced to 0 when periodic.

    /// Trend loess span; default the next odd integer above \f$1.5\,p/(1-1.5/n_s)\f$.
    std::optional<std::size_t> trend_window;
    int trend_degree{1}; ///< Degree (0 or 1) of the trend loess.

    /// Span of the low-pass loess applied after the moving averages; default the next odd above the period.
    std::optional<std::size_t> lowpass_window;
    std::optional<int> lowpass_degree; ///< Degree of the low-pass loess; default `trend_degree`.

    /// Evaluate the seasonal loess at every k-th point and interpolate; default \f$\lceil n_s/10\rceil\f$.
    std::optional<std::size_t> seasonal_jump;
    std::optional<std::size_t> trend_jump;   ///< Trend loess stride; default \f$\lceil n_t/10\rceil\f$.
    std::optional<std::size_t> lowpass_jump; ///< Low-pass loess stride; default \f$\lceil n_l/10\rceil\f$.
    bool robust{false}; ///< Downweight outliers with bisquare weights from outer iterations.
    std::optional<std::size_t> inner; ///< Inner loop passes; default 1 if robust, else 2.
    std::optional<std::size_t> outer; ///< Robustness iterations; default 15 if robust, else 0.
};

/// @brief Result of stl_decomposition(): \f$y_t=S_t+T_t+R_t\f$.
/// @par Example
/// @code
/// const dm::stats::StlDecomposition d;
/// std::println("{}", d.seasonal.size()); // 0
/// @endcode
struct StlDecomposition {
    std::vector<double> seasonal;  ///< Seasonal component \f$S_t\f$.
    std::vector<double> trend;     ///< Trend component \f$T_t\f$.
    std::vector<double> remainder; ///< \f$R_t=y_t-S_t-T_t\f$.

    /// Final robustness weights in [0, 1] (all 1 without outer iterations).
    std::vector<double> weights;
    std::size_t seasonal_window{}; ///< Seasonal span used (\f$10n+1\f$ when periodic).
    std::size_t trend_window{};    ///< Trend span used.
    std::size_t lowpass_window{};  ///< Low-pass span used.
};

/// @brief Seasonal-trend decomposition by loess (Cleveland et al. 1990).
///
/// The inner loop detrends the series, smooths each cycle-subseries (all January values, all
/// February values, ...) by loess, removes any trend leaking into that seasonal estimate with a
/// low-pass filter (moving averages of length p, p and 3 followed by loess), and smooths the
/// deseasonalized series to update the trend. Robust fitting repeats the inner loop with bisquare
/// weights \f$(1-(|R_t|/6\,\mathrm{median}|R|)^2)^2\f$. A periodic fit replaces the seasonal component
/// by its cycle-position means. The series is assumed to start at the first position of a cycle;
/// nonrobust results match R's `stats::stl` (a port of the Fortran reference implementation) and
/// robust results match `statsmodels.tsa.seasonal.STL`, which computes the median exactly.
/// @param y Series, finite, more than two full periods long.
/// @param options Period, spans, degrees, strides and robustness.
/// @return Seasonal, trend, remainder, robustness weights and the spans used.
/// @throws std::invalid_argument For nonfinite values, period below 2, a series of at most two periods,
/// degrees other than 0 or 1, or zero strides or iteration counts where positive ones are needed.
/// @par Example
/// @code
/// std::vector<double> y;
/// for (int t = 0; t < 48; ++t)
///     y.push_back(.1 * t + (t % 4 == 0 ? 2.0 : 0.0));
/// const auto d = dm::stats::stl_decomposition(y, {.period = 4});
/// std::println("{:.3f} {:.3f}", d.seasonal[0], d.seasonal[1]); // 1.485 -0.511
/// @endcode
/// @par References
/// - R. B. Cleveland, W. S. Cleveland, J. E. McRae and I. Terpenning (1990). "STL: A
///   seasonal-trend decomposition procedure based on loess". Journal of Official Statistics 6(1), 3-73.
/// @see fit_arima_exact()
[[nodiscard]] StlDecomposition stl_decomposition(const std::vector<double>& y, const StlOptions& options);

} // namespace datamunge::stats
