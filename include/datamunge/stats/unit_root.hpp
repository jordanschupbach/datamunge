#pragma once

/// @file unit_root.hpp
/// @brief Augmented Dickey–Fuller and Phillips–Perron unit-root tests with MacKinnon critical values and p-values,
/// and the KPSS stationarity test.

#include <cstddef>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Deterministic terms in a unit-root regression.
/// @par Example
/// @code
/// const auto t = dm::stats::UnitRootTrend::linear;
/// std::println("{}", t == dm::stats::UnitRootTrend::constant); // false
/// @endcode
enum class UnitRootTrend {
    none,     ///< No deterministic terms ("n"): a random walk against a zero-mean AR process.
    constant, ///< Intercept ("c"): against a stationary process with nonzero mean.
    linear,   ///< Intercept and linear trend ("ct"): against trend stationarity.
};

/// @brief Lag-length rule for augmented_dickey_fuller().
/// @par Example
/// @code
/// const auto s = dm::stats::LagSelection::bic;
/// std::println("{}", s == dm::stats::LagSelection::fixed); // false
/// @endcode
enum class LagSelection {
    fixed, ///< Use exactly the given number of lagged differences.
    aic,   ///< Minimize AIC over 0..max lags on a common sample, then refit.
    bic,   ///< Minimize BIC over 0..max lags on a common sample, then refit.
};

/// @brief MacKinnon (2010) finite-sample critical values for the \f$\tau\f$ statistic.
/// @par Example
/// @code
/// const dm::stats::UnitRootCriticalValues c;
/// std::println("{}", c.five_percent); // 0
/// @endcode
struct UnitRootCriticalValues {
    double one_percent{};  ///< Reject a unit root at 1% when the statistic is below this.
    double five_percent{}; ///< 5% critical value.
    double ten_percent{};  ///< 10% critical value.
};

/// @brief Result of a unit-root test; the null hypothesis is a unit root.
/// @par Example
/// @code
/// const dm::stats::UnitRootTest r;
/// std::println("{}", r.lags); // 0
/// @endcode
struct UnitRootTest {
    double statistic{};      ///< ADF \f$t\f$ statistic, or the Phillips–Perron \f$Z_\tau\f$.
    double p_value{};        ///< MacKinnon (1994) approximate left-tail p-value of `statistic`.

    /// Phillips–Perron \f$Z_\alpha=T(\hat\rho-1)-\tfrac12(T^2\hat\sigma_\rho^2/s^2)(\lambda^2-\gamma_0)\f$;
    /// NaN for the ADF test.
    double z_alpha{};
    double coefficient{};    ///< Estimated \f$\rho-1\f$ on the lagged level.
    std::size_t lags{};      ///< Lagged differences (ADF) or Newey–West lags (Phillips–Perron).
    std::size_t observations{}; ///< Observations in the test regression.
    UnitRootCriticalValues critical; ///< Critical values for `observations`.
};

/// @brief Options for augmented_dickey_fuller().
/// @par Example
/// @code
/// const dm::stats::AdfOptions options{.selection = dm::stats::LagSelection::fixed, .lags = 4};
/// std::println("{}", *options.lags); // 4
/// @endcode
struct AdfOptions {
    UnitRootTrend trend{UnitRootTrend::constant}; ///< Deterministic terms.
    LagSelection selection{LagSelection::aic};     ///< Lag rule.

    /// Fixed lag count, or the maximum for selection; default \f$\lceil12(n/100)^{1/4}\rceil\f$,
    /// capped at \f$\lfloor n/2\rfloor-k-1\f$ for k deterministic terms, as statsmodels.
    std::optional<std::size_t> lags;
};

/// @brief Augmented Dickey–Fuller test of a unit root.
///
/// Fits \f$\Delta y_t=\text{deterministic}+\gamma y_{t-1}+\sum_{i=1}^p\delta_i\Delta y_{t-i}+e_t\f$
/// by least squares and returns \f$t=\hat\gamma/\mathrm{se}(\hat\gamma)\f$. With selection, every
/// \f$p\le p_{\max}\f$ is fitted on the same \f$n-p_{\max}-1\f$ observations, the information
/// criterion picks p, and the chosen model is refitted on all \f$n-p-1\f$ usable observations.
/// Matches `statsmodels.tsa.stattools.adfuller`.
/// @param y Series, finite, in time order.
/// @param options Deterministic terms and lag rule.
/// @return Statistic, p-value, lags, observations and critical values.
/// @throws std::invalid_argument For nonfinite values, too short a series for the lags, or a constant series.
/// @par Example
/// @code
/// std::mt19937 g(1); // raw mt19937 output is the same on every standard library
/// std::vector<double> y{0};
/// for (int t = 1; t < 100; ++t)
///     y.push_back(.5 * y.back() + g() / 4294967296.0 - .5);
/// const auto r = dm::stats::augmented_dickey_fuller(y, {.selection = dm::stats::LagSelection::fixed, .lags = 1});
/// std::println("{:.3f} {}", r.statistic, r.p_value < .01); // -5.320 true
/// @endcode
/// @par References
/// - D. A. Dickey and W. A. Fuller (1979). "Distribution of the estimators for autoregressive
///   time series with a unit root". JASA 74(366), 427-431. doi:10.1080/01621459.1979.10482531
/// - S. E. Said and D. A. Dickey (1984). "Testing for unit roots in autoregressive-moving average
///   models of unknown order". Biometrika 71(3), 599-607. doi:10.1093/biomet/71.3.599
/// - J. G. MacKinnon (1994). "Approximate asymptotic distribution functions for unit-root and
///   cointegration tests". JBES 12(2), 167-176. doi:10.1080/07350015.1994.10510005
/// - J. G. MacKinnon (2010). "Critical values for cointegration tests". Queen's Economics
///   Department Working Paper 1227.
/// @see phillips_perron(), unit_root_p_value()
[[nodiscard]] UnitRootTest augmented_dickey_fuller(const std::vector<double>& y, const AdfOptions& options = {});

/// @brief Options for phillips_perron().
/// @par Example
/// @code
/// const dm::stats::PhillipsPerronOptions options{.lags = 4};
/// std::println("{}", *options.lags); // 4
/// @endcode
struct PhillipsPerronOptions {
    UnitRootTrend trend{UnitRootTrend::constant}; ///< Deterministic terms.

    /// Newey–West (Bartlett) lags for the long-run variance; default
    /// \f$\lfloor4(n/100)^{1/4}\rfloor\f$, the "short" rule of R's `tseries::pp.test`.
    std::optional<std::size_t> lags;
};

/// @brief Phillips–Perron test of a unit root.
///
/// Fits \f$y_t=\text{deterministic}+\rho y_{t-1}+u_t\f$ by least squares and corrects the
/// \f$t\f$ statistic for serial correlation nonparametrically:
/// \f$Z_\tau=\sqrt{\gamma_0/\lambda^2}\,t_\rho-\tfrac12(\lambda^2-\gamma_0)T\hat\sigma_\rho/(\lambda s)\f$,
/// where \f$\gamma_0=\sum u_t^2/T\f$, \f$\lambda^2\f$ is the Bartlett long-run variance of the
/// residuals and \f$s^2=\sum u_t^2/(T-k)\f$. \f$Z_\tau\f$ has the Dickey–Fuller \f$\tau\f$
/// distribution, so it shares the ADF critical values and p-values. Matches
/// `arch.unitroot.PhillipsPerron`.
/// @param y Series, finite, in time order.
/// @param options Deterministic terms and Newey–West lags.
/// @return \f$Z_\tau\f$, its p-value, \f$Z_\alpha\f$, lags, observations and critical values.
/// @throws std::invalid_argument For nonfinite values, a series too short for the regression or lags,
/// or a constant series.
/// @par Example
/// @code
/// std::mt19937 g(1);
/// std::vector<double> y{0};
/// for (int t = 1; t < 100; ++t)
///     y.push_back(y.back() + g() / 4294967296.0 - .5); // a random walk
/// const auto r = dm::stats::phillips_perron(y);
/// std::println("{:.3f} {}", r.statistic, r.p_value > .05); // -1.949 true
/// @endcode
/// @par References
/// - P. C. B. Phillips and P. Perron (1988). "Testing for a unit root in time series regression".
///   Biometrika 75(2), 335-346. doi:10.1093/biomet/75.2.335
/// - W. K. Newey and K. D. West (1987). "A simple, positive semi-definite, heteroskedasticity and
///   autocorrelation consistent covariance matrix". Econometrica 55(3), 703-708. doi:10.2307/1913610
/// @see augmented_dickey_fuller()
[[nodiscard]] UnitRootTest phillips_perron(const std::vector<double>& y, const PhillipsPerronOptions& options = {});

/// @brief MacKinnon (1994) approximate p-value of a Dickey–Fuller \f$\tau\f$ statistic.
/// @param tau Test statistic.
/// @param trend Deterministic terms of the regression.
/// @return \f$\Phi(\text{polynomial in }\tau)\f$, clamped to 0 and 1 outside the fitted range.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::unit_root_p_value(-2.86, dm::stats::UnitRootTrend::constant)); // 0.0502
/// @endcode
/// @see unit_root_critical_values()
[[nodiscard]] double unit_root_p_value(double tau, UnitRootTrend trend);

/// @brief MacKinnon (2010) response-surface critical values \f$\beta_\infty+\beta_1/T+\beta_2/T^2+\beta_3/T^3\f$.
/// @param trend Deterministic terms.
/// @param observations Regression sample size T (positive).
/// @return 1%, 5% and 10% critical values.
/// @throws std::invalid_argument For zero observations.
/// @par Example
/// @code
/// const auto c = dm::stats::unit_root_critical_values(dm::stats::UnitRootTrend::constant, 146);
/// std::println("{:.4f}", c.five_percent); // -2.8815
/// @endcode
/// @see unit_root_p_value()
[[nodiscard]] UnitRootCriticalValues unit_root_critical_values(UnitRootTrend trend, std::size_t observations);

/// @brief Options for kpss_test().
/// @par Example
/// @code
/// const dm::stats::KpssOptions options{.trend = dm::stats::UnitRootTrend::linear};
/// std::println("{}", options.lags.has_value()); // false
/// @endcode
struct KpssOptions {
    UnitRootTrend trend{UnitRootTrend::constant}; ///< Level (constant) or trend (linear) stationarity; none is invalid.

    /// Bartlett lags for the long-run variance; default chosen from the data by Hobijn, Franses
    /// and Ooms (1998), as statsmodels `nlags = "auto"`.
    std::optional<std::size_t> lags;
};

/// @brief KPSS critical values from Kwiatkowski et al. (1992), Table 1.
/// @par Example
/// @code
/// const dm::stats::KpssCriticalValues c;
/// std::println("{}", c.five_percent); // 0
/// @endcode
struct KpssCriticalValues {
    double ten_percent{};          ///< Reject stationarity at 10% when the statistic exceeds this.
    double five_percent{};         ///< 5% critical value.
    double two_and_a_half_percent{}; ///< 2.5% critical value.
    double one_percent{};          ///< 1% critical value.
};

/// @brief Result of the KPSS test; the null hypothesis is (level or trend) stationarity.
/// @par Example
/// @code
/// const dm::stats::KpssTest r;
/// std::println("{}", r.p_value_truncated); // false
/// @endcode
struct KpssTest {
    double statistic{};        ///< \f$\eta=\sum_tS_t^2/(n^2\hat\lambda^2)\f$.

    /// p-value interpolated linearly in the critical value table, clamped to [0.01, 0.10].
    double p_value{};

    /// Whether the statistic lies outside the table, so the true p-value is beyond the clamp.
    bool p_value_truncated{};
    std::size_t lags{};        ///< Bartlett lags used.
    KpssCriticalValues critical; ///< Asymptotic critical values for the chosen trend.
};

/// @brief Kwiatkowski–Phillips–Schmidt–Shin test of stationarity against a unit root.
///
/// Regresses y on a constant (or a constant and a linear trend), forms the partial sums
/// \f$S_t=\sum_{s\le t}e_s\f$ of the residuals and compares \f$\sum S_t^2/n^2\f$ with the
/// Bartlett long-run variance \f$\hat\lambda^2=n^{-1}\{\sum e_t^2+2\sum_{j\le L}(1-\tfrac j{L+1})\sum_te_te_{t-j}\}\f$.
/// Large values reject stationarity. Matches `statsmodels.tsa.stattools.kpss`.
/// @param y Series, finite, at least four values.
/// @param options Trend and lags.
/// @return Statistic, p-value, truncation flag, lags and critical values.
/// @throws std::invalid_argument For nonfinite values, too short a series, trend none, or a constant series.
/// @par Example
/// @code
/// std::mt19937 g(1);
/// std::vector<double> y;
/// for (int t = 0; t < 100; ++t)
///     y.push_back(g() / 4294967296.0); // white noise
/// const auto r = dm::stats::kpss_test(y);
/// std::println("{:.3f} {:.2f}", r.statistic, r.p_value); // 0.111 0.10
/// @endcode
/// @par References
/// - D. Kwiatkowski, P. C. B. Phillips, P. Schmidt and Y. Shin (1992). "Testing the null hypothesis
///   of stationarity against the alternative of a unit root". Journal of Econometrics 54(1-3),
///   159-178. doi:10.1016/0304-4076(92)90104-Y
/// - B. Hobijn, P. H. Franses and M. Ooms (1998). "Generalisations of the KPSS-test for
///   stationarity". Econometric Institute Report 9802/A, Erasmus University Rotterdam.
/// @see augmented_dickey_fuller()
[[nodiscard]] KpssTest kpss_test(const std::vector<double>& y, const KpssOptions& options = {});

} // namespace datamunge::stats
