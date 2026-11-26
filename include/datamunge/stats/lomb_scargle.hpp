#pragma once

/// @file lomb_scargle.hpp
/// @brief Generalized (floating-mean) Lomb–Scargle periodogram for irregular sampling, with false-alarm calibration.

#include <cstddef>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Normalization of Lomb–Scargle power, as Astropy's `LombScargle`.
/// @par Example
/// @code
/// const auto n = dm::stats::LombScargleNormalization::standard;
/// std::println("{}", n == dm::stats::LombScargleNormalization::psd); // false
/// @endcode
enum class LombScargleNormalization {
    standard, ///< \f$P=1-\chi^2_{\text{sin}}/\chi^2_0\f$, the fraction of weighted variance explained, in [0, 1].
    model,    ///< \f$P=\chi^2_0/\chi^2_{\text{sin}}-1\f$.
    log,      ///< \f$P=\ln(\chi^2_0/\chi^2_{\text{sin}})\f$.
    psd,      ///< \f$P=(\chi^2_0-\chi^2_{\text{sin}})/2\f$ in units of \f$\sum_i\sigma_i^{-2}\f$-weighted power.
};

/// @brief Approximation for the false-alarm probability of the highest peak.
/// @par Example
/// @code
/// const auto m = dm::stats::FalseAlarmMethod::baluev;
/// std::println("{}", m == dm::stats::FalseAlarmMethod::naive); // false
/// @endcode
enum class FalseAlarmMethod {
    single, ///< Probability for one pre-chosen frequency (no search correction).
    naive,  ///< \f$1-(1-p_1)^{N_{\text{eff}}}\f$ with \f$N_{\text{eff}}=f_{\max}T\f$ independent frequencies.
    davies, ///< Davies upper bound \f$p_1+\tau\f$.
    baluev, ///< Baluev (2008) alias-free approximation \f$1-(1-p_1)e^{-\tau}\f$; accurate for small probabilities.
};

/// @brief Options for lomb_scargle() and the related functions.
/// @par Example
/// @code
/// const dm::stats::LombScargleOptions options{.samples_per_peak = 10};
/// std::println("{}", options.samples_per_peak); // 10
/// @endcode
struct LombScargleOptions {
    bool fit_mean{true};     ///< Fit an offset at each frequency (Zechmeister and Kürster's floating mean).
    bool center_data{true};  ///< Subtract the weighted mean first (always done when `fit_mean`).
    LombScargleNormalization normalization{LombScargleNormalization::standard}; ///< Power units.
    double samples_per_peak{5};  ///< Grid points per peak width \f$1/T\f$.
    double nyquist_factor{5};    ///< Upper frequency as a multiple of the average Nyquist \f$n/(2T)\f$.
    std::optional<double> minimum_frequency{}; ///< Lower grid limit (default half a grid step).
    std::optional<double> maximum_frequency{}; ///< Upper grid limit (default from `nyquist_factor`).
    FalseAlarmMethod method{FalseAlarmMethod::baluev}; ///< False-alarm approximation.
};

/// @brief Periodogram over the automatic grid with its best peak.
/// @par Example
/// @code
/// const dm::stats::LombScargleResult r;
/// std::println("{}", r.power.size()); // 0
/// @endcode
struct LombScargleResult {
    std::vector<double> frequencies; ///< Evenly spaced grid \f$f_{\min}+k\,\delta f\f$, \f$\delta f=1/(T\cdot\text{samples per peak})\f$.
    std::vector<double> power;       ///< Power at each frequency.
    double best_frequency{};         ///< Frequency of the highest power.
    double best_power{};             ///< Highest power.
    double false_alarm_probability{}; ///< Probability of a peak at least this high from noise, over the searched band.
    double offset{};                 ///< Fitted mean level at the best frequency.
    double amplitude{};              ///< Fitted semi-amplitude \f$\sqrt{a^2+b^2}\f$ of the sinusoid at the best frequency.
};

/// @brief Lomb–Scargle power at given frequencies, as Astropy's `LombScargle.power`.
///
/// For each frequency the weighted least-squares fit of
/// \f$y=c+a\sin2\pi ft+b\cos2\pi ft\f$ (without \f$c\f$ unless `fit_mean`) is compared with a
/// constant, using the time offset \f$\tau\f$ that decouples the sine and cosine terms. Times may be
/// irregular, and weights are \f$1/\sigma_i^2\f$.
/// @param t Observation times (finite, not all equal).
/// @param y Values (finite).
/// @param dy Measurement uncertainties (empty for unit weights; one per value otherwise, positive).
/// @param frequencies Positive frequencies in cycles per unit time.
/// @param options Mean handling and normalization (grid fields unused).
/// @return Power at each frequency.
/// @throws std::invalid_argument For mismatched sizes, fewer than 3 points, nonfinite inputs,
/// nonpositive uncertainties, constant times or nonpositive frequencies.
/// @par Example
/// @code
/// std::vector<double> t, y;
/// for (int i = 0; i < 40; ++i) {
///     t.push_back(i + .3 * std::sin(i * 1.7));
///     y.push_back(std::sin(2 * std::numbers::pi * .1 * t.back()));
/// }
/// const auto p = dm::stats::lomb_scargle_power(t, y, {}, {.05, .1, .2});
/// std::println("{:.3f} {:.3f}", p[1], p[2]); // 1.000 0.001
/// @endcode
/// @par References
/// - J. D. Scargle (1982). "Studies in astronomical time series analysis II." ApJ 263, 835-853.
///   doi:10.1086/160554
/// - M. Zechmeister and M. Kürster (2009). "The generalised Lomb-Scargle periodogram." A&A 496,
///   577-584. doi:10.1051/0004-6361:200811296
/// @see lomb_scargle()
[[nodiscard]] std::vector<double> lomb_scargle_power(const std::vector<double>& t, const std::vector<double>& y,
                                                     const std::vector<double>& dy, const std::vector<double>& frequencies,
                                                     const LombScargleOptions& options = {});

/// @brief Automatic frequency grid, as Astropy's `autofrequency`.
/// @param t Observation times (at least 2, not all equal).
/// @param options Grid density and limits.
/// @return Evenly spaced frequencies.
/// @throws std::invalid_argument For constant times, nonpositive grid settings or an empty band.
/// @par Example
/// @code
/// const auto f = dm::stats::lomb_scargle_frequencies({0, 1, 2, 3, 4, 10});
/// std::println("{} {:.2f}", f.size(), f.front()); // 76 0.01
/// @endcode
/// @see lomb_scargle()
[[nodiscard]] std::vector<double> lomb_scargle_frequencies(const std::vector<double>& t,
                                                           const LombScargleOptions& options = {});

/// @brief Periodogram over the automatic grid, its highest peak, the peak's false-alarm probability and sinusoid fit.
/// @param t Observation times.
/// @param y Values.
/// @param dy Uncertainties (empty for unit weights).
/// @param options Mean handling, normalization, grid and false-alarm method.
/// @return Grid, power, best frequency, false-alarm probability, offset and amplitude.
/// @throws std::invalid_argument As lomb_scargle_power() and lomb_scargle_false_alarm_probability().
/// @par Example
/// @code
/// std::mt19937 g(3); // raw mt19937 output is portable
/// std::vector<double> t, y;
/// for (int i = 0; i < 80; ++i) {
///     t.push_back(100 * (g() / 4294967296.0));
///     y.push_back(2 + std::cos(2 * std::numbers::pi * t.back() / 7) + (g() / 4294967296.0 - .5));
/// }
/// const auto r = dm::stats::lomb_scargle(t, y);
/// std::println("{:.0f} {:.2f} {}", 1 / r.best_frequency, r.amplitude, r.false_alarm_probability < 1e-10); // 7 1.00 true
/// @endcode
/// @see lomb_scargle_power(), lomb_scargle_false_alarm_probability(), lomb_scargle_model()
[[nodiscard]] LombScargleResult lomb_scargle(const std::vector<double>& t, const std::vector<double>& y,
                                             const std::vector<double>& dy = {}, const LombScargleOptions& options = {});

/// @brief Probability that noise alone gives a highest peak at least `power` over the searched band.
///
/// Uses the single-frequency distribution of the chosen normalization with \f$N-3\f$ degrees of
/// freedom, and for the search corrections the band's upper limit from the automatic grid and
/// the effective baseline \f$\sqrt{4\pi\operatorname{Var}_w(t)}\f$ (Baluev 2008), as Astropy's
/// `false_alarm_probability`. The PSD normalization assumes the uncertainties are correct.
/// @param power Observed peak power (in the options' normalization).
/// @param t Observation times.
/// @param y Values.
/// @param dy Uncertainties (empty for unit weights).
/// @param options Normalization, grid limits and method.
/// @return Probability in [0, 1] (the Davies bound may exceed 1).
/// @throws std::invalid_argument When neither `fit_mean` nor `center_data` is set, or for invalid data.
/// @par Example
/// @code
/// const std::vector<double> t{0, 1, 2.5, 3, 4.2, 5, 6.1, 7, 8, 9.4};
/// const std::vector<double> y{1, 0, -1, 0, 1, 0, -1, 0, 1, 0};
/// const dm::stats::LombScargleOptions single{.method = dm::stats::FalseAlarmMethod::single};
/// std::println("{:.4f}", dm::stats::lomb_scargle_false_alarm_probability(.5, t, y, {}, single)); // 0.0884
/// @endcode
/// @par References
/// - R. V. Baluev (2008). "Assessing the statistical significance of periodogram peaks." MNRAS
///   385, 1279-1285. doi:10.1111/j.1365-2966.2008.12689.x
/// - J. T. VanderPlas (2018). "Understanding the Lomb-Scargle periodogram." ApJS 236, 16.
///   doi:10.3847/1538-4365/aab766
/// @see lomb_scargle_false_alarm_level()
[[nodiscard]] double lomb_scargle_false_alarm_probability(double power, const std::vector<double>& t,
                                                          const std::vector<double>& y, const std::vector<double>& dy,
                                                          const LombScargleOptions& options = {});

/// @brief Peak power whose false-alarm probability equals `probability` (inverse of the above).
/// @param probability Target probability in (0, 1).
/// @param t Observation times.
/// @param y Values.
/// @param dy Uncertainties (empty for unit weights).
/// @param options Normalization, grid limits and method.
/// @return Power threshold.
/// @throws std::invalid_argument For a probability outside (0, 1) or as lomb_scargle_false_alarm_probability().
/// @par Example
/// @code
/// const std::vector<double> t{0, 1, 2.5, 3, 4.2, 5, 6.1, 7, 8, 9.4};
/// const std::vector<double> y{1, 0, -1, 0, 1, 0, -1, 0, 1, 0};
/// const dm::stats::LombScargleOptions single{.method = dm::stats::FalseAlarmMethod::single};
/// std::println("{:.4f}", dm::stats::lomb_scargle_false_alarm_level(.0884, t, y, {}, single)); // 0.5000
/// @endcode
/// @see lomb_scargle_false_alarm_probability()
[[nodiscard]] double lomb_scargle_false_alarm_level(double probability, const std::vector<double>& t,
                                                    const std::vector<double>& y, const std::vector<double>& dy,
                                                    const LombScargleOptions& options = {});

/// @brief Best-fitting sinusoid at one frequency evaluated at new times, as Astropy's `LombScargle.model`.
/// @param t Observation times.
/// @param y Values.
/// @param dy Uncertainties (empty for unit weights).
/// @param frequency Frequency of the sinusoid.
/// @param t_fit Times at which to evaluate the fit.
/// @param options Mean handling.
/// @return Fitted values at `t_fit`.
/// @throws std::invalid_argument As lomb_scargle_power().
/// @par Example
/// @code
/// const std::vector<double> t{0, 1, 2.5, 3, 4.2, 5, 6.1, 7};
/// std::vector<double> y;
/// for (const double s : t)
///     y.push_back(3 + 2 * std::cos(2 * std::numbers::pi * s / 4));
/// const auto m = dm::stats::lomb_scargle_model(t, y, {}, .25, {8});
/// std::println("{:.3f}", m[0]); // 5.000
/// @endcode
/// @see lomb_scargle()
[[nodiscard]] std::vector<double> lomb_scargle_model(const std::vector<double>& t, const std::vector<double>& y,
                                                     const std::vector<double>& dy, double frequency,
                                                     const std::vector<double>& t_fit, const LombScargleOptions& options = {});

} // namespace datamunge::stats
