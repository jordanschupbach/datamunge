#pragma once

/// @file circular.hpp
/// @brief Descriptive statistics, intervals, uniformity tests, von Mises fitting and sampling for
/// directional (circular) data in radians.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief A trigonometric moment of order p: \f$\bar C_p+i\bar S_p=\frac1n\sum e^{ip\theta_j}\f$.
/// @par Example
/// @code
/// dm::stats::TrigonometricMoment m{.direction = 0.5, .resultant_length = 0.9};
/// std::println("{}", m.resultant_length); // 0.9
/// @endcode
struct TrigonometricMoment {
    double direction{};        ///< atan2(S_p, C_p) in (-pi, pi] (NaN when the resultant is below 1e-12).
    double resultant_length{}; ///< sqrt(C_p^2 + S_p^2) in [0, 1].
    double cosine{};           ///< C_p.
    double sine{};             ///< S_p.
};

/// @brief Uncentred trigonometric moment of order p, as `circular::trigonometric.moment`.
/// @param angles Angles in radians.
/// @param order Moment order p (at least 1).
/// @return The moment.
/// @throws std::invalid_argument For an empty sample, nonfinite angles or order 0.
/// @par Example
/// @code
/// const auto m = dm::stats::trigonometric_moment({0.0, 3.14159265358979}, 2);
/// std::println("{:.3f}", m.resultant_length); // 1.000
/// @endcode
TrigonometricMoment trigonometric_moment(const std::vector<double>& angles, std::size_t order = 1);

/// @brief Descriptive statistics of a circular sample.
/// @par Example
/// @code
/// dm::stats::CircularSummary s;
/// std::println("{}", s.n); // 0
/// @endcode
struct CircularSummary {
    std::size_t n{};             ///< Sample size.
    double mean_direction{};     ///< atan2 of the mean resultant, in (-pi, pi] (NaN if R < 1e-12).
    double resultant_length{};   ///< Mean resultant length R in [0, 1].
    double circular_variance{};  ///< 1 - R.
    double circular_sd{};        ///< sqrt(-2 log R) (infinite if R is 0).
    double angular_deviation{};  ///< sqrt(2 (1 - R)).
    double second_central_cosine{}; ///< \f$\bar\rho_2=\frac1n\sum\cos2(\theta_j-\bar\theta)\f$.
    double circular_dispersion{};   ///< Fisher's \f$\hat\delta=(1-\bar\rho_2)/(2R^2)\f$.
    double standard_error{};     ///< \f$\sqrt{\hat\delta/n}\f$, the circular standard error of the mean direction.
    double lower{};              ///< Lower end of the interval for the mean direction (may be below -pi).
    double upper{};              ///< Upper end (lower < upper; NaN when the interval is the whole circle).
};

/// @brief Mean direction, resultant length, spread measures and Fisher's interval for the mean direction.
///
/// The interval is \f$\bar\theta\pm\arcsin(z_{1-\alpha/2}\hat\sigma)\f$ with \f$\hat\sigma=\sqrt{\hat\delta/n}\f$
/// (Fisher 1993, section 4.4.4); it does not assume a von Mises distribution but needs n of about 25
/// or more. When \f$z\hat\sigma\ge1\f$ the data do not determine a direction and the bounds are NaN.
/// Values match `circular::mean.circular`, `rho.circular`, `var.circular`, `sd.circular` and
/// `angular.deviation`.
/// @param angles Angles in radians.
/// @param confidence Interval coverage in (0, 1).
/// @return The summary.
/// @throws std::invalid_argument For an empty sample, nonfinite angles or confidence outside (0, 1).
/// @par Example
/// @code
/// const auto s = dm::stats::circular_summary({0.1, -0.1, 0.2, -0.2, 0.0});
/// std::println("{:.3f} {:.3f}", s.mean_direction, s.resultant_length); // 0.000 0.990
/// @endcode
CircularSummary circular_summary(const std::vector<double>& angles, double confidence = 0.95);

/// @brief Percentile bootstrap interval for the mean direction (Fisher 1993, section 4.4.5).
/// @par Example
/// @code
/// dm::stats::CircularInterval ci{.lower = -0.2, .upper = 0.3};
/// std::println("{}", ci.upper - ci.lower); // 0.5
/// @endcode
struct CircularInterval {
    double lower{}; ///< Lower end (the mean direction minus the lower deviation; may be below -pi).
    double upper{}; ///< Upper end (lower < upper).
};

/// @brief Bootstrap interval for the mean direction from resampled mean directions.
///
/// Each resample's mean direction is expressed as a signed deviation from the sample mean direction in
/// (-pi, pi], and the interval is the sample mean plus the alpha/2 and 1 - alpha/2 quantiles of the
/// deviations. Recommended over circular_summary()'s large-sample interval for small samples.
/// @param angles Angles in radians.
/// @param confidence Interval coverage in (0, 1).
/// @param replicates Bootstrap resamples (at least 100).
/// @param seed Random seed.
/// @return The interval.
/// @throws std::invalid_argument For fewer than two angles, nonfinite angles, a zero resultant, or
/// invalid confidence or replicates.
/// @par Example
/// @code
/// const auto ci = dm::stats::circular_mean_bootstrap_interval({0.1, -0.1, 0.2, -0.2, 0.0, 0.05});
/// std::println("{}", ci.lower < 0 && ci.upper > 0); // true
/// @endcode
CircularInterval circular_mean_bootstrap_interval(const std::vector<double>& angles, double confidence = 0.95,
                                                  std::size_t replicates = 2000, std::uint64_t seed = 1);

/// @brief Draws from the von Mises distribution \f$f(\theta)\propto e^{\kappa\cos(\theta-\mu)}\f$ by the
/// Best-Fisher (1979) rejection algorithm.
/// @param n Number of draws.
/// @param mean_direction mu in radians.
/// @param concentration kappa (0 gives the uniform distribution).
/// @param seed Random seed.
/// @return Angles in (-pi, pi].
/// @throws std::invalid_argument For a negative or nonfinite concentration or a nonfinite mean.
/// @par Example
/// @code
/// const auto x = dm::stats::von_mises_sample(5000, 1.0, 4.0, 3);
/// std::println("{:.1f}", dm::stats::circular_summary(x).mean_direction); // 1.0
/// @endcode
std::vector<double> von_mises_sample(std::size_t n, double mean_direction, double concentration, std::uint64_t seed);

/// @brief Result of a circular uniformity test.
/// @par Example
/// @code
/// dm::stats::CircularTest t{.statistic = 0.4, .p_value = 0.01};
/// std::println("{}", t.p_value < 0.05); // true
/// @endcode
struct CircularTest {
    double statistic{};          ///< Test statistic as reported by the circular package (see each test).
    double p_value{};            ///< Analytic (corrected or asymptotic) p-value.
    double monte_carlo_p_value{}; ///< (1 + #simulated >= observed) / (1 + simulations); NaN without simulations.
};

/// @brief Options for the Monte Carlo calibration of circular tests.
/// @par Example
/// @code
/// const dm::stats::CircularTestOptions o{.simulations = 9999};
/// std::println("{}", o.seed); // 1
/// @endcode
struct CircularTestOptions {
    std::size_t simulations{0}; ///< Uniform samples of the same size simulated under the null (0 skips).
    std::uint64_t seed{1};      ///< Seed of the simulations.
};

/// @brief Rayleigh test of uniformity against a unimodal alternative with unknown mean direction.
///
/// The statistic is the mean resultant length \f$\bar R\f$; with \f$Z=n\bar R^2\f$ the p-value is
/// \f$e^{-Z}[1+(2Z-Z^2)/(4n)-(24Z-132Z^2+76Z^3-9Z^4)/(288n^2)]\f$ for n < 50 and \f$e^{-Z}\f$ otherwise
/// (Greenwood and Durand 1955), as `circular::rayleigh.test`. Powerful against von Mises departures;
/// powerless against symmetric multimodal ones such as axial data.
/// @param angles Angles in radians.
/// @param options Optional Monte Carlo calibration.
/// @return R-bar and the p-values.
/// @throws std::invalid_argument For an empty sample or nonfinite angles.
/// @par Example
/// @code
/// const auto t = dm::stats::rayleigh_test(dm::stats::von_mises_sample(40, 0.0, 2.0, 1));
/// std::println("{}", t.p_value < 0.001); // true
/// @endcode
CircularTest rayleigh_test(const std::vector<double>& angles, const CircularTestOptions& options = {});

/// @brief Rayleigh (V) test of uniformity against a unimodal alternative with a specified mean direction.
///
/// The statistic is \f$\bar R_0=\frac1n\sum\cos(\theta_j-\mu_0)\f$; with \f$z_0=\sqrt{2n}\bar R_0\f$ the
/// p-value is \f$1-\Phi(z_0)+\phi(z_0)[(3z_0-z_0^3)/(16n)+(15z_0+305z_0^3-125z_0^5+9z_0^7)/(4608n^2)]\f$,
/// as `circular::rayleigh.test(x, mu)`. More powerful than rayleigh_test() when the direction is known.
/// @param angles Angles in radians.
/// @param direction Hypothesised mean direction under the alternative.
/// @param options Optional Monte Carlo calibration.
/// @return R-bar_0 and the p-values.
/// @throws std::invalid_argument For an empty sample or nonfinite inputs.
/// @par Example
/// @code
/// const auto t = dm::stats::rayleigh_test_direction(dm::stats::von_mises_sample(40, 0.0, 1.0, 1), 0.0);
/// std::println("{}", t.statistic > 0); // true
/// @endcode
CircularTest rayleigh_test_direction(const std::vector<double>& angles, double direction,
                                     const CircularTestOptions& options = {});

/// @brief Kuiper's test that angles follow a specified continuous circular distribution.
///
/// With \f$u_{(i)}=F(\theta_{(i)})\f$, \f$V=\max_i(i/n-u_{(i)})+\max_i(u_{(i)}-(i-1)/n)\f$ is invariant to the
/// choice of origin. The reported statistic is Stephens' modified \f$V^*=V(\sqrt n+0.155+0.24/\sqrt n)\f$,
/// as `circular::kuiper.test`, and the p-value is its asymptotic tail
/// \f$2\sum_{k\ge1}(4k^2V^{*2}-1)e^{-2k^2V^{*2}}\f$ (capped to [0, 1]), accurate for n of about 8 or more.
/// Kuiper's test has power against all alternatives, including multimodal ones. Monte Carlo p-values
/// simulate uniform probability-integral transforms, so they are exact for any continuous null.
/// @param angles Angles in radians (reduced modulo 2 pi).
/// @param cdf Null CDF on [0, 2 pi) (default uniform).
/// @param options Optional Monte Carlo calibration.
/// @return V* and the p-values.
/// @throws std::invalid_argument For an empty sample, nonfinite angles, or CDF values outside [0, 1].
/// @par Example
/// @code
/// const auto t = dm::stats::kuiper_test(dm::stats::von_mises_sample(100, 0.0, 0.0, 2));
/// std::println("{}", t.p_value > 0.01); // true
/// @endcode
CircularTest kuiper_test(const std::vector<double>& angles, const std::function<double(double)>& cdf = {},
                         const CircularTestOptions& options = {});

/// @brief Mean resultant length of the von Mises distribution, \f$A_1(\kappa)=I_1(\kappa)/I_0(\kappa)\f$.
///
/// Computed by backward recurrence of Bessel-function ratios, so it is accurate and finite for any
/// concentration.
/// @param concentration kappa >= 0.
/// @return A1(kappa) in [0, 1).
/// @throws std::invalid_argument For a negative or NaN concentration.
/// @par Example
/// @code
/// std::println("{:.6f}", dm::stats::von_mises_a1(1.0)); // 0.446390
/// @endcode
double von_mises_a1(double concentration);

/// @brief Exact inverse of von_mises_a1(): the kappa whose mean resultant length is r.
/// @param resultant_length r in [0, 1).
/// @return kappa (0 for r = 0, infinity for r = 1).
/// @throws std::invalid_argument For r outside [0, 1].
/// @par Example
/// @code
/// std::println("{:.6f}", dm::stats::von_mises_a1_inverse(0.446390)); // 1.000000
/// @endcode
double von_mises_a1_inverse(double resultant_length);

/// @brief von Mises log-likelihood \f$\sum_j\kappa\cos(\theta_j-\mu)-n\log(2\pi I_0(\kappa))\f$.
/// @param angles Angles in radians.
/// @param mean_direction mu.
/// @param concentration kappa >= 0.
/// @return The log-likelihood.
/// @throws std::invalid_argument For an empty sample, nonfinite inputs or a negative concentration.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::von_mises_log_likelihood({0.0}, 0.0, 0.0)); // -1.8379
/// @endcode
double von_mises_log_likelihood(const std::vector<double>& angles, double mean_direction, double concentration);

/// @brief von Mises CDF on [0, 2 pi): \f$F(\theta)=\int_0^\theta f(t)\,dt\f$ from the Fourier series of the
/// density, \f$\theta/2\pi+\frac1\pi\sum_{p\ge1}\frac{A_p(\kappa)}{p}[\sin p(\theta-\mu)+\sin p\mu]\f$ with
/// \f$A_p=I_p/I_0\f$.
///
/// Suitable as the null CDF of kuiper_test(). The series needs about kappa + 40 sqrt(kappa) terms, so
/// very large concentrations are slow (but accurate).
/// @param angle Angle in radians (reduced to [0, 2 pi)).
/// @param mean_direction mu.
/// @param concentration kappa >= 0.
/// @return F in [0, 1].
/// @throws std::invalid_argument For nonfinite inputs or a negative concentration.
/// @par Example
/// @code
/// std::println("{:.3f}", dm::stats::von_mises_cdf(3.14159265358979, 3.14159265358979, 5.0)); // 0.500
/// @endcode
double von_mises_cdf(double angle, double mean_direction, double concentration);

/// @brief Options for von_mises_fit().
/// @par Example
/// @code
/// const dm::stats::VonMisesFitOptions o{.bias_correction = true};
/// std::println("{}", o.exact_inverse); // true
/// @endcode
struct VonMisesFitOptions {
    double mean_direction{std::numeric_limits<double>::quiet_NaN()}; ///< Known mu (NaN estimates it).
    bool exact_inverse{true};    ///< Solve A1(kappa) = R exactly; false uses Fisher's approximation as `circular`.
    bool bias_correction{false}; ///< Best-Fisher small-sample correction of kappa, as `mle.vonmises(bias = TRUE)`.
    double confidence{0.95};     ///< Coverage of the intervals.
};

/// @brief Result of von_mises_fit().
/// @par Example
/// @code
/// dm::stats::VonMisesFit f;
/// std::println("{}", f.boundary); // false
/// @endcode
struct VonMisesFit {
    std::size_t n{};                ///< Sample size.
    double mean_direction{};        ///< mu-hat in (-pi, pi] (or the known value); NaN when undefined.
    double concentration{};         ///< kappa-hat (after any bias correction); infinite if all angles coincide.
    double mean_se{};               ///< \f$1/\sqrt{n\hat\kappa A_1(\hat\kappa)}\f$ (0 for a known mean; infinite at kappa 0).
    double concentration_se{};      ///< \f$1/\sqrt{n(1-A_1/\hat\kappa-A_1^2)}\f$.
    double resultant_length{};      ///< \f$\bar R_0=\frac1n\sum\cos(\theta_j-\hat\mu)\f$ (R-bar when mu is estimated).
    double log_likelihood{};        ///< At the estimates.
    double aic{};                   ///< -2 log L + 2 (number of estimated parameters).
    double mean_lower{};            ///< Wald interval for mu, mu-hat -/+ z se (NaN when undefined or wider than the circle).
    double mean_upper{};            ///< Upper end of the mean interval.
    double concentration_lower{};   ///< Profile-likelihood interval for kappa (mu fixed at mu-hat); 0 at the boundary.
    double concentration_upper{};   ///< Upper end of the profile interval.
    double rayleigh_p_value{};      ///< Rayleigh test of uniformity (whether a mean direction exists).
    bool boundary{};                ///< kappa-hat is 0: the fitted distribution is uniform and mu is unidentified.
    bool vanishing_resultant{};     ///< The Rayleigh p-value exceeds 1 - confidence: treat mu-hat as unreliable.
};

/// @brief Maximum-likelihood fit of the von Mises distribution with diagnostics for a vanishing resultant.
///
/// The estimates are \f$\hat\mu=\operatorname{atan2}(\bar S,\bar C)\f$ and \f$\hat\kappa=A_1^{-1}(\bar R_0)\f$
/// (0 when \f$\bar R_0\le0\f$), matching `circular::mle.vonmises`, whose standard errors are reproduced.
/// \f$\hat\kappa\f$ is biased upward in small samples; the Best-Fisher correction reduces the bias. When the
/// resultant is small the likelihood for mu is flat: the fit flags this through `boundary`,
/// `vanishing_resultant` and an undefined mean interval rather than reporting a spurious direction.
/// @param angles Angles in radians.
/// @param options Known mean, inverse method, bias correction and confidence.
/// @return The fit.
/// @throws std::invalid_argument For fewer than two angles, nonfinite angles or confidence outside (0, 1).
/// @par Example
/// @code
/// const auto f = dm::stats::von_mises_fit(dm::stats::von_mises_sample(2000, 1.0, 3.0, 5));
/// std::println("{:.1f} {:.1f}", f.mean_direction, f.concentration); // 1.0 3.0
/// @endcode
VonMisesFit von_mises_fit(const std::vector<double>& angles, const VonMisesFitOptions& options = {});

} // namespace datamunge::stats
