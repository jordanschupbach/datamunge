#pragma once

/// @file gev.hpp
/// @brief Generalized extreme value (GEV) distribution for block maxima: distribution functions, maximum-likelihood
/// and L-moment fitting, the Gumbel (xi = 0) limit and its likelihood-ratio test, and return levels with
/// delta-method and profile-likelihood intervals.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief GEV distribution function \f$\exp\{-[1+\xi(x-\mu)/\sigma]_+^{-1/\xi}\}\f$ (Gumbel when xi = 0).
/// @param x Point.
/// @param location \f$\mu\f$.
/// @param scale \f$\sigma>0\f$.
/// @param shape \f$\xi\f$.
/// @return Probability.
/// @throws std::invalid_argument If scale <= 0.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::gev_cdf(0, 0, 1, 0)); // 0.3679
/// @endcode
double gev_cdf(double x, double location, double scale, double shape);

/// @brief GEV quantile \f$\mu+\sigma[(-\log p)^{-\xi}-1]/\xi\f$.
/// @param p Probability in (0, 1).
/// @param location \f$\mu\f$.
/// @param scale \f$\sigma>0\f$.
/// @param shape \f$\xi\f$.
/// @return Quantile.
/// @throws std::invalid_argument If p is outside (0, 1) or scale <= 0.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::gev_quantile(0.99, 0, 1, 0)); // 4.6001
/// @endcode
double gev_quantile(double p, double location, double scale, double shape);

/// @brief GEV log density (\f$-\infty\f$ outside the support).
/// @param x Point.
/// @param location \f$\mu\f$.
/// @param scale \f$\sigma>0\f$.
/// @param shape \f$\xi\f$.
/// @return Log density.
/// @throws std::invalid_argument If scale <= 0.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::gev_log_density(0, 0, 1, 0)); // -1.0000
/// @endcode
double gev_log_density(double x, double location, double scale, double shape);

/// @brief GEV estimator.
/// @par Example
/// @code
/// const auto m = dm::stats::GevMethod::l_moments;
/// std::println("{}", m == dm::stats::GevMethod::l_moments); // true
/// @endcode
enum class GevMethod {
    maximum_likelihood, ///< Maximum likelihood (Newton-Raphson from the L-moment start).
    l_moments,          ///< Hosking, Wallis and Wood (1985) L-moment (probability-weighted moment) estimator.
};

/// @brief Options for fit_gev().
/// @par Example
/// @code
/// dm::stats::GevOptions o{.gumbel = true};
/// std::println("{}", o.gumbel); // true
/// @endcode
struct GevOptions {
    GevMethod method{GevMethod::maximum_likelihood}; ///< Estimator.
    bool gumbel{false};                              ///< Fix the shape at 0 (Gumbel model).
    std::size_t max_iterations{200};                 ///< Newton iterations.
    double tolerance{1e-10};                         ///< Convergence tolerance on the Newton step.
};

/// @brief A fitted GEV (or Gumbel) model.
/// @par Example
/// @code
/// dm::stats::GevFit f{.location = 3};
/// std::println("{}", f.location); // 3
/// @endcode
struct GevFit {
    double location{};                     ///< \f$\hat\mu\f$.
    double scale{};                        ///< \f$\hat\sigma\f$.
    double shape{};                        ///< \f$\hat\xi\f$ (0 for Gumbel).
    double location_se{};                  ///< Standard error (ML; NaN for L-moments).
    double scale_se{};                     ///< Standard error.
    double shape_se{};                     ///< Standard error (NaN for Gumbel).
    linalg::DenseMatrix<double> covariance; ///< Inverse observed information of (mu, sigma[, xi]) (ML only).
    double log_likelihood{};               ///< Log-likelihood at the estimate.
    double aic{};                          ///< \f$-2\ell+2k\f$.
    std::size_t n{};                       ///< Number of maxima.
    bool gumbel{};                         ///< Whether the shape was fixed at 0.
    std::size_t iterations{};              ///< Newton iterations.
    bool converged{};                      ///< Whether the Newton step fell below tolerance.
};

/// @brief Fit a GEV or Gumbel distribution to block maxima.
/// @param maxima Block maxima (n >= 3, finite, not all equal).
/// @param options Options.
/// @return The fit.
/// @throws std::invalid_argument On fewer than 3 values, nonfinite values or zero spread.
/// @par Example
/// @code
/// const std::vector<double> x{3.1, 2.4, 4.0, 2.9, 3.3, 5.2, 2.7, 3.6, 3.0, 4.4};
/// std::println("{:.2f}", dm::stats::fit_gev(x, {.gumbel = true}).location); // 3.09
/// @endcode
GevFit fit_gev(const std::vector<double>& maxima, const GevOptions& options = {});

/// @brief Likelihood-ratio test of the Gumbel model (xi = 0) within the GEV.
/// @par Example
/// @code
/// dm::stats::GumbelTest t{.statistic = 1};
/// std::println("{}", t.statistic); // 1
/// @endcode
struct GumbelTest {
    GevFit gev;          ///< Unrestricted fit.
    GevFit gumbel;       ///< Gumbel fit.
    double statistic{};  ///< \f$2(\ell_{GEV}-\ell_{Gumbel})\f$.
    double p_value{};    ///< \f$\chi^2_1\f$ upper tail.
};

/// @brief Likelihood-ratio test of xi = 0.
/// @param maxima Block maxima.
/// @return Both fits and the test.
/// @throws std::invalid_argument As for fit_gev().
/// @par Example
/// @code
/// const std::vector<double> x{3.1, 2.4, 4.0, 2.9, 3.3, 5.2, 2.7, 3.6, 3.0, 4.4};
/// std::println("{}", dm::stats::gumbel_test(x).p_value > 0.05); // true
/// @endcode
GumbelTest gumbel_test(const std::vector<double>& maxima);

/// @brief A return level with intervals.
/// @par Example
/// @code
/// dm::stats::ReturnLevel r{.period = 100};
/// std::println("{}", r.period); // 100
/// @endcode
struct ReturnLevel {
    double period{};        ///< Return period T in blocks: the level exceeded once per T blocks on average.
    double level{};         ///< Estimated \f$z_T\f$, the \f$1-1/T\f$ quantile.
    double standard_error{}; ///< Delta-method standard error.
    double wald_lower{};    ///< Wald (delta-method) interval.
    double wald_upper{};    ///< Upper Wald limit.
    double profile_lower{}; ///< Profile-likelihood interval (NaN if not found).
    double profile_upper{}; ///< Upper profile limit (NaN if unbounded within the search range).
};

/// @brief Return level for a period of T blocks from a maximum-likelihood fit.
/// @param fit A maximum-likelihood GevFit of `maxima`.
/// @param maxima The data used for the fit (for the profile likelihood).
/// @param period T > 1.
/// @param confidence Interval level.
/// @return Level with delta-method and profile-likelihood intervals.
/// @throws std::invalid_argument If period <= 1, confidence is outside (0, 1) or the fit has no covariance.
/// @par Example
/// @code
/// const std::vector<double> x{3.1, 2.4, 4.0, 2.9, 3.3, 5.2, 2.7, 3.6, 3.0, 4.4};
/// const auto f = dm::stats::fit_gev(x, {.gumbel = true});
/// std::println("{:.2f}", dm::stats::gev_return_level(f, x, 100).level); // 5.91
/// @endcode
ReturnLevel gev_return_level(const GevFit& fit, const std::vector<double>& maxima, double period,
                             double confidence = 0.95);

} // namespace datamunge::stats
