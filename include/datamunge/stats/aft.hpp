#pragma once

/// @file aft.hpp
/// @brief Parametric accelerated failure time regression with exact, right-, left- and interval-censored times.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief An event time known exactly or only to lie in an interval.
///
/// `lower == upper` is an exact time, `upper` infinite is right censoring at `lower`, `lower == 0`
/// is left censoring at `upper`, and otherwise the event lies in (lower, upper].
/// @par Example
/// @code
/// const auto r = dm::stats::CensoredTime::right(4.0);
/// std::println("{}", std::isinf(r.upper)); // true
/// @endcode
struct CensoredTime {
    double lower{}; ///< Lower bound (0 for left censoring).
    double upper{}; ///< Upper bound (infinite for right censoring).

    /// @brief An exactly observed time.
    /// @param t Positive time.
    /// @return The record.
    /// @par Example
    /// @code
    /// std::println("{}", dm::stats::CensoredTime::exact(2).upper); // 2
    /// @endcode
    static CensoredTime exact(double t) { return {t,t}; }
    /// @brief An event known to occur after t.
    /// @param t Positive censoring time.
    /// @return The record.
    /// @par Example
    /// @code
    /// std::println("{}", dm::stats::CensoredTime::right(2).lower); // 2
    /// @endcode
    static CensoredTime right(double t) { return {t,std::numeric_limits<double>::infinity()}; }
    /// @brief An event known to occur by t.
    /// @param t Positive time.
    /// @return The record.
    /// @par Example
    /// @code
    /// std::println("{}", dm::stats::CensoredTime::left(2).lower); // 0
    /// @endcode
    static CensoredTime left(double t) { return {0,t}; }
    /// @brief An event in (lower, upper].
    /// @param lower Positive lower bound.
    /// @param upper Upper bound above lower.
    /// @return The record.
    /// @par Example
    /// @code
    /// std::println("{}", dm::stats::CensoredTime::interval(1, 3).upper); // 3
    /// @endcode
    static CensoredTime interval(double lower, double upper) { return {lower,upper}; }
};

/// @brief Error distribution of \f$\log T\f$, as `survreg(dist =)`.
/// @par Example
/// @code
/// const auto d = dm::stats::AftDistribution::lognormal;
/// std::println("{}", d == dm::stats::AftDistribution::weibull); // false
/// @endcode
enum class AftDistribution {
    weibull,     ///< Minimum extreme value errors; \f$T\f$ Weibull with shape \f$1/\sigma\f$.
    exponential, ///< Weibull with \f$\sigma=1\f$ fixed.
    lognormal,   ///< Normal errors.
    loglogistic, ///< Logistic errors.
};

/// @brief Options for fit_aft().
/// @par Example
/// @code
/// const dm::stats::AftOptions o{.distribution = dm::stats::AftDistribution::loglogistic};
/// std::println("{}", o.confidence); // 0.95
/// @endcode
struct AftOptions {
    AftDistribution distribution{AftDistribution::weibull}; ///< Error distribution.
    double confidence{0.95};                                ///< Coverage of time-ratio intervals.
    std::size_t max_iterations{100};                        ///< Newton iteration limit.
    double tolerance{1e-10};                                ///< Relative log-likelihood change that stops iteration.
};

/// @brief Fitted accelerated failure time model \f$\log T=\beta_0+x^\top\beta+\sigma W\f$.
/// @par Example
/// @code
/// const dm::stats::AftFit fit;
/// std::println("{}", fit.scale); // 1
/// @endcode
struct AftFit {
    AftDistribution distribution{};          ///< Error distribution.
    std::vector<double> coefficients;        ///< Intercept, then one per covariate column (log time scale).
    std::vector<double> standard_errors;     ///< Standard errors of the coefficients.
    double scale{1};                         ///< \f$\sigma\f$.
    double log_scale_standard_error{};       ///< Standard error of \f$\log\sigma\f$ (0 when fixed).
    linalg::DenseMatrix<double> covariance;  ///< Covariance of (coefficients, \f$\log\sigma\f$) or of the coefficients when \f$\sigma\f$ is fixed.
    std::vector<double> time_ratios;         ///< \f$e^{\beta_j}\f$ for each covariate (not the intercept).
    std::vector<double> time_ratio_lower;    ///< Lower interval ends from the Wald interval for \f$\beta_j\f$.
    std::vector<double> time_ratio_upper;    ///< Upper interval ends.
    double log_likelihood{};                 ///< Log likelihood on the time scale (with the Jacobian for exact times).
    double null_log_likelihood{};            ///< Log likelihood of the intercept-only model.
    double chi_square{};                     ///< Likelihood-ratio statistic against the intercept-only model.
    std::size_t degrees_of_freedom{};        ///< Number of covariates.
    double p_value{1};                       ///< Upper chi-square tail.
    std::size_t iterations{};                ///< Newton iterations.
    bool converged{false};                   ///< True if the tolerance was met.

    /// @brief Predicted p-quantile of T for a covariate row, \f$\exp(\eta+\sigma w_p)\f$.
    /// @param covariates One value per covariate column.
    /// @param p Probability in (0, 1).
    /// @return The quantile (p = 0.5 gives the median).
    /// @throws std::invalid_argument For a wrong length or p outside (0, 1).
    /// @par Example
    /// @code
    /// const auto fit = dm::stats::fit_aft({dm::stats::CensoredTime::exact(1), dm::stats::CensoredTime::exact(2),
    ///                                      dm::stats::CensoredTime::exact(4)}, dm::linalg::DenseMatrix<double>(3, 0));
    /// std::println("{:.2f}", fit.predict_quantile({}, .5)); // 2.21
    /// @endcode
    [[nodiscard]] double predict_quantile(const std::vector<double>& covariates, double p) const;

    /// @brief Predicted survival probability \f$P(T>t)\f$ for a covariate row.
    /// @param time Positive time.
    /// @param covariates One value per covariate column.
    /// @return Survival probability.
    /// @throws std::invalid_argument For a wrong length or nonpositive time.
    /// @par Example
    /// @code
    /// const auto fit = dm::stats::fit_aft({dm::stats::CensoredTime::exact(1), dm::stats::CensoredTime::exact(2),
    ///                                      dm::stats::CensoredTime::exact(4)}, dm::linalg::DenseMatrix<double>(3, 0));
    /// std::println("{:.3f}", fit.predict_survival(fit.predict_quantile({}, .5), {})); // 0.500
    /// @endcode
    [[nodiscard]] double predict_survival(double time, const std::vector<double>& covariates) const;
};

/// @brief Fits a parametric accelerated failure time model by maximum likelihood, as R's `survreg`.
///
/// Each covariate multiplies event times by \f$e^{\beta_j}\f$ (a time ratio). Exact times contribute
/// \f$\log f\f$, right-censored \f$\log S\f$, left-censored \f$\log F\f$ and interval-censored
/// \f$\log\{S(l)-S(u)\}\f$, all with analytic first and second derivatives. Newton's method on
/// \f$(\beta,\log\sigma)\f$ starts from least squares on log times and uses step halving, with a ridge
/// if the Hessian is not negative definite. For a Weibull fit, \f$1/\sigma\f$ is the shape and
/// \f$-\beta_j/\sigma\f$ the log hazard ratio.
/// @param times Positive exact or censored times.
/// @param covariates One row per time, no intercept column (zero columns fits the intercept only).
/// @param options Distribution, interval coverage and iteration control.
/// @return The fit, its likelihood-ratio test against the intercept-only model and time ratios.
/// @throws std::invalid_argument For invalid times or covariates, mismatched sizes, or no
/// observation that is exact or bounded on both sides when needed to identify the scale.
/// @throws std::runtime_error If Newton's method cannot improve the likelihood.
/// @par Example
/// @code
/// using C = dm::stats::CensoredTime;
/// const auto fit = dm::stats::fit_aft({C::exact(2), C::exact(3), C::right(5), C::exact(6), C::exact(8),
///                                      C::exact(9)}, dm::linalg::DenseMatrix<double>(6, 1, {0, 0, 0, 1, 1, 1}));
/// std::println("{:.2f}", fit.time_ratios[0]); // 1.83
/// @endcode
/// @par References
/// - J. D. Kalbfleisch and R. L. Prentice (2002). The Statistical Analysis of Failure Time Data,
///   2nd ed., chapter 2. Wiley. doi:10.1002/9781118032985
/// - L. J. Wei (1992). "The accelerated failure time model: a useful alternative to the Cox
///   regression model in survival analysis." Statistics in Medicine 11, 1871-1879.
///   doi:10.1002/sim.4780111409
/// @see fit_cox_counting()
[[nodiscard]] AftFit fit_aft(const std::vector<CensoredTime>& times, const linalg::DenseMatrix<double>& covariates,
                             const AftOptions& options = {});

} // namespace datamunge::stats
