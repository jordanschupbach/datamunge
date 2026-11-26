#pragma once

/// @file likelihood_tests.hpp
/// @brief Wald, score (Rao) and likelihood-ratio tests of parameter restrictions.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/maximum_likelihood.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief A chi-square test of parameter restrictions and the rank used as its degrees of freedom.
/// @par Example
/// @code
/// const dm::stats::RestrictionTestResult result;
/// std::println("{}", result.rank); // 0
/// @endcode
struct RestrictionTestResult {
    double statistic{};  ///< Chi-square statistic.
    std::size_t rank{};  ///< Number of linearly independent restrictions, used as degrees of freedom.
    double p_value{};    ///< Upper tail of the chi-square law with `rank` degrees of freedom.
    std::string method;  ///< Test name, including the covariance used for Wald tests.
};

/// @brief Wald test of linear restrictions \f$R\theta=r\f$ at a maximum likelihood fit.
///
/// The statistic is \f$(R\hat\theta-r)^\top(RVR^\top)^{+}(R\hat\theta-r)\f$, with V the
/// model-based or sandwich covariance of the fit. The rank is that of \f$RVR^\top\f$,
/// with eigenvalues below \f$10^{-10}\f$ times the largest treated as zero, so redundant
/// rows of R do not add degrees of freedom; the Moore–Penrose inverse is used. The test
/// needs only the unrestricted fit but is not invariant to reparameterization.
/// @param fit Unrestricted fit from maximum_likelihood().
/// @param restrictions \f$q\times p\f$ matrix R with \f$q\ge1\f$.
/// @param values Right-hand side r of length q.
/// @param robust Use the sandwich covariance instead of \f$J^{-1}\f$.
/// @return Statistic, rank, p-value and method.
/// @throws std::invalid_argument For mismatched shapes, nonfinite entries, a covariance containing NaN,
/// or restrictions of rank zero.
/// @par Example
/// @code
/// dm::stats::MaximumLikelihoodFit fit;
/// fit.estimate = {1.0, 0.5};
/// fit.covariance = dm::linalg::DenseMatrix<double>(2, 2, {.25, 0, 0, .25});
/// const auto w = dm::stats::wald_test(fit, dm::linalg::DenseMatrix<double>(1, 2, {1, 0}), {0});
/// std::println("{:.1f} {}", w.statistic, w.rank); // 4.0 1
/// @endcode
/// @par References
/// - A. Wald (1943). "Tests of statistical hypotheses concerning several parameters when the number
///   of observations is large". Trans. Amer. Math. Soc. 54(3), 426-482. doi:10.1090/S0002-9947-1943-0012401-3
/// - R. F. Engle (1984). "Wald, likelihood ratio, and Lagrange multiplier tests in econometrics".
///   Handbook of Econometrics 2, 775-826. doi:10.1016/S1573-4412(84)02005-5
/// @see score_test(), likelihood_ratio_test()
[[nodiscard]] RestrictionTestResult wald_test(const MaximumLikelihoodFit& fit,
                                              const linalg::DenseMatrix<double>& restrictions,
                                              const std::vector<double>& values, bool robust = false);

/// @brief Rao score (Lagrange multiplier) test from the full model evaluated at the restricted estimate.
///
/// With U the total score and J the observed information of the full model at the
/// restricted maximum likelihood estimate \f$\tilde\theta\f$, the statistic is
/// \f$U^\top J^{-1}U\f$. Obtain the input with evaluate_likelihood() at
/// \f$\tilde\theta\f$ written in the full parameterization. Only the restricted model
/// must be fitted. Using the observed rather than expected information can give a
/// different value from software that uses Fisher information.
/// @param at_restricted Full-model scores and information at the restricted estimate.
/// @param rank Number of independent restrictions (degrees of freedom), at least one.
/// @return Statistic, rank, p-value and method.
/// @throws std::invalid_argument For a zero rank, a rank above the parameter count, or a singular information matrix.
/// @par Example
/// @code
/// const std::vector<double> y{2.1, 0.7, 1.4, 3.2, 0.9};
/// const auto at = dm::stats::evaluate_likelihood(y.size(), {1.0},
///     [&](const std::vector<double>& t, std::size_t i) { return std::log(t[0]) - t[0] * y[i]; });
/// std::println("{:.4f}", dm::stats::score_test(at, 1).statistic); // 2.1780
/// @endcode
/// @par References
/// - C. R. Rao (1948). "Large sample tests of statistical hypotheses concerning several parameters
///   with applications to problems of estimation". Proc. Cambridge Philos. Soc. 44(1), 50-57.
///   doi:10.1017/S0305004100023987
/// @see evaluate_likelihood(), wald_test()
[[nodiscard]] RestrictionTestResult score_test(const MaximumLikelihoodFit& at_restricted, std::size_t rank);

/// @brief Likelihood-ratio test \f$2(\ell_{\text{full}}-\ell_{\text{restricted}})\f$ of nested models.
///
/// Small negative differences from optimization error are set to zero. Regularity
/// requires the restricted values to be interior to the full parameter space; on a
/// boundary the chi-square reference is conservative.
/// @param full_log_likelihood Maximized log likelihood of the full model.
/// @param restricted_log_likelihood Maximized log likelihood of the nested model.
/// @param rank Number of independent restrictions (degrees of freedom), at least one.
/// @return Statistic, rank, p-value and method.
/// @throws std::invalid_argument For nonfinite log likelihoods, a zero rank, or a restricted
/// log likelihood exceeding the full one by more than \f$10^{-6}(1+|\ell_{\text{full}}|)\f$.
/// @par Example
/// @code
/// const auto lr = dm::stats::likelihood_ratio_test(-10.0, -12.5, 2);
/// std::println("{:.1f} {:.4f}", lr.statistic, lr.p_value); // 5.0 0.0821
/// @endcode
/// @par References
/// - S. S. Wilks (1938). "The large-sample distribution of the likelihood ratio for testing
///   composite hypotheses". Annals of Mathematical Statistics 9(1), 60-62. doi:10.1214/aoms/1177732360
/// @see wald_test(), score_test()
[[nodiscard]] RestrictionTestResult likelihood_ratio_test(double full_log_likelihood, double restricted_log_likelihood,
                                                          std::size_t rank);

} // namespace datamunge::stats
