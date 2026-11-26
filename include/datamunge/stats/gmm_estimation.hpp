#pragma once

/// @file gmm_estimation.hpp
/// @brief Generalized method of moments: one-step, two-step and iterated estimators with sandwich
/// covariance and Hansen's overidentification test.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <functional>
#include <vector>

namespace datamunge::stats {

/// @brief Moment contributions: returns the n-by-q matrix of g(x_i, theta) at theta.
using MomentFunction = std::function<linalg::DenseMatrix<double>(const std::vector<double>&)>;

/// @brief Jacobian of the mean moments: returns the q-by-k matrix d gbar / d theta at theta.
using MomentJacobian = std::function<linalg::DenseMatrix<double>(const std::vector<double>&)>;

/// @brief How the weighting matrix is chosen.
enum class GmmEstimator {
    one_step, ///< Minimise with the initial weights only (identity unless given).
    two_step, ///< Re-estimate with W = S(theta_1)^{-1} from the one-step estimate.
    iterated, ///< Update W = S(theta)^{-1} and re-estimate until the estimate stops changing.
};

/// @brief Options for generalized_method_of_moments().
/// @par Example
/// @code
/// const dm::stats::GmmOptions o{.estimator = dm::stats::GmmEstimator::iterated};
/// std::println("{}", o.centered); // true
/// @endcode
struct GmmOptions {
    GmmEstimator estimator{GmmEstimator::two_step}; ///< Weighting scheme.
    linalg::DenseMatrix<double> initial_weights;    ///< First-step q-by-q weights; empty means identity.
    MomentJacobian jacobian;      ///< Analytic Jacobian of the mean moments; empty uses central differences.
    bool centered{true};          ///< Centre the moments when estimating their covariance S.
    std::size_t max_iterations{100}; ///< Weight updates of the iterated estimator.
    double tolerance{1e-10};      ///< Relative change in the estimate that ends iteration.
    double confidence{0.95};      ///< Level of the normal confidence intervals.
};

/// @brief Result of generalized_method_of_moments().
/// @par Example
/// @code
/// dm::stats::GmmResult r;
/// r.j_statistic = 1.2;
/// r.df = 2;
/// std::println("{}", r.df); // 2
/// @endcode
struct GmmResult {
    std::vector<double> coefficients;     ///< Estimates.
    std::vector<double> standard_errors;  ///< Square roots of the covariance diagonal.
    std::vector<double> z_values;         ///< coefficients / standard_errors.
    std::vector<double> p_values;         ///< Two-sided normal p-values.
    std::vector<double> lower;            ///< Lower normal confidence limits.
    std::vector<double> upper;            ///< Upper normal confidence limits.
    linalg::DenseMatrix<double> covariance; ///< Estimated covariance of the estimates.
    linalg::DenseMatrix<double> weights;  ///< Weighting matrix used in the final minimisation.
    std::vector<double> mean_moments;     ///< gbar at the estimate.
    double objective{};   ///< gbar' W gbar at the estimate.
    double j_statistic{}; ///< Hansen's J (0 when exactly identified).
    double j_p_value{};   ///< Upper chi-squared(q - k) p-value of J (1 when exactly identified).
    std::size_t df{};     ///< Overidentifying restrictions q - k.
    std::size_t n{};      ///< Observations (rows of the moment matrix).
    std::size_t steps{};  ///< Minimisations performed (1, 2 or the iterated count).
    bool converged{};     ///< All minimisations and, if iterated, the weight updates converged.
};

/// @brief Generalized method of moments estimate of theta from E[g(x, theta)] = 0.
///
/// Each minimisation of \f$\bar g(\theta)^\top W\bar g(\theta)\f$ uses damped Gauss-Newton steps,
/// exact in one step for linear moments. With q > k moments, the two-step estimator re-weights by the
/// inverse of \f$S=n^{-1}\sum_i (g_i-\bar g)(g_i-\bar g)^\top\f$ (uncentred if requested) at the
/// one-step estimate. Its covariance is \f$(G^\top S^{-1}G)^{-1}/n\f$ with G and S at the
/// estimate, and Hansen's \f$J=n\,\bar g^\top W\bar g\f$ is asymptotically \f$\chi^2_{q-k}\f$ under
/// valid moments. The one-step estimator uses the sandwich
/// \f$(G^\top WG)^{-1}G^\top WSWG(G^\top WG)^{-1}/n\f$ and \f$J=n\,\bar g^\top S^{-1}\bar g\f$.
/// These conventions follow R's `gmm` with `vcov = "MDS"`. For linear instrumental-variable moments
/// `gmm` starts from 2SLS; passing `initial_weights` \f$(Z^\top Z/n)^{-1}\f$ reproduces its two-step fit.
/// @param moments Moment contributions as a function of theta.
/// @param start Starting value of theta (k entries).
/// @param options Estimator, weights, Jacobian, tolerance and confidence level.
/// @return Estimates, covariance and J test.
/// @throws std::invalid_argument For an empty start, fewer moments than parameters, inconsistent
/// shapes, bad initial weights, or confidence outside (0, 1).
/// @throws std::runtime_error For nonfinite moments, singular S, or an unidentified Jacobian.
/// @par Example
/// Mean m and variance v of a sample from three moments, E[x - m], E[(x - m)^2 - v] and E[(x - m)^3]:
/// @code
/// const std::vector<double> x{1.2, 0.4, 2.3, 1.9, 0.7, 1.1, 1.6, 0.9, 1.4, 2.0};
/// const auto g = [&](const std::vector<double>& t) {
///     dm::linalg::DenseMatrix<double> m(x.size(), 3);
///     for (std::size_t i = 0; i < x.size(); ++i) {
///         const double e = x[i] - t[0];
///         m(i, 0) = e;
///         m(i, 1) = e * e - t[1];
///         m(i, 2) = e * e * e;
///     }
///     return m;
/// };
/// const auto r = dm::stats::generalized_method_of_moments(g, {1.0, 0.5});
/// std::println("{} {}", r.df, r.coefficients[1] > 0); // 1 true
/// @endcode
GmmResult generalized_method_of_moments(const MomentFunction& moments, const std::vector<double>& start,
                                        const GmmOptions& options = {});

} // namespace datamunge::stats
