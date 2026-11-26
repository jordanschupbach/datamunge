#pragma once

/// @file maximum_likelihood.hpp
/// @brief Maximum likelihood for a supplied per-observation log likelihood with box constraints.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <functional>
#include <vector>

namespace datamunge::stats {

/// @brief Log likelihood contribution \f$\ell_i(\theta)\f$ of observation i.
///
/// Return negative infinity where the parameter has zero likelihood; NaN is an error.
using ObservationLogLikelihood = std::function<double(const std::vector<double>& theta, std::size_t i)>;

/// @brief Score contribution \f$\partial\ell_i/\partial\theta\f$ of observation i, one entry per parameter.
using ObservationScore = std::function<std::vector<double>(const std::vector<double>& theta, std::size_t i)>;

/// @brief Constraints and controls for maximum_likelihood().
/// @par Example
/// @code
/// const dm::stats::MaximumLikelihoodOptions options{.lower = {0.0}, .max_iterations = 200};
/// std::println("{}", options.max_iterations); // 200
/// @endcode
struct MaximumLikelihoodOptions {
    /// Inclusive lower bounds, one per parameter or empty for none; use negative infinity
    /// for an unbounded parameter.
    std::vector<double> lower{};

    /// Inclusive upper bounds, one per parameter or empty for none; use positive infinity
    /// for an unbounded parameter.
    std::vector<double> upper{};
    std::size_t max_iterations{500};  ///< Maximum quasi-Newton iterations.

    /// Convergence threshold on the largest absolute score of the total log likelihood,
    /// in the optimizer's unconstrained coordinates.
    double gradient_tolerance{1e-6};
    double relative_step{1e-5};       ///< Relative finite-difference step for numerical derivatives.
};

/// @brief A maximum likelihood fit with per-observation scores and three covariance estimates.
/// @par Example
/// @code
/// const dm::stats::MaximumLikelihoodFit fit;
/// std::println("{}", fit.estimate.size()); // 0
/// @endcode
struct MaximumLikelihoodFit {
    std::vector<double> estimate;          ///< Maximizer \f$\hat\theta\f$ on the original scale.
    double log_likelihood{};               ///< \f$\sum_i\ell_i(\hat\theta)\f$.

    /// \f$n\times p\f$ matrix whose row i is the score contribution \f$s_i(\hat\theta)\f$.
    linalg::DenseMatrix<double> score_contributions;
    linalg::DenseMatrix<double> observed_information; ///< \f$J=-\nabla^2\sum_i\ell_i(\hat\theta)\f$.

    /// Outer product of the scores \f$K=\sum_i s_is_i^\top\f$, the BHHH information estimate.
    linalg::DenseMatrix<double> outer_product;
    linalg::DenseMatrix<double> covariance;        ///< Model-based \f$J^{-1}\f$.
    linalg::DenseMatrix<double> robust_covariance; ///< Huber–White sandwich \f$J^{-1}KJ^{-1}\f$.
    std::vector<double> standard_errors;           ///< Square roots of the diagonal of `covariance`.

    /// Whether each estimate lies within a relative \f$10^{-6}\f$ of a finite bound; regular
    /// asymptotic theory, and the covariances, do not apply to such parameters.
    std::vector<bool> at_bound;
    std::size_t iterations{};                      ///< Quasi-Newton iterations taken.
    bool converged{};                              ///< Whether the gradient tolerance was met.
};

/// @brief Maximizes \f$\sum_{i<n}\ell_i(\theta)\f$ subject to \f$l\le\theta\le u\f$.
///
/// Each bounded parameter is mapped to an unconstrained coordinate (log for one bound,
/// logit for two) and the total log likelihood is maximized there by BFGS with a
/// backtracking Armijo line search. Scores come from `score` when supplied, otherwise from
/// central differences of each \f$\ell_i\f$. At the maximizer the observed information
/// is the negated Hessian on the original scale, computed by central differences of the
/// total score, symmetrized, and stepped one-sidedly next to a bound. When J is
/// singular, `covariance`, `robust_covariance` and `standard_errors` are filled with NaN.
/// The sandwich estimate remains valid for a misspecified model with independent
/// observations; the model-based one requires a correct model. The callbacks are
/// called serially and must be deterministic.
/// @param n Number of observations (at least one).
/// @param start Finite starting values strictly inside the bounds.
/// @param log_likelihood Per-observation log likelihood.
/// @param options Bounds, iteration limit and tolerances.
/// @param score Optional analytic per-observation score.
/// @return Estimate, log likelihood, scores, information matrices, covariances and convergence.
/// @throws std::invalid_argument For an empty or misplaced start, malformed bounds, a missing callback,
/// a nonfinite log likelihood at the start, NaN from a callback, or a score of the wrong length.
/// @par Example
/// @code
/// const std::vector<double> y{2.1, 0.7, 1.4, 3.2, 0.9};
/// const auto fit = dm::stats::maximum_likelihood(y.size(), {1.0},
///     [&](const std::vector<double>& t, std::size_t i) { return std::log(t[0]) - t[0] * y[i]; },
///     {.lower = {0.0}});
/// std::println("{:.4f} {:.4f}", fit.estimate[0], fit.standard_errors[0]); // 0.6024 0.2694
/// @endcode
/// @par References
/// - R. A. Fisher (1922). "On the mathematical foundations of theoretical statistics". Phil. Trans.
///   R. Soc. A 222, 309-368. doi:10.1098/rsta.1922.0009
/// - E. K. Berndt, B. H. Hall, R. E. Hall and J. A. Hausman (1974). "Estimation and inference in
///   nonlinear structural models". Annals of Economic and Social Measurement 3(4), 653-665.
/// - H. White (1982). "Maximum likelihood estimation of misspecified models". Econometrica 50(1),
///   1-25. doi:10.2307/1912526
/// - J. Nocedal and S. J. Wright (2006). Numerical Optimization, 2nd ed., Springer, ch. 6.
///   doi:10.1007/978-0-387-40065-5
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats4/html/mle.html
/// @see profile_likelihood_interval()
[[nodiscard]] MaximumLikelihoodFit maximum_likelihood(std::size_t n, const std::vector<double>& start,
                                                      const ObservationLogLikelihood& log_likelihood,
                                                      const MaximumLikelihoodOptions& options = {},
                                                      const ObservationScore& score = {});

/// @brief Scores, information and covariances of a log likelihood at given parameters, without fitting.
///
/// Computes the same quantities as maximum_likelihood() at `theta`, which may lie on a
/// bound. Use it to evaluate a full model at a restricted estimate, as in score_test().
/// `iterations` is zero and `converged` is false.
/// @param n Number of observations (at least one).
/// @param theta Finite parameters inside the inclusive bounds.
/// @param log_likelihood Per-observation log likelihood.
/// @param options Bounds and finite-difference step; the iteration controls are unused.
/// @param score Optional analytic per-observation score.
/// @return Log likelihood, scores, information matrices and covariances at theta.
/// @throws std::invalid_argument For invalid input as in maximum_likelihood(), or a nonfinite log likelihood at theta.
/// @par Example
/// @code
/// const std::vector<double> y{2.1, 0.7, 1.4, 3.2, 0.9};
/// const auto at = dm::stats::evaluate_likelihood(y.size(), {1.0},
///     [&](const std::vector<double>& t, std::size_t i) { return std::log(t[0]) - t[0] * y[i]; });
/// std::println("{:.4f}", at.score_contributions(0, 0)); // -1.1000
/// @endcode
/// @see maximum_likelihood(), score_test()
[[nodiscard]] MaximumLikelihoodFit evaluate_likelihood(std::size_t n, const std::vector<double>& theta,
                                                       const ObservationLogLikelihood& log_likelihood,
                                                       const MaximumLikelihoodOptions& options = {},
                                                       const ObservationScore& score = {});

} // namespace datamunge::stats
