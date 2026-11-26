#pragma once

/// @file mvn_probability.hpp
/// @brief Multivariate normal rectangle probabilities by Genz's randomized quasi-Monte Carlo method.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief Accuracy and work controls for mvn_rectangle_probability().
/// @par Example
/// @code
/// const dm::stats::MvnProbabilityOptions options{.abs_tol=1e-5};
/// std::println("{}", options.shifts); // 12
/// @endcode
struct MvnProbabilityOptions {
    double abs_tol{1e-4};        ///< Stop once the error estimate is at most this absolute value.
    std::size_t max_points{100000}; ///< Upper limit on integrand evaluations over all shifts.
    std::size_t shifts{12};      ///< Independent random shifts of each lattice rule (at least 2).
    std::uint64_t seed{1};       ///< Seed for the random shifts.
};

/// @brief A rectangle probability, its integration error estimate and the work spent.
/// @par Example
/// @code
/// const dm::stats::MvnProbability result;
/// std::println("{}", result.converged); // false
/// @endcode
struct MvnProbability {
    double value{};          ///< Estimate of \f$P(a\le X\le b)\f$, clipped to \f$[0,1]\f$.

    /// Three standard errors of the mean over the randomized shifts, about a 99.7%
    /// half-width; zero when the probability is computed in closed form.
    double error{};
    std::size_t points{};    ///< Integrand evaluations used, counting each antithetic pair twice.
    bool converged{};        ///< Whether `error` reached `abs_tol` within `max_points`.
};

/// @brief Probability that \f$X\sim N(\mu,\Sigma)\f$ falls in the box \f$a\le X\le b\f$.
///
/// Implements Genz's separation of variables: after a Cholesky factorization the
/// integral becomes one over the unit cube of a product of conditional normal
/// interval probabilities. Variables are reordered as the factorization proceeds so
/// that the narrowest expected intervals come first (Gibson, Glasbey and Elston),
/// which reduces the variance of the integrand. The cube is sampled with randomly
/// shifted rank-1 lattice rules with generators \f$\sqrt{p_j}\f$ for primes \f$p_j\f$ and
/// the baker's (antithetic) transform. The rule size grows by about 1.5 per round
/// until the error estimate reaches `abs_tol` or `max_points` is spent. One
/// dimension is exact. Infinite bounds are allowed; \f$a_j=b_j\f$ gives zero.
/// Results depend on the seed only through the integration error.
/// @param lower Lower bounds \f$a\f$, each finite or \f$-\infty\f$.
/// @param upper Upper bounds \f$b\f$, each finite or \f$+\infty\f$, with \f$a_j\le b_j\f$.
/// @param mean Mean \f$\mu\f$ with the same length.
/// @param covariance Symmetric positive definite \f$d\times d\f$ covariance.
/// @param options Tolerance, work limit, shifts and seed.
/// @return Probability, error estimate, evaluations and convergence flag.
/// @throws std::invalid_argument For empty or mismatched inputs, NaN or wrongly signed infinite
/// bounds, \f$a_j>b_j\f$, a nonfinite mean, a covariance that is not symmetric positive definite,
/// a nonpositive tolerance, fewer than two shifts or zero points.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> sigma(2, 2, {1, .5, .5, 1});
/// const double inf = std::numeric_limits<double>::infinity();
/// const auto p = dm::stats::mvn_rectangle_probability({-inf, -inf}, {0, 0}, {0, 0}, sigma);
/// std::println("{:.4f}", p.value); // 0.3333
/// @endcode
/// @par References
/// - A. Genz (1992). "Numerical computation of multivariate normal probabilities". Journal of
///   Computational and Graphical Statistics 1(2), 141-149. doi:10.1080/10618600.1992.10477010
/// - A. Genz and F. Bretz (2009). Computation of Multivariate Normal and t Probabilities.
///   Lecture Notes in Statistics 195, Springer. doi:10.1007/978-3-642-01689-9
/// - G. J. Gibson, C. A. Glasbey and D. A. Elston (1994). "Monte Carlo evaluation of multivariate
///   normal integrals and sensitivity to variate ordering". Advances in Numerical Methods and
///   Applications, World Scientific, 120-126.
/// - https://cran.r-project.org/package=mvtnorm
/// @see mvn_cdf()
[[nodiscard]] MvnProbability mvn_rectangle_probability(
    const std::vector<double>& lower, const std::vector<double>& upper, const std::vector<double>& mean,
    const linalg::DenseMatrix<double>& covariance, const MvnProbabilityOptions& options = {});

/// @brief Multivariate normal distribution function \f$P(X\le b)\f$.
///
/// Same as mvn_rectangle_probability() with every lower bound \f$-\infty\f$.
/// @param upper Upper bounds \f$b\f$, each finite or \f$+\infty\f$.
/// @param mean Mean \f$\mu\f$ with the same length.
/// @param covariance Symmetric positive definite covariance.
/// @param options Tolerance, work limit, shifts and seed.
/// @return Probability, error estimate, evaluations and convergence flag.
/// @throws std::invalid_argument As for mvn_rectangle_probability().
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> sigma(3, 3, {1, .5, .5, .5, 1, .5, .5, .5, 1});
/// const auto p = dm::stats::mvn_cdf({0, 0, 0}, {0, 0, 0}, sigma);
/// std::println("{:.4f}", p.value); // 0.2500
/// @endcode
/// @par References
/// A. Genz (1992), doi:10.1080/10618600.1992.10477010
/// @see mvn_rectangle_probability()
[[nodiscard]] MvnProbability mvn_cdf(const std::vector<double>& upper, const std::vector<double>& mean,
                                     const linalg::DenseMatrix<double>& covariance,
                                     const MvnProbabilityOptions& options = {});

/// @brief Probability that a multivariate t vector \f$X=\mu+Z/\sqrt{W/\nu}\f$ falls in the box \f$a\le X\le b\f$.
///
/// Here \f$Z\sim N(0,\Sigma)\f$ and \f$W\sim\chi^2_\nu\f$ independently, so \f$\Sigma\f$ is the scale
/// matrix (the covariance is \f$\nu\Sigma/(\nu-2)\f$ for \f$\nu>2\f$). The chi scale is one more
/// lattice coordinate, mapped through the chi-square quantile, and the remaining
/// coordinates follow mvn_rectangle_probability() with bounds multiplied by that scale
/// (Genz and Bretz's separation of variables for the t law). One dimension is exact.
/// @param lower Lower bounds, each finite or \f$-\infty\f$.
/// @param upper Upper bounds, each finite or \f$+\infty\f$, not below the lower bounds.
/// @param location Location \f$\mu\f$.
/// @param scale Symmetric positive definite scale matrix \f$\Sigma\f$.
/// @param df Degrees of freedom \f$\nu>0\f$; positive infinity gives the normal law.
/// @param options Tolerance, work limit, shifts and seed.
/// @return Probability, error estimate, evaluations and convergence flag.
/// @throws std::invalid_argument For a nonpositive or NaN df and the errors of mvn_rectangle_probability().
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> r(2, 2, {1, .5, .5, 1});
/// const double inf = std::numeric_limits<double>::infinity();
/// const auto p = dm::stats::mvt_rectangle_probability({-inf, -inf}, {0, 0}, {0, 0}, r, 5);
/// std::println("{:.4f}", p.value); // 0.3333
/// @endcode
/// @par References
/// - A. Genz and F. Bretz (2002). "Comparison of methods for the computation of multivariate t
///   probabilities". Journal of Computational and Graphical Statistics 11(4), 950-971.
///   doi:10.1198/106186002394
/// - https://cran.r-project.org/package=mvtnorm
/// @see mvn_rectangle_probability(), dunnett_test()
[[nodiscard]] MvnProbability mvt_rectangle_probability(
    const std::vector<double>& lower, const std::vector<double>& upper, const std::vector<double>& location,
    const linalg::DenseMatrix<double>& scale, double df, const MvnProbabilityOptions& options = {});

} // namespace datamunge::stats
