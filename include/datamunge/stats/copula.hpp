#pragma once

/// @file copula.hpp
/// @brief Gaussian and Student t copulas: pseudo-observations, densities, fitting by maximum pseudo-likelihood or
/// Kendall's-tau inversion, tail dependence and simulation.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Elliptical copula family.
/// @par Example
/// @code
/// const auto f = dm::stats::CopulaFamily::student_t;
/// std::println("{}", f == dm::stats::CopulaFamily::student_t); // true
/// @endcode
enum class CopulaFamily {
    gaussian,  ///< Gaussian (normal) copula: no tail dependence.
    student_t, ///< Student t copula with \f$\nu\f$ degrees of freedom: symmetric tail dependence.
};

/// @brief Estimation method for fit_copula().
/// @par Example
/// @code
/// const auto m = dm::stats::CopulaMethod::kendall_inversion;
/// std::println("{}", m == dm::stats::CopulaMethod::kendall_inversion); // true
/// @endcode
enum class CopulaMethod {
    maximum_pseudo_likelihood, ///< Maximize the copula log-likelihood of the pseudo-observations (`copula` "mpl").
    kendall_inversion,         ///< \f$R_{jk}=\sin(\pi\tau_{jk}/2)\f$; for t, \f$\nu\f$ then by MPL with R fixed ("itau.mpl").
};

/// @brief Options for fit_copula().
/// @par Example
/// @code
/// dm::stats::CopulaOptions o{.family = dm::stats::CopulaFamily::student_t};
/// std::println("{}", o.max_iterations); // 1000
/// @endcode
struct CopulaOptions {
    CopulaFamily family{CopulaFamily::gaussian};                      ///< Family.
    CopulaMethod method{CopulaMethod::maximum_pseudo_likelihood};     ///< Estimator.
    double df{std::numeric_limits<double>::quiet_NaN()};              ///< Fixed t degrees of freedom (NaN: estimate within [0.5, 1000]).
    std::size_t max_iterations{1000};                                 ///< L-BFGS iterations.
    double tolerance{1e-10};                                          ///< L-BFGS tolerance.
};

/// @brief A fitted elliptical copula.
/// @par Example
/// @code
/// dm::stats::CopulaFit f{.n = 10};
/// std::println("{}", f.n); // 10
/// @endcode
struct CopulaFit {
    CopulaFamily family{CopulaFamily::gaussian}; ///< Family.
    linalg::DenseMatrix<double> correlation;     ///< Correlation (dispersion) matrix R.
    double df{std::numeric_limits<double>::infinity()}; ///< Degrees of freedom (infinity for Gaussian).
    double log_likelihood{};                     ///< Copula log pseudo-likelihood at the estimate.
    double aic{};                                ///< \f$-2\ell+2k\f$.
    double bic{};                                ///< \f$-2\ell+k\log n\f$.
    std::size_t parameters{};                    ///< Free parameters k.
    std::size_t n{};                             ///< Observations.
    linalg::DenseMatrix<double> tail_dependence; ///< Pairwise upper (= lower) tail-dependence coefficients.
    bool projected{};                            ///< Whether a Kendall-inverted matrix was projected to be positive definite.
};

/// @brief Pseudo-observations: column ranks divided by n + 1 (average ranks for ties, as `copula::pobs`).
/// @param x n x d data.
/// @return n x d matrix in (0, 1).
/// @throws std::invalid_argument With fewer than 2 rows or nonfinite values.
/// @par Example
/// @code
/// const auto u = dm::stats::pseudo_observations(dm::linalg::DenseMatrix<double>(3, 1, {5, 1, 3}));
/// std::println("{}", u(0, 0)); // 0.75
/// @endcode
linalg::DenseMatrix<double> pseudo_observations(const linalg::DenseMatrix<double>& x);

/// @brief Log density of an elliptical copula at one point u in (0, 1)^d.
/// @param u Point.
/// @param correlation Positive-definite correlation matrix.
/// @param family Family.
/// @param df Degrees of freedom (t only).
/// @return \f$\log c(u)\f$.
/// @throws std::invalid_argument On a dimension mismatch, u outside (0, 1), a non-PD matrix or df <= 0.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> r(2, 2, {1, 0, 0, 1});
/// const double c = dm::stats::copula_log_density({0.3, 0.8}, r, dm::stats::CopulaFamily::gaussian);
/// std::println("{:.1f}", std::abs(c)); // 0.0
/// @endcode
double copula_log_density(const std::vector<double>& u, const linalg::DenseMatrix<double>& correlation,
                          CopulaFamily family, double df = std::numeric_limits<double>::infinity());

/// @brief Fit a Gaussian or t copula to pseudo-observations (or any data in (0, 1)^d).
/// @param u n x d pseudo-observations, d >= 2.
/// @param options Options.
/// @return The fit.
/// @throws std::invalid_argument With d < 2, n <= d, values outside (0, 1) or a nonpositive fixed df.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> u(4, 2, {.2, .3, .4, .35, .6, .7, .8, .6});
/// std::println("{:.3f}", dm::stats::fit_copula(u, {.method = dm::stats::CopulaMethod::kendall_inversion})
///                            .correlation(0, 1)); // 0.866
/// @endcode
CopulaFit fit_copula(const linalg::DenseMatrix<double>& u, const CopulaOptions& options = {});

/// @brief Tail-dependence coefficient \f$\lambda=2t_{\nu+1}\bigl(-\sqrt{(\nu+1)(1-\rho)/(1+\rho)}\bigr)\f$ of a
/// bivariate t copula (0 for the Gaussian, \f$\nu=\infty\f$).
/// @param rho Correlation in (-1, 1].
/// @param df Degrees of freedom (> 0).
/// @return \f$\lambda_U=\lambda_L\f$.
/// @throws std::invalid_argument If rho is outside (-1, 1] or df <= 0.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::t_copula_tail_dependence(0.5, 4)); // 0.2532
/// @endcode
double t_copula_tail_dependence(double rho, double df);

/// @brief Simulate n points from an elliptical copula.
/// @param correlation d x d positive-definite correlation matrix.
/// @param family Family.
/// @param df Degrees of freedom (t only).
/// @param n Number of draws.
/// @param seed Random seed.
/// @return n x d matrix of uniforms.
/// @throws std::invalid_argument For a non-PD matrix or df <= 0.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> r(2, 2, {1, .5, .5, 1});
/// std::println("{}", dm::stats::simulate_copula(r, dm::stats::CopulaFamily::student_t, 4, 10, 1).rows()); // 10
/// @endcode
linalg::DenseMatrix<double> simulate_copula(const linalg::DenseMatrix<double>& correlation, CopulaFamily family,
                                            double df, std::size_t n, std::uint64_t seed);

} // namespace datamunge::stats
