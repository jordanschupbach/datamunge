#pragma once

/// @file polychoric.hpp
/// @brief Tetrachoric and polychoric correlations: the correlation of a bivariate normal latent variable
/// discretized by thresholds into an observed contingency table, by two-step or full maximum likelihood.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Standard bivariate normal distribution function \f$P(X\le h,Y\le k)\f$ with correlation rho.
///
/// Genz's (2004) Gauss-Legendre method (the algorithm of `mvtnorm`'s bivariate case), accurate to about
/// \f$10^{-15}\f$; infinite limits are allowed.
/// @param h Upper limit for X.
/// @param k Upper limit for Y.
/// @param rho Correlation in [-1, 1].
/// @return The probability.
/// @throws std::invalid_argument If rho is outside [-1, 1] or an argument is NaN.
/// @par Example
/// @code
/// std::println("{:.6f}", dm::stats::bivariate_normal_cdf(0, 0, 0.5)); // 0.333333
/// @endcode
double bivariate_normal_cdf(double h, double k, double rho);

/// @brief Estimation method for polychoric_correlation().
/// @par Example
/// @code
/// const auto m = dm::stats::PolychoricEstimator::maximum_likelihood;
/// std::println("{}", m == dm::stats::PolychoricEstimator::maximum_likelihood); // true
/// @endcode
enum class PolychoricEstimator {
    two_step,           ///< Thresholds from the marginal proportions, then rho by maximum likelihood (Olsson 1979).
    maximum_likelihood, ///< Rho and all thresholds jointly by maximum likelihood.
};

/// @brief Options for polychoric_correlation().
/// @par Example
/// @code
/// dm::stats::PolychoricOptions o{.confidence = 0.9};
/// std::println("{}", o.confidence); // 0.9
/// @endcode
struct PolychoricOptions {
    PolychoricEstimator estimator{PolychoricEstimator::two_step}; ///< Estimator.
    double confidence{0.95};          ///< Level of the Fisher-z Wald interval.
    std::size_t max_iterations{200};  ///< Newton/Fisher-scoring iterations.
    double tolerance{1e-11};          ///< Convergence tolerance on the score.
};

/// @brief A polychoric (or tetrachoric) correlation estimate with thresholds and model test.
/// @par Example
/// @code
/// dm::stats::PolychoricCorrelation p{.rho = 0.4};
/// std::println("{}", p.rho); // 0.4
/// @endcode
struct PolychoricCorrelation {
    double rho{};                          ///< Latent correlation.
    double standard_error{};               ///< From the observed information (two-step: rho only, thresholds fixed).
    double lower{};                        ///< Wald interval on the Fisher-z scale, back-transformed.
    double upper{};                        ///< Upper limit.
    std::vector<double> row_thresholds;    ///< Latent thresholds for the rows (r - 1 values).
    std::vector<double> column_thresholds; ///< Latent thresholds for the columns (c - 1 values).
    std::vector<double> row_threshold_se;  ///< ML only: threshold standard errors (empty for two-step).
    std::vector<double> column_threshold_se; ///< ML only: threshold standard errors.
    double log_likelihood{};               ///< Multinomial log-likelihood at the estimate.
    double chi_square{};                   ///< Likelihood-ratio test of the bivariate-normal model against the saturated table.
    double df{};                           ///< rc - r - c (0 for a 2 x 2 table).
    double p_value{};                      ///< Upper-tail chi-square p-value (NaN when df = 0).
    double n{};                            ///< Total count.
    std::size_t iterations{};              ///< Iterations used.
    bool converged{};                      ///< Whether the score fell below tolerance.
};

/// @brief Polychoric correlation from an r x c contingency table of counts (rows: categories of the first
/// variable in order; columns: of the second).
///
/// All-zero rows and columns are dropped (as in `polycor::polychor`). Zero cells are allowed.
/// @param table Counts, table[i][j] >= 0.
/// @param options Options.
/// @return The estimate.
/// @throws std::invalid_argument If the table is ragged, has negative or nonfinite counts, or has fewer than two
/// nonempty rows or columns.
/// @par Example
/// @code
/// const auto p = dm::stats::polychoric_correlation({{40, 10}, {10, 40}});
/// std::println("{:.4f}", p.rho); // 0.8090
/// @endcode
PolychoricCorrelation polychoric_correlation(const std::vector<std::vector<double>>& table,
                                             const PolychoricOptions& options = {});

/// @brief Polychoric correlation from paired ordinal codes (categories ordered by value).
/// @param x First variable's codes.
/// @param y Second variable's codes.
/// @param options Options.
/// @return The estimate.
/// @throws std::invalid_argument On length mismatch or fewer than two distinct values in either variable.
/// @par Example
/// @code
/// const std::vector<int> x{1, 1, 2, 2, 3, 3, 1, 3}, y{1, 2, 2, 3, 3, 3, 1, 2};
/// std::println("{}", dm::stats::polychoric_correlation(x, y).rho > 0.5); // true
/// @endcode
PolychoricCorrelation polychoric_correlation(const std::vector<int>& x, const std::vector<int>& y,
                                             const PolychoricOptions& options = {});

/// @brief Tetrachoric correlation of a 2 x 2 table \f$\begin{pmatrix}a&b\\c&d\end{pmatrix}\f$.
/// @param a Count (row 1, column 1).
/// @param b Count (row 1, column 2).
/// @param c Count (row 2, column 1).
/// @param d Count (row 2, column 2).
/// @param options Options (two-step and ML coincide for 2 x 2 tables).
/// @return The estimate.
/// @throws std::invalid_argument As for the table overload.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::tetrachoric_correlation(40, 10, 10, 40).rho); // 0.8090
/// @endcode
PolychoricCorrelation tetrachoric_correlation(double a, double b, double c, double d,
                                              const PolychoricOptions& options = {});

/// @brief Matrix of pairwise two-step polychoric correlations of ordinal variables (unit diagonal).
///
/// The result need not be positive definite; smooth it before factor analysis if necessary.
/// @param columns Ordinal codes, one vector per variable, all of length n.
/// @return Symmetric correlation matrix.
/// @throws std::invalid_argument With fewer than two variables or on any pairwise error.
/// @par Example
/// @code
/// const std::vector<std::vector<int>> v{{1, 2, 2, 3, 1, 3}, {1, 2, 3, 3, 1, 2}, {2, 1, 2, 1, 2, 1}};
/// std::println("{}", dm::stats::polychoric_matrix(v)(0, 0)); // 1
/// @endcode
linalg::DenseMatrix<double> polychoric_matrix(const std::vector<std::vector<int>>& columns);

} // namespace datamunge::stats
