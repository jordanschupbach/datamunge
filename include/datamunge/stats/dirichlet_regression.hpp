#pragma once

/// @file dirichlet_regression.hpp
/// @brief Dirichlet regression for compositional responses in the mean/precision ("alternative")
/// parametrization of Maier (2014): multinomial-logit means with a reference part and a log-linear precision.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Options for dirichlet_regression().
/// @par Example
/// @code
/// dm::stats::DirichletRegressionOptions o{.base = 2};
/// std::println("{}", o.base); // 2
/// @endcode
struct DirichletRegressionOptions {
    std::size_t base{0};             ///< Reference part whose mean coefficients are fixed at 0 (DirichletReg `base` - 1).
    bool compress_boundaries{true};  ///< If any closed response is 0 or 1, use \f$(y(n-1)+1/D)/n\f$ (Smithson and
                                     ///< Verkuilen 2006, as `DirichletReg::DR_data`); otherwise such data throw.
    std::size_t max_iterations{200}; ///< Fisher-scoring iterations.
    double tolerance{1e-10};         ///< Convergence tolerance on the largest coefficient step.
};

/// @brief Means, precisions and Dirichlet parameters at a set of covariates.
/// @par Example
/// @code
/// dm::stats::DirichletPrediction p;
/// std::println("{}", p.precision.size()); // 0
/// @endcode
struct DirichletPrediction {
    linalg::DenseMatrix<double> mean;  ///< n x D expected compositions \f$\mu_{ij}\f$.
    std::vector<double> precision;     ///< \f$\phi_i=\exp(z_i^\top\gamma)\f$.
    linalg::DenseMatrix<double> alpha; ///< n x D Dirichlet parameters \f$\alpha_{ij}=\mu_{ij}\phi_i\f$.
};

/// @brief A fitted Dirichlet regression.
/// @par Example
/// @code
/// dm::stats::DirichletRegression f;
/// std::println("{}", f.converged); // false
/// @endcode
struct DirichletRegression {
    linalg::DenseMatrix<double> mean_coefficients;    ///< p x D: column j holds \f$\beta_j\f$ (0 for the base part).
    linalg::DenseMatrix<double> mean_standard_errors; ///< p x D (NaN for the base part).
    std::vector<double> precision_coefficients;       ///< \f$\gamma\f$ (length q).
    std::vector<double> precision_standard_errors;    ///< Standard errors of \f$\gamma\f$.
    linalg::DenseMatrix<double> covariance; ///< Inverse observed information; order: \f$\beta_j\f$ for each non-base part
                                            ///< in part order, then \f$\gamma\f$. Empty if singular.
    double log_likelihood{};                ///< Maximized log-likelihood (of the possibly compressed response).
    double aic{};                           ///< \f$-2\ell+2k\f$.
    double bic{};                           ///< \f$-2\ell+k\log n\f$.
    std::size_t observations{};             ///< n.
    std::size_t parameters{};               ///< k = p(D - 1) + q.
    std::size_t iterations{};               ///< Fisher-scoring iterations.
    bool converged{};                       ///< Whether the tolerance was reached.
    std::size_t base{};                     ///< Reference part.
    bool compressed{};                      ///< Whether the boundary compression was applied.
    DirichletPrediction fitted;             ///< Fitted values.
    linalg::DenseMatrix<double> standardized_residuals; ///< \f$(y-\mu)/\sqrt{\mu(1-\mu)/(1+\phi)}\f$.

    /// @brief Predict at new covariates.
    /// @param mean_design n x p mean design (same columns as the fit).
    /// @param precision_design n x q precision design.
    /// @return Predictions.
    /// @throws std::invalid_argument On a column-count or row-count mismatch.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> one(4, 1, 1.0);
    /// const dm::linalg::DenseMatrix<double> y(4, 2, {0.2, 0.8, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5});
    /// const auto f = dm::stats::dirichlet_regression(one, one, y);
    /// std::println("{:.3f}", f.predict(one, one).mean(0, 0)); // 0.350
    /// @endcode
    [[nodiscard]] DirichletPrediction predict(const linalg::DenseMatrix<double>& mean_design,
                                              const linalg::DenseMatrix<double>& precision_design) const;
};

/// @brief Fit \f$y_i\sim\mathrm{Dir}(\mu_i\phi_i)\f$ with \f$\mu_{ij}=\exp(x_i^\top\beta_j)/\sum_k\exp(x_i^\top\beta_k)\f$,
/// \f$\beta_{\text{base}}=0\f$, and \f$\log\phi_i=z_i^\top\gamma\f$, by maximum likelihood (as
/// `DirichletReg::DirichReg(..., model = "alternative")`).
///
/// Supply intercept columns explicitly. Responses are closed to sum 1. Fisher scoring uses the analytic score and
/// expected information from least-squares alr and method-of-moments starting values. The covariance inverts the
/// observed information, which is obtained by central differences of the analytic score. With D = 2 this is beta
/// regression.
/// @param mean_design n x p mean design.
/// @param precision_design n x q precision design.
/// @param response n x D nonnegative compositions (D >= 2).
/// @param options Options.
/// @return The fit.
/// @throws std::invalid_argument On mismatched shapes, nonfinite input, a base out of range, rows without a positive
/// sum, or boundary values with compression disabled.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> one(4, 1, 1.0);
/// const dm::linalg::DenseMatrix<double> y(4, 2, {0.2, 0.8, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5});
/// std::println("{}", dm::stats::dirichlet_regression(one, one, y).parameters); // 2
/// @endcode
/// @par References
/// - M. J. Maier (2014), DirichletReg: Dirichlet regression for compositional data in R, Research Report Series 125,
///   WU Vienna.
/// - R. H. Hijazi and R. W. Jernigan (2009), Modelling compositional data using Dirichlet regression models,
///   Journal of Applied Probability and Statistics 4(1), 77-91.
DirichletRegression dirichlet_regression(const linalg::DenseMatrix<double>& mean_design,
                                         const linalg::DenseMatrix<double>& precision_design,
                                         const linalg::DenseMatrix<double>& response,
                                         const DirichletRegressionOptions& options = {});

} // namespace datamunge::stats
