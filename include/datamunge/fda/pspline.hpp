#pragma once

/// @file pspline.hpp
/// @brief Fixed-basis penalized least squares, coefficient differences and smoother diagnostics.

#include <cstddef>
#include <vector>

#include <datamunge/fda/bspline.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

namespace datamunge::fda {

/// @brief Difference-penalty settings for fit_pspline().
///
/// The loss is summed squared residuals plus lambda times squared coefficient
/// differences, without division by the observation count. Basis degree is independent
/// of penalty order; knot spacing is not included in the difference operator.
/// @par Example
/// @code
/// dm::fda::PSplineOptions options{2.0, 1}; // First differences, lambda = 2.
/// @endcode
/// @par References
/// - P. H. C. Eilers and B. D. Marx (1996). Flexible smoothing with B-splines and penalties.
///   https://doi.org/10.1214/ss/1038425655
/// @see fit_pspline()
struct PSplineOptions {
    double      lambda{1.0}; ///< Finite nonnegative multiplier of the unscaled difference penalty.
    std::size_t penalty_order{2}; ///< Positive difference order, less than the basis size.
};

/// @brief Owned P-spline fit on a supplied BSplineBasis, with fixed-lambda diagnostics.
///
/// Fields preserve observation/basis order. The default size-based fit uses repeated
/// boundary knots; PSplineRegression instead extends equally spaced knots past the domain.
/// Changing fields manually can invalidate diagnostics. Const methods are safe to call
/// concurrently if the object is not mutated. No uncertainty intervals are estimated.
/// @par Example
/// @code
/// const auto fit = dm::fda::fit_pspline({0,1,2}, {0,1,0}, 1, 3, {1.0,2});
/// std::print("{:.6f}\n", fit.coefficients[1]); // 0.428571
/// @endcode
/// @par References
/// - P. H. C. Eilers and B. D. Marx (1996). Flexible smoothing with B-splines and penalties.
///   https://doi.org/10.1214/ss/1038425655
/// @see fit_pspline(), PSplineRegression
struct PSplineFitResult {
    BSplineBasis               basis; ///< Owned basis, including its full knot sequence.
    std::vector<double>        coefficients; ///< K fitted basis coefficients.
    std::vector<double>        fitted; ///< n fitted responses in training order.
    std::vector<double>        residuals; ///< n observed minus fitted responses.
    linalg::DenseMatrix<double> penalty_matrix; ///< K by K matrix D transpose D.
    double                     lambda{1.0}; ///< Fixed smoothing multiplier used by the fit.
    std::size_t                penalty_order{2}; ///< Order of adjacent coefficient differences.
    double                     residual_sum_of_squares{0.0}; ///< Sum of squared residuals.
    double                     total_sum_of_squares{0.0}; ///< Sum of squared centred responses.
    double                     penalty{0.0}; ///< Squared norm of D times coefficients, without lambda.
    double                     objective{0.0}; ///< RSS plus lambda times penalty.
    double                     r_squared{0.0}; ///< 1 - RSS/TSS; constant response gives 1 for RSS=0, otherwise 0.
    double                     adjusted_r_squared{0.0}; ///< Uses n-edf as residual df; falls back to R squared if undefined.
    double                     effective_degrees_of_freedom{0.0}; ///< Trace of the fixed-lambda smoother matrix.
    double                     residual_degrees_of_freedom{0.0}; ///< n minus effective df, not the residual variance divisor.
    std::size_t                observations{0}; ///< Number n of paired training observations.
    std::vector<double>        leverage; ///< n diagonal entries of B (B transpose B + lambda P) inverse B transpose.

    /// @brief Evaluates the fitted basis expansion; finite out-of-domain points return zero.
    /// @param points Finite evaluation coordinates; empty is accepted.
    /// @return Owned predictions, one per point, in input order.
    /// @throws std::invalid_argument For non-finite coordinates or coefficient/basis size mismatch.
    /// @par Example
    /// @code
    /// const auto fit = dm::fda::fit_pspline({0,1,2}, {0,1,0}, 1, 3, {1.0,2});
    /// std::print("{:.6f}\n", fit.predict({1})[0]); // 0.428571
    /// @endcode
    /// @see PSplineRegression::predict()
    std::vector<double> predict(const std::vector<double>& points) const;

    /// @brief Generalized cross-validation n RSS / (n - edf) squared, with response-squared units.
    /// @return Fixed-lambda score; infinity when n-edf is numerically zero or negative.
    /// @par Example
    /// @code
    /// const auto fit = dm::fda::fit_pspline({0,1,2}, {0,1,0}, 1, 3, {1.0,2});
    /// std::print("{:.3f}\n", fit.gcv()); // 2.000
    /// @endcode
    /// @par References
    /// - G. H. Golub, M. Heath and G. Wahba (1979). Generalized cross-validation as a
    ///   method for choosing a good ridge parameter. https://doi.org/10.1080/00401706.1979.10489751
    /// @see loo_cv()
    double gcv() const noexcept;

    /// @brief Delete-one residuals e_i/(1-h_ii), keeping knots and lambda fixed.
    /// @return Owned n-vector; NaN where 1-h_ii <= 64 machine epsilons (deletion not identified).
    /// @throws std::invalid_argument If leverage and residual vector lengths differ.
    /// @par Example
    /// @code
    /// const auto fit = dm::fda::fit_pspline({0,1,2}, {0,1,0}, 1, 3, {1.0,2});
    /// std::print("{:.1f}\n", fit.loo_residuals()[1]); // 1.0
    /// @endcode
    /// @par References
    /// - G. H. Golub, M. Heath and G. Wahba (1979). Generalized cross-validation as a
    ///   method for choosing a good ridge parameter. https://doi.org/10.1080/00401706.1979.10489751
    /// @see loo_cv()
    std::vector<double> loo_residuals() const;

    /// @brief Mean squared fixed-basis, fixed-lambda delete-one residual, without refitting GCV.
    /// @return Score in response-squared units; infinity for empty/undefined deletion diagnostics.
    /// @throws std::invalid_argument If leverage and residual vector lengths differ.
    /// @par Example
    /// @code
    /// const auto fit = dm::fda::fit_pspline({0,1,2}, {0,1,0}, 1, 3, {1.0,2});
    /// std::print("{:.3f}\n", fit.loo_cv()); // 3.000
    /// @endcode
    /// @see loo_residuals(), gcv()
    double loo_cv() const;
};

/// @brief Forward difference operator D with signed binomial coefficients in each row.
/// @param coefficient_count Number K of coefficients, strictly greater than penalty_order.
/// @param penalty_order Positive order d; second differences have rows (1,-2,1).
/// @return Owned (K-d) by K matrix; D annihilates coefficient sequences polynomial of degree < d.
/// @throws std::invalid_argument For d=0 or K<=d.
/// @throws std::overflow_error If difference weights are outside finite double range.
/// @par Example
/// @code
/// const auto D = dm::fda::pspline_difference_matrix(3,2);
/// std::print("{} {} {}\n", D(0,0),D(0,1),D(0,2)); // 1 -2 1
/// @endcode
/// @par References
/// - P. H. C. Eilers and B. D. Marx (1996). Flexible smoothing with B-splines and penalties.
///   https://doi.org/10.1214/ss/1038425655
/// @see pspline_penalty_matrix()
linalg::DenseMatrix<double> pspline_difference_matrix(std::size_t coefficient_count,
                                                     std::size_t penalty_order);

/// @brief Positive semidefinite coefficient penalty P = D transpose D.
/// @param coefficient_count Number K of coefficients, strictly greater than penalty_order.
/// @param penalty_order Positive difference order d.
/// @return Owned K by K symmetric penalty matrix of rank K-d in exact arithmetic.
/// @throws std::invalid_argument For d=0 or K<=d.
/// @throws std::overflow_error If difference weights or penalty entries overflow.
/// @par Example
/// @code
/// const auto P = dm::fda::pspline_penalty_matrix(3,2);
/// std::print("{}\n", P(1,1)); // 4
/// @endcode
/// @par References
/// See pspline_difference_matrix().
/// @see pspline_difference_matrix()
linalg::DenseMatrix<double> pspline_penalty_matrix(std::size_t coefficient_count,
                                                   std::size_t penalty_order);

/// @brief Fits summed squared error plus lambda times squared differences on an explicit basis.
///
/// Solves dense penalized normal equations with Cholesky, using sparse local basis
/// evaluation to form the Gram matrix and leverages. O(n(p+1)^2 + K^3) work, O(n(p+1)+K^2)
/// storage, with degree p, basis size K and n observations. Positive lambda identifies
/// the fit only if the design identifies the penalty nullspace. No data are dropped.
/// @param x Nonempty finite coordinates within the supplied basis domain.
/// @param y Equally sized finite responses in matching order.
/// @param basis Owned copy retained in the result; all knots must be finite.
/// @param opts Finite nonnegative lambda and positive order less than basis size; defaults {1,2}.
/// @return Owned coefficients, predictions, residuals, leverage and scalar diagnostics.
/// @throws std::invalid_argument For invalid inputs, domain, knots or options.
/// @throws std::runtime_error If the penalized system is not numerically positive definite or overflows.
/// @throws std::overflow_error If the difference penalty overflows.
/// @par Example
/// @code
/// const auto basis = dm::fda::BSplineBasis::open_uniform(1,3,0,2);
/// const auto fit = dm::fda::fit_pspline({0,1,2},{0,1,0},basis,{1.0,2});
/// std::print("{:.6f}\n", fit.objective); // 0.571429
/// @endcode
/// @par References
/// - P. H. C. Eilers and B. D. Marx (1996). Flexible smoothing with B-splines and penalties.
///   https://doi.org/10.1214/ss/1038425655
/// @see PSplineRegression, PSplineFitResult
PSplineFitResult fit_pspline(const std::vector<double>& x,
                             const std::vector<double>& y,
                             const BSplineBasis&        basis,
                             PSplineOptions             opts = {});

/// @brief Fits P-splines with repeated boundary knots on the observed x range.
///
/// Unlike PSplineRegression, open_uniform() clamps the boundary knots. The difference
/// penalty nullspace is polynomial in coefficient index, which need not represent a
/// polynomial in x near these repeated knots. Work and other contracts as in the basis overload.
/// @param x Nonempty finite coordinates spanning a positive-width interval.
/// @param y Finite responses matching x.
/// @param degree Polynomial degree p, nonnegative.
/// @param basis_size Number K of basis functions, at least p+1 and greater than penalty order.
/// @param opts Penalty settings, default {1,2}.
/// @return Owned fit with an open-uniform basis spanning min(x) to max(x).
/// @throws std::invalid_argument For invalid data, basis size, degree or options.
/// @throws std::runtime_error For a singular/non-finite numerical system.
/// @throws std::overflow_error If the difference penalty overflows.
/// @par Example
/// See PSplineFitResult's class example.
/// @see fit_pspline(), BSplineBasis::open_uniform(), PSplineRegression
PSplineFitResult fit_pspline(const std::vector<double>& x,
                             const std::vector<double>& y,
                             std::size_t                degree,
                             std::size_t                basis_size,
                             PSplineOptions             opts = {});

} // namespace datamunge::fda
