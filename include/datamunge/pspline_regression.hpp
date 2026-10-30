#pragma once

/// @file pspline_regression.hpp
/// @brief P-spline regression on R-style vectors, bounded GCV selection and diagnostic tables.

#include <datamunge/fda/pspline.hpp>
#include <datamunge/vector.hpp>
#include <datamunge/dstruct/dataframe.hpp>

#include <cstddef>
#include <utility>

namespace datamunge {

/// @brief A P-spline regression curve y ≈ Σ_j β_j B_j(x) (Eilers & Marx). `dm::PSR` for short.
///
/// Minimizes ||y - Bβ||² + λ ||D_d β||², where D_d is the order-d difference matrix on the
/// coefficients (d = 2 by default). Uniform knots continue past both ends of the data
/// without repeated boundary knots. When d <= degree+1, polynomials of degree < d
/// go unpenalized: λ → ∞ gives the least-squares line for d = 2. A rich basis can
/// include functions with no data under them if the data identify the penalty nullspace.
/// Without an explicit λ, it is chosen
/// by minimizing generalized cross-validation, GCV(λ) = n·RSS / (n − edf)².
///
/// The polynomial-nullspace claim requires uniform spacing; it need not hold for irregular
/// supplied knots. A positive lambda identifies a rich basis only when the design identifies
/// the penalty nullspace. No values are dropped. The class owns its fit and permits concurrent
/// const reads. Sparse assembly and dense Cholesky cost O(n(p+1)^2+K^3); GCV repeats
/// this work. Very large lambda can make the equations numerically singular.
///
/// @par Example
/// @code
/// const dm::PSR fit({0,1,2},{0,1,0},3,1.0,1,2);
/// std::print("{:.6f} {:.6f}\n",fit.predict({1})[0],fit.effective_df()); // 0.428571 2.142857
/// @endcode
/// @par References
/// - P. H. C. Eilers and B. D. Marx (1996). Flexible smoothing with B-splines and penalties.
///   https://doi.org/10.1214/ss/1038425655
/// @see fda::fit_pspline(), smoothing_path(), summary()
class PSplineRegression {
 public:
  /// @brief @p n_basis cubic B-splines with equally spaced knots over the range of @p x; λ chosen
  /// by GCV.
  /// Searches log10(lambda) on [-8,8] in steps of 0.5, then uses golden-section refinement
  /// around the best grid point. Singular candidates score infinity. This bounded refinement
  /// does not guarantee a global continuous minimum. Penalty order is two.
  /// @param x Nonempty finite coordinates spanning a positive-width interval.
  /// @param y Finite responses matching x.
  /// @param n_basis Basis count K, at least four.
  /// @throws std::invalid_argument For invalid data or basis size.
  /// @throws std::runtime_error If all candidate systems fail numerically.
  /// @par Example
  /// @code
  /// const dm::Vector<> x=dm::seql(0,1,30);
  /// const dm::PSR fit(x,dm::sin(6*x),10);
  /// std::print("{}\n",fit.lambda()>0); // true
  /// @endcode
  /// @par References
  /// - G. H. Golub, M. Heath and G. Wahba (1979). Generalized cross-validation as a
  ///   method for choosing a good ridge parameter. https://doi.org/10.1080/00401706.1979.10489751
  /// @see gcv(), smoothing_path()
  PSplineRegression(const Vector<>& x, const Vector<>& y, std::size_t n_basis);

  /// @brief @p n_basis B-splines of the given @p degree with equally spaced knots over the range of
  /// @p x, penalized with smoothing parameter @p lambda on differences of order @p penalty_order.
  /// @param x Nonempty finite coordinates spanning a positive-width interval.
  /// @param y Finite responses matching x.
  /// @param n_basis Basis count at least degree+1 and greater than penalty_order.
  /// @param lambda Finite nonnegative multiplier of summed squared coefficient differences.
  /// @param degree Polynomial degree, default three.
  /// @param penalty_order Positive difference order, default two.
  /// @throws std::invalid_argument For invalid data/options or non-finite extended knots.
  /// @throws std::runtime_error If the penalized system is singular or non-finite.
  /// @throws std::overflow_error If difference weights or the penalty overflow.
  /// @par Example
  /// See the class example for a degree-one, order-two fit.
  /// @see PSplineRegression, fda::fit_pspline()
  PSplineRegression(const Vector<>& x, const Vector<>& y, std::size_t n_basis, double lambda,
                    std::size_t degree = 3, std::size_t penalty_order = 2);

  /// @brief Cubic basis on the given knot locations; boundary knots are the smallest/largest of
  /// @p knots and @p x. The difference penalty assumes roughly even spacing. λ chosen by GCV.
  /// Uses the same bounded GCV search as the basis-count constructor.
  /// @param x Nonempty finite coordinates.
  /// @param y Finite matching responses.
  /// @param knots Nonempty finite breakpoints, sorted/deduplicated and enlarged to contain x.
  /// @throws std::invalid_argument For invalid data/knots or zero-width combined domain.
  /// @throws std::runtime_error If every candidate system fails numerically.
  /// @par Example
  /// @code
  /// const dm::Vector<> x=dm::seql(0,1,30);
  /// const dm::PSR fit(x,dm::sin(6*x),dm::seql(0,1,8));
  /// std::print("{}\n",fit.basis_size()); // 10
  /// @endcode
  /// @see PSplineRegression, gcv(), knots()
  PSplineRegression(const Vector<>& x, const Vector<>& y, const Vector<>& knots);

  /// @brief Basis on the given knot locations with a fixed smoothing parameter @p lambda.
  /// Outer spacings are continued past the domain, without repeated boundary knots.
  /// The penalty acts on coefficient index and does not correct for unequal knot spacing.
  /// @param x Nonempty finite coordinates.
  /// @param y Finite matching responses.
  /// @param knots Nonempty finite breakpoints, sorted/deduplicated and enlarged to contain x.
  /// @param lambda Finite nonnegative smoothing multiplier.
  /// @param degree Polynomial degree, default three.
  /// @param penalty_order Positive difference order smaller than the resulting basis, default two.
  /// @throws std::invalid_argument For invalid data, options, knots or zero-width combined domain.
  /// @throws std::runtime_error If the system is singular or non-finite.
  /// @throws std::overflow_error If difference weights or the penalty overflow.
  /// @par Example
  /// @code
  /// const dm::PSR fit({0,1,2},{0,1,0},dm::Vector<>{0,1,2},1.0,1,2);
  /// std::print("{:.6f}\n",fit.predict({1})[0]); // 0.428571
  /// @endcode
  /// @see PSplineRegression, fda::pspline_difference_matrix()
  PSplineRegression(const Vector<>& x, const Vector<>& y, const Vector<>& knots, double lambda,
                    std::size_t degree = 3, std::size_t penalty_order = 2);

  /// @brief Evaluates the fitted curve at @p x. Points outside the fitted domain take the value
  /// at the nearest boundary, i.e. constant extrapolation.
  /// @param x Finite evaluation coordinates; empty is accepted.
  /// @return Owned predictions in input order, after clamping x to domain().
  /// @throws std::invalid_argument For non-finite coordinates.
  /// @par Example
  /// @code
  /// const dm::PSR fit({0,1,2},{0,1,0},3,1.0,1,2);
  /// std::print("{:.6f}\n",fit.predict({-1})[0]); // 0.285714
  /// @endcode
  /// @see basis_matrix(), domain()
  Vector<> predict(const Vector<>& x) const;

  /// @brief Fitted coefficients in basis order.
  /// @return Owned K-vector.
  Vector<>      coefficients() const { return fit_.coefficients; }
  /// @brief Predictions at the training coordinates.
  /// @return Owned n-vector in training order.
  Vector<>      fitted() const { return fit_.fitted; }
  /// @brief Observed minus fitted training responses.
  /// @return Owned n-vector in training order.
  Vector<>      residuals() const { return fit_.residuals; }
  /// @brief In-sample 1-RSS/TSS, with the constant-response convention of PSplineFitResult.
  /// @return Scalar R squared.
  double      r_squared() const noexcept { return fit_.r_squared; }
  /// @brief The smoothing parameter λ used for the fit (the GCV minimizer when not given).
  /// @return Finite nonnegative multiplier used in the fit.
  double      lambda() const noexcept { return fit_.lambda; }
  /// @brief Order of adjacent coefficient differences.
  /// @return Positive difference order, independent of spline degree.
  std::size_t penalty_order() const noexcept { return fit_.penalty_order; }
  /// @brief Effective degrees of freedom, trace of the hat matrix.
  /// @return Fixed-lambda smoother trace, the sum of training leverages.
  double      effective_df() const noexcept { return fit_.effective_degrees_of_freedom; }
  /// @brief Generalized cross-validation score n·RSS / (n − edf)² of the fit.
  /// @return Fixed-lambda score in response-squared units; infinity when n-edf is numerically zero.
  /// @see fda::PSplineFitResult::gcv(), PSplineRegression
  double      gcv() const noexcept;
  /// @brief Number of basis functions, including boundary functions.
  /// @return Basis dimension K.
  std::size_t basis_size() const noexcept { return fit_.basis.basis_size(); }
  /// @brief Polynomial degree of each spline segment.
  /// @return Degree p, independent of penalty order.
  std::size_t degree() const noexcept { return fit_.basis.degree(); }
  /// @brief The full knot sequence, including the `degree` extra knots past each end of the data.
  /// @return Owned sequence of K+degree+1 knots.
  Vector<> knots() const { return fit_.basis.knots(); }
  /// @brief The [min, max] interval the basis covers.
  /// @return Closed interval endpoints (minimum,maximum).
  std::pair<double, double> domain() const noexcept { return fit_.basis.domain(); }

  /// @brief Dense basis matrix with the same clamping convention as predict().
  /// @param x Finite evaluation coordinates; empty and out-of-domain points are accepted.
  /// @return Owned x.size() by K matrix; at most degree+1 nonzeros per row.
  /// @throws std::invalid_argument For non-finite coordinates.
  /// @par Example
  /// @code
  /// const dm::PSR fit({0,1,2},{0,1,0},3,1.0,1,2);
  /// std::print("{}\n",fit.basis_matrix({0,1,2})(1,1)); // 1
  /// @endcode
  /// @see predict(), fda::pspline_difference_matrix()
  linalg::DenseMatrix<double> basis_matrix(const Vector<>& x) const;

  /// @brief Fitted order-d coefficient differences, whose squared norm is roughness.
  /// @return Owned K-d vector in left-to-right difference order.
  /// @par Example
  /// @code
  /// const dm::PSR fit({0,1,2},{0,1,0},3,1.0,1,2);
  /// std::print("{:.6f}\n",fit.coefficient_differences()[0]); // -0.285714
  /// @endcode
  /// @see summary(), fda::pspline_difference_matrix()
  Vector<> coefficient_differences() const;

  /// @brief One-row smoothing diagnostics table ready for DataFrame::to_md().
  /// @return Owned numeric columns n, basis, degree, order, lambda, edf, rss, roughness,
  /// objective, r_squared, gcv and loo_cv. Deletion scores hold knots and lambda fixed.
  /// @par Example
  /// @code
  /// const dm::PSR fit({0,1,2},{0,1,0},3,1.0,1,2);
  /// std::print("{:.1f}\n",fit.summary().numeric_column("loo_cv").values()[0]); // 3.0
  /// @endcode
  /// @see observation_details(), smoothing_path()
  dstruct::DataFrame summary() const;

  /// @brief Per-observation fit, leverage and fixed-lambda deletion diagnostics.
  /// @return Owned numeric columns observation (one-based), response, fitted, residual,
  /// leverage, loo_residual. Undefined deletion residuals are NaN; x is not stored.
  /// @par Example
  /// @code
  /// const dm::PSR fit({0,1,2},{0,1,0},3,1.0,1,2);
  /// std::print("{:.1f}\n",fit.observation_details().numeric_column("loo_residual").values()[1]); // 1.0
  /// @endcode
  /// @see summary(), fda::PSplineFitResult::loo_residuals()
  dstruct::DataFrame observation_details() const;

  /// @brief Summarizes a lambda grid on one uniform extended basis, without automatic selection.
  ///
  /// Singular candidates are reported as exceptions, not omitted. Work is the grid length
  /// times the cost of a fit. Input is not mutated; rows preserve the supplied lambda order.
  /// @param x Nonempty finite coordinates spanning a positive-width interval.
  /// @param y Finite matching responses.
  /// @param n_basis Basis count at least degree+1 and greater than penalty_order.
  /// @param lambdas Nonempty finite nonnegative multiplier grid.
  /// @param degree Spline degree, default three.
  /// @param penalty_order Positive difference order, default two.
  /// @return Owned numeric columns lambda, edf, rss, roughness, objective, gcv, loo_cv.
  /// @throws std::invalid_argument For invalid data/options or an empty lambda grid.
  /// @throws std::runtime_error If any system is singular or non-finite.
  /// @throws std::overflow_error If difference weights or the penalty overflow.
  /// @par Example
  /// @code
  /// const auto path=dm::PSR::smoothing_path({0,1,2},{0,1,0},3,{0.1,1,10},1,2);
  /// std::print("{:.6f}\n",path.numeric_column("edf").values()[1]); // 2.142857
  /// @endcode
  /// @see summary(), PSplineRegression
  static dstruct::DataFrame smoothing_path(const Vector<>& x, const Vector<>& y,
      std::size_t n_basis, const Vector<>& lambdas, std::size_t degree=3,
      std::size_t penalty_order=2);

 private:
  fda::PSplineFitResult fit_; ///< Owned basis, coefficients, penalty settings and training diagnostics.
};

/// @brief Short name for PSplineRegression; constructors and contracts are identical.
/// @par Example
/// See PSplineRegression's class example.
/// @see PSplineRegression
using PSR = PSplineRegression;

} // namespace datamunge
