#pragma once

/// @file pspline_regression.hpp
/// @brief Penalized B-spline (P-spline) regression on R-style dm::Vector data.
///
/// @code
/// namespace dm = datamunge;
/// dm::PSR fit(x, y, 100);                       // 100 basis functions, lambda chosen by GCV
/// dm::PSR fixed(x, y, 100, 0.5);                // or a fixed smoothing parameter
/// dm::PSR by_knots(x, y, dm::seql(0, 1, 41));   // or explicit knot locations
/// dm::Vector yhat = fit.predict(dm::seql(0, 1, 200));
/// @endcode

#include <datamunge/fda/pspline.hpp>
#include <datamunge/vector.hpp>

#include <cstddef>
#include <utility>

namespace datamunge {

/// @brief A P-spline regression curve y ≈ Σ_j β_j B_j(x) (Eilers & Marx). `dm::PSR` for short.
///
/// Minimizes ||y - Bβ||² + λ ||D_d β||², where D_d is the order-d difference matrix on the
/// coefficients (d = 2 by default). Knots are evenly spaced and continue past both ends of the data
/// (no repeated boundary knots), so polynomials of degree < d go unpenalized: λ → ∞ gives the
/// least-squares line for d = 2. Unlike BSplineRegression, the penalty keeps the fit well posed
/// with a rich basis -- even when some basis functions have no data under them -- so the usual
/// recipe is "many basis functions, let λ control smoothness". Without an explicit λ, it is chosen
/// by minimizing generalized cross-validation, GCV(λ) = n·RSS / (n − edf)².
///
/// Throws std::invalid_argument for mismatched/empty inputs, a negative λ, too few basis functions
/// for the degree, or a basis no larger than the penalty order.
class PSplineRegression {
 public:
  /// @brief @p n_basis cubic B-splines with equally spaced knots over the range of @p x; λ chosen
  /// by GCV.
  PSplineRegression(const Vector& x, const Vector& y, std::size_t n_basis);

  /// @brief @p n_basis B-splines of the given @p degree with equally spaced knots over the range of
  /// @p x, penalized with smoothing parameter @p lambda on differences of order @p penalty_order.
  PSplineRegression(const Vector& x, const Vector& y, std::size_t n_basis, double lambda,
                    std::size_t degree = 3, std::size_t penalty_order = 2);

  /// @brief Cubic basis on the given knot locations; boundary knots are the smallest/largest of
  /// @p knots and @p x. The difference penalty assumes roughly even spacing. λ chosen by GCV.
  PSplineRegression(const Vector& x, const Vector& y, const Vector& knots);

  /// @brief Basis on the given knot locations with a fixed smoothing parameter @p lambda.
  PSplineRegression(const Vector& x, const Vector& y, const Vector& knots, double lambda,
                    std::size_t degree = 3, std::size_t penalty_order = 2);

  /// @brief Evaluates the fitted curve at @p x. Points outside the fitted domain take the value
  /// at the nearest boundary, i.e. constant extrapolation.
  Vector predict(const Vector& x) const;

  Vector      coefficients() const { return fit_.coefficients; }
  Vector      fitted() const { return fit_.fitted; }
  Vector      residuals() const { return fit_.residuals; }
  double      r_squared() const noexcept { return fit_.r_squared; }
  /// @brief The smoothing parameter λ used for the fit (the GCV minimizer when not given).
  double      lambda() const noexcept { return fit_.lambda; }
  std::size_t penalty_order() const noexcept { return fit_.penalty_order; }
  /// @brief Effective degrees of freedom, trace of the hat matrix.
  double      effective_df() const noexcept { return fit_.effective_degrees_of_freedom; }
  /// @brief Generalized cross-validation score n·RSS / (n − edf)² of the fit.
  double      gcv() const noexcept;
  std::size_t basis_size() const noexcept { return fit_.basis.basis_size(); }
  std::size_t degree() const noexcept { return fit_.basis.degree(); }
  /// @brief The full knot sequence, including the `degree` extra knots past each end of the data.
  Vector knots() const { return fit_.basis.knots(); }
  /// @brief The [min, max] interval the basis covers.
  std::pair<double, double> domain() const noexcept { return fit_.basis.domain(); }

 private:
  fda::PSplineFitResult fit_;
};

using PSR = PSplineRegression;

} // namespace datamunge
