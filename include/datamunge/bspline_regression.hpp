#pragma once

/// @file bspline_regression.hpp
/// @brief Least-squares B-spline regression on R-style dm::Vector<> data.
///
/// @code
/// namespace dm = datamunge;
/// dm::BSR fit(x, y, 20);                        // 20 equally spaced basis functions
/// dm::BSR by_knots(x, y, dm::seql(0, 1, 11));    // or explicit knot locations
/// dm::Vector<> yhat = fit.predict(dm::seql(0, 1, 200));
/// @endcode

#include <datamunge/fda/pspline.hpp>
#include <datamunge/vector.hpp>

#include <cstddef>
#include <utility>

namespace datamunge {

/// @brief A least-squares B-spline regression curve y ≈ Σ_j β_j B_j(x). `dm::BSR` for short.
///
/// The fit is ordinary least squares (no roughness penalty). Throws std::invalid_argument for
/// mismatched/empty inputs or too few basis functions for the degree, and std::runtime_error when
/// the design is rank deficient (some basis function has too little data under it; use fewer
/// basis functions / knots).
class BSplineRegression {
 public:
  /// @brief Fixed-width basis: @p n_basis B-splines of the given @p degree (cubic by default)
  /// whose knots are equally spaced over the range of @p x.
  BSplineRegression(const Vector<>& x, const Vector<>& y, std::size_t n_basis, std::size_t degree = 3);

  /// @brief Basis built on the given knot locations (breakpoints), like `knots=` in R's
  /// `splines::bs()`. Boundary knots are the smallest/largest of @p knots and @p x; the rest are
  /// interior knots. Gives (number of interior knots) + degree + 1 basis functions.
  BSplineRegression(const Vector<>& x, const Vector<>& y, const Vector<>& knots, std::size_t degree = 3);

  /// @brief Evaluates the fitted curve at @p x. Points outside the fitted domain take the value
  /// at the nearest boundary, i.e. constant extrapolation.
  Vector<> predict(const Vector<>& x) const;

  Vector<>      coefficients() const { return fit_.coefficients; }
  Vector<>      fitted() const { return fit_.fitted; }
  Vector<>      residuals() const { return fit_.residuals; }
  double      r_squared() const noexcept { return fit_.r_squared; }
  std::size_t basis_size() const noexcept { return fit_.basis.basis_size(); }
  std::size_t degree() const noexcept { return fit_.basis.degree(); }
  /// @brief The full knot sequence (boundary knots repeated degree + 1 times).
  Vector<> knots() const { return fit_.basis.knots(); }
  /// @brief The [min, max] interval the basis covers.
  std::pair<double, double> domain() const noexcept { return fit_.basis.domain(); }

 private:
  fda::PSplineFitResult fit_;
};

using BSR = BSplineRegression;

} // namespace datamunge
