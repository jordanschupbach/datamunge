#pragma once

/// @file bspline_regression.hpp
/// @brief Reusable B-spline designs and least-squares regression on R-style dm::Vector<> data.
///
/// @code
/// namespace dm = datamunge;
/// dm::BSR fit(x, y, 20);                        // 20 equally spaced basis functions
/// dm::BSR by_knots(x, y, dm::seql(0, 1, 11));    // or explicit knot locations
/// dm::Vector<> yhat = fit.predict(dm::seql(0, 1, 200));
/// @endcode

#include <cstddef>
#include <datamunge/fda/pspline.hpp>
#include <datamunge/linalg/matrix.hpp>
#include <datamunge/vector.hpp>
#include <utility>

namespace datamunge {

/// @brief B-spline design factory, analogous to R's splines::bs().
/// Owns a clamped basis on fixed boundaries, reused for training and prediction.
/// By default the first basis column is omitted, so a formula's intercept does
/// not duplicate the partition of unity. No fitting or smoothing is performed.
/// Const evaluations are independent and safe to run concurrently.
/// @par Example
/// @code
/// const dm::Bspline basis(0, 1, {0, 0.5, 1});
/// const auto design = basis.eval({0, 0.5, 1});
/// std::print("{} {}\n", design.rows(), design.cols()); // 3 4
/// @endcode
/// @par References
/// - C. de Boor (1972). "On calculating with B-splines". Journal of Approximation
///   Theory 6(1), 50-62. doi:10.1016/0021-9045(72)90080-9.
/// @see bs(), fda::BSplineBasis, stats::LM
class Bspline {
 public:
  /// @brief Build a clamped basis from strictly increasing knot breakpoints.
  /// @param lower Finite left boundary, strictly below upper.
  /// @param upper Finite right boundary.
  /// @param knots Finite increasing breakpoints inside [lower, upper]; boundary
  /// entries are optional and ignored when building interior knots. May be empty.
  /// @param degree Polynomial degree, default 3 (cubic).
  /// @param intercept Include the first basis column, default false; use a
  /// no-intercept formula when true. Column count is interior knots + degree
  /// + (intercept ? 1 : 0).
  /// @throws std::invalid_argument For invalid boundaries/knots or overflowing degree.
  /// @par Example
  /// @code
  /// const dm::Bspline full(0, 1, {}, 1, true);
  /// std::print("{:.1f}\n", full.eval({0.5})(0, 0)); // 0.5
  /// @endcode
  Bspline(
      double                     lower,
      double                     upper,
      const std::vector<double>& knots,
      std::size_t                degree    = 3,
      bool                       intercept = false);

  /// @brief Evaluate the same basis on training or prediction coordinates.
  /// @param x Finite coordinates in the closed fitted domain; an empty vector is valid.
  /// @return Owned dense double design, one row per x and deterministic colnames().
  /// @throws std::invalid_argument For nonfinite or out-of-domain coordinates.
  /// @par Example
  /// @code
  /// const dm::Bspline basis(0, 1, {}, 1);
  /// std::print("{:.2f}\n", basis.eval({0.25})(0, 0)); // 0.25
  /// @endcode
  Mat<> eval(const std::vector<double>& x) const;

  /// @brief Evaluate this fixed basis directly into compressed sparse rows.
  /// Uses local support without allocating a dense design; omitted first-column
  /// and boundary conventions are identical to eval().
  /// @param x Finite coordinates in the closed fitted domain; may be empty.
  /// @return Owned CSR design with at most degree + 1 stored entries per row.
  /// @throws std::invalid_argument For nonfinite or out-of-domain coordinates.
  /// @par Example
  /// @code
  /// const auto design = dm::bs(0, 1, {}, 1).speval({0, 0.5, 1});
  /// std::print("{} {} {}\n", design.rows(), design.cols(), design.nnz()); // 3 1 2
  /// @endcode
  /// @see eval(), LM
  SpMat<> speval(const std::vector<double>& x) const;

 private:
  fda::BSplineBasis basis_;     ///< Owned canonical basis, including repeated boundary knots.
  bool              intercept_; ///< Whether to keep the first basis function.
};

/// @brief Construct an owned B-spline basis for repeated evaluation with eval().
/// Free-function spelling of Bspline: construction fixes the knots and boundaries
/// without evaluating a design matrix or fitting a model. The returned value owns
/// its basis; the input knot vector may be modified or destroyed after this call.
/// @param lower Finite left boundary, strictly below upper.
/// @param upper Finite right boundary.
/// @param knots Finite, strictly increasing breakpoints in [lower, upper]. Boundary
/// entries are optional; an empty vector creates a basis without interior knots.
/// @param degree Polynomial degree, default 3 (cubic).
/// @param intercept Keep the first basis column, default false. Leave false with
/// an intercept formula such as `y ~ .`; use a no-intercept formula when true.
/// @return Owned Bspline with the same degree, intercept policy and validation
/// as its constructor. Call eval() on training and prediction coordinates.
/// @throws std::invalid_argument For invalid boundaries, knots or overflowing degree.
/// @par Example
/// @code
/// const auto basis = dm::bs(0, 1, dm::seql(0, 1, 3));
/// const auto design = basis.eval(dm::c(0, 0.5, 1));
/// std::print("{} {}\n", design.rows(), design.cols()); // 3 4
/// @endcode
/// @see Bspline, Bspline::eval()
inline Bspline bs(
    double                     lower,
    double                     upper,
    const std::vector<double>& knots,
    std::size_t                degree    = 3,
    bool                       intercept = false) {
  return Bspline(lower, upper, knots, degree, intercept);
}

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
  BSplineRegression(
      const Vector<>& x,
      const Vector<>& y,
      std::size_t     n_basis,
      std::size_t     degree = 3);

  /// @brief Basis built on the given knot locations (breakpoints), like `knots=` in R's
  /// `splines::bs()`. Boundary knots are the smallest/largest of @p knots and @p x; the rest are
  /// interior knots. Gives (number of interior knots) + degree + 1 basis functions.
  BSplineRegression(
      const Vector<>& x,
      const Vector<>& y,
      const Vector<>& knots,
      std::size_t     degree = 3);

  /// @brief Evaluates the fitted curve at @p x. Points outside the fitted domain take the value
  /// at the nearest boundary, i.e. constant extrapolation.
  Vector<> predict(const Vector<>& x) const;

  Vector<>    coefficients() const { return fit_.coefficients; }
  Vector<>    fitted() const { return fit_.fitted; }
  Vector<>    residuals() const { return fit_.residuals; }
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
