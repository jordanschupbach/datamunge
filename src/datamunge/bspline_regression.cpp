/// @file bspline_regression.cpp
/// @brief Reusable clamped B-spline designs and ordinary least-squares spline fits.
#include <datamunge/bspline_regression.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge {

namespace {

fda::PSplineFitResult fit_ols(const Vector<>& x, const Vector<>& y, const fda::BSplineBasis& basis) {
  fda::PSplineOptions ols;
  ols.lambda = 0.0;       // no roughness penalty: plain least squares
  ols.penalty_order = 1;  // unused at lambda = 0; the lowest order admits the smallest bases
  try {
    return fda::fit_pspline(x.values(), y.values(), basis, ols);
  } catch (const std::runtime_error& e) {
    throw std::runtime_error("BSplineRegression: least-squares system is singular with " +
                             std::to_string(basis.basis_size()) +
                             " basis functions (some have no data under them); use fewer (" +
                             e.what() + ")");
  }
}

void check_xy(const Vector<>& x, const Vector<>& y) {
  if (x.size() != y.size()) {
    throw std::invalid_argument("BSplineRegression: x and y must have the same length");
  }
  if (x.empty()) {
    throw std::invalid_argument("BSplineRegression: inputs must not be empty");
  }
}

fda::BSplineBasis uniform_basis(const Vector<>& x, const Vector<>& y, std::size_t n_basis,
                                std::size_t degree) {
  check_xy(x, y);
  const auto [lo, hi] = std::minmax_element(x.begin(), x.end());
  return fda::BSplineBasis::open_uniform(degree, n_basis, *lo, *hi);
}

fda::BSplineBasis knot_basis(const Vector<>& x, const Vector<>& y, const Vector<>& knots,
                             std::size_t degree) {
  check_xy(x, y);
  if (knots.empty()) {
    throw std::invalid_argument("BSplineRegression: knots must not be empty");
  }
  std::vector<double> breaks = knots.values();
  std::sort(breaks.begin(), breaks.end());
  breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
  const auto [xlo, xhi] = std::minmax_element(x.begin(), x.end());
  const double lo = std::min(breaks.front(), *xlo);
  const double hi = std::max(breaks.back(), *xhi);
  if (!(lo < hi)) {
    throw std::invalid_argument("BSplineRegression: x and knots span a single point");
  }

  std::vector<double> full(degree + 1, lo);
  for (double k : breaks) {
    if (k > lo && k < hi) full.push_back(k);
  }
  full.insert(full.end(), degree + 1, hi);
  return fda::BSplineBasis(degree, std::move(full));
}

} // namespace

BSplineRegression::BSplineRegression(const Vector<>& x, const Vector<>& y, std::size_t n_basis,
                                     std::size_t degree)
    : fit_(fit_ols(x, y, uniform_basis(x, y, n_basis, degree))) {}

BSplineRegression::BSplineRegression(const Vector<>& x, const Vector<>& y, const Vector<>& knots,
                                     std::size_t degree)
    : fit_(fit_ols(x, y, knot_basis(x, y, knots, degree))) {}

Vector<> BSplineRegression::predict(const Vector<>& x) const {
  const auto [lo, hi] = domain();
  Vector<> clamped = x;
  for (double& v : clamped) v = std::clamp(v, lo, hi);
  return fit_.predict(clamped.values());
}

Bspline::Bspline(
    double                     lower,
    double                     upper,
    const std::vector<double>& knots,
    std::size_t                degree,
    bool                       intercept)
    : intercept_(intercept) {
  if (!std::isfinite(lower) || !std::isfinite(upper) || !(lower < upper))
    throw std::invalid_argument("Bspline: require finite lower < upper");
  if (degree > std::numeric_limits<std::size_t>::max() / 2 - 1)
    throw std::invalid_argument("Bspline: degree is too large");
  std::vector<double> full(degree + 1, lower);
  for (std::size_t i = 0; i < knots.size(); ++i) {
    const auto knot = knots[i];
    if (!std::isfinite(knot) || knot < lower || knot > upper || (i && !(knots[i - 1] < knot)))
      throw std::invalid_argument(
          "Bspline: knots must be finite, increasing and within boundaries");
    if (knot > lower && knot < upper)
      full.push_back(knot);
  }
  full.insert(full.end(), degree + 1, upper);
  basis_ = fda::BSplineBasis(degree, std::move(full));
}

Mat<> Bspline::eval(const std::vector<double>& x) const {
  const auto [lower, upper] = basis_.domain();
  const std::size_t skip    = intercept_ ? 0 : 1;
  Mat<> out(x.size(), basis_.basis_size() - skip);
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!std::isfinite(x[i]) || x[i] < lower || x[i] > upper)
      throw std::invalid_argument(
          "Bspline: coordinates must be finite and within the fitted domain");
    const auto local = basis_.evaluate_point(x[i]);
    for (std::size_t k = 0; k < local.indices.size(); ++k)
      if (local.indices[k] >= skip)
        out.set(i, local.indices[k] - skip, local.values[k]);
  }
  return out;
}

SpMat<> Bspline::speval(const std::vector<double>& x) const {
  const auto [lower, upper] = basis_.domain();
  const std::size_t skip = intercept_ ? 0 : 1;
  linalg::SparseCSR<double> out;
  out.nrows = x.size();
  out.ncols = basis_.basis_size() - skip;
  out.row_ptr.push_back(0);
  for (double coordinate : x) {
    if (!std::isfinite(coordinate) || coordinate < lower || coordinate > upper)
      throw std::invalid_argument("Bspline: coordinates must be finite and within the fitted domain");
    const auto local = basis_.evaluate_point(coordinate);
    for (std::size_t k = 0; k < local.indices.size(); ++k)
      if (local.indices[k] >= skip) {
        out.col_idx.push_back(local.indices[k] - skip);
        out.values.push_back(local.values[k]);
      }
    out.row_ptr.push_back(out.values.size());
  }
  return SpMat<>(std::move(out));
}

} // namespace datamunge
