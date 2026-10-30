#include <datamunge/bspline_regression.hpp>

#include <algorithm>
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

} // namespace datamunge
