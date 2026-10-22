#include <datamunge/pspline_regression.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge {

namespace {

constexpr const char* kWho = "PSplineRegression";

void check_xy(const Vector& x, const Vector& y) {
  if (x.size() != y.size()) {
    throw std::invalid_argument(std::string(kWho) + ": x and y must have the same length");
  }
  if (x.empty()) {
    throw std::invalid_argument(std::string(kWho) + ": inputs must not be empty");
  }
}

// Eilers & Marx's knot layout: the breakpoints, plus `degree` more on each side continuing the
// first/last spacing (rather than the clamped, repeated boundary knots BSplineRegression uses).
// With evenly spaced breakpoints the coefficients of a polynomial of degree < penalty order are
// then themselves such a polynomial in the index, so the difference penalty leaves it untouched.
fda::BSplineBasis extended_basis(std::vector<double> breaks, std::size_t degree) {
  const double h_lo = breaks[1] - breaks[0];
  const double h_hi = breaks[breaks.size() - 1] - breaks[breaks.size() - 2];
  std::vector<double> full;
  full.reserve(breaks.size() + 2 * degree);
  for (std::size_t k = degree; k > 0; --k) full.push_back(breaks.front() - k * h_lo);
  full.insert(full.end(), breaks.begin(), breaks.end());
  for (std::size_t k = 1; k <= degree; ++k) full.push_back(breaks.back() + k * h_hi);
  return fda::BSplineBasis(degree, std::move(full));
}

fda::BSplineBasis uniform_basis(const Vector& x, const Vector& y, std::size_t n_basis,
                                std::size_t degree) {
  check_xy(x, y);
  if (n_basis < degree + 1) {
    throw std::invalid_argument(std::string(kWho) + ": need at least degree + 1 basis functions");
  }
  const auto [lo, hi] = std::minmax_element(x.begin(), x.end());
  if (!(*lo < *hi)) throw std::invalid_argument(std::string(kWho) + ": x spans a single point");
  const std::size_t segments = n_basis - degree;
  std::vector<double> breaks(segments + 1);
  for (std::size_t i = 0; i <= segments; ++i) {
    breaks[i] = *lo + (*hi - *lo) * static_cast<double>(i) / static_cast<double>(segments);
  }
  breaks.back() = *hi;
  return extended_basis(std::move(breaks), degree);
}

fda::BSplineBasis knot_basis(const Vector& x, const Vector& y, const Vector& knots,
                             std::size_t degree) {
  check_xy(x, y);
  if (knots.empty()) throw std::invalid_argument(std::string(kWho) + ": knots must not be empty");
  std::vector<double> breaks = knots.values();
  std::sort(breaks.begin(), breaks.end());
  breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
  const auto [xlo, xhi] = std::minmax_element(x.begin(), x.end());
  const double lo = std::min(breaks.front(), *xlo);
  const double hi = std::max(breaks.back(), *xhi);
  if (!(lo < hi)) {
    throw std::invalid_argument(std::string(kWho) + ": x and knots span a single point");
  }
  std::vector<double> inner{lo};
  for (double k : breaks) {
    if (k > lo && k < hi) inner.push_back(k);
  }
  inner.push_back(hi);
  return extended_basis(std::move(inner), degree);
}

double gcv_score(const fda::PSplineFitResult& fit) {
  const double n = static_cast<double>(fit.observations);
  const double dof = n - fit.effective_degrees_of_freedom;
  if (!(dof > 0.0)) return std::numeric_limits<double>::infinity();
  return n * fit.residual_sum_of_squares / (dof * dof);
}

fda::PSplineFitResult fit_fixed(const Vector& x, const Vector& y, const fda::BSplineBasis& basis,
                                double lambda, std::size_t penalty_order) {
  if (!(lambda >= 0.0)) {
    throw std::invalid_argument(std::string(kWho) + ": lambda must be nonnegative");
  }
  fda::PSplineOptions opts;
  opts.lambda = lambda;
  opts.penalty_order = penalty_order;
  try {
    return fda::fit_pspline(x.values(), y.values(), basis, opts);
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(kWho) + ": " + e.what());
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(std::string(kWho) + ": penalized system is singular (" + e.what() +
                             "); increase lambda");
  }
}

// Minimizes GCV over log10(lambda): a coarse grid over [1e-8, 1e8], then golden-section search
// on the bracket around the best grid point.
fda::PSplineFitResult fit_gcv(const Vector& x, const Vector& y, const fda::BSplineBasis& basis) {
  constexpr std::size_t kPenaltyOrder = 2;
  constexpr double kLo = -8.0, kHi = 8.0, kStep = 0.5;

  auto fit_at = [&](double log_lambda) -> fda::PSplineFitResult {
    return fit_fixed(x, y, basis, std::pow(10.0, log_lambda), kPenaltyOrder);
  };
  // A tiny lambda can leave the system numerically singular when basis functions lack data;
  // score those as +inf rather than failing the whole search.
  auto score_at = [&](double log_lambda) {
    try {
      return gcv_score(fit_at(log_lambda));
    } catch (const std::runtime_error&) {
      return std::numeric_limits<double>::infinity();
    }
  };

  double best = kLo, best_score = std::numeric_limits<double>::infinity();
  for (double l = kLo; l <= kHi + 1e-12; l += kStep) {
    const double s = score_at(l);
    if (s < best_score) best = l, best_score = s;
  }
  if (!std::isfinite(best_score)) return fit_at(kHi);  // smoothest fit; rethrows if even that fails

  const double phi = (std::sqrt(5.0) - 1.0) / 2.0;
  double a = std::max(kLo, best - kStep), b = std::min(kHi, best + kStep);
  double c = b - phi * (b - a), d = a + phi * (b - a);
  double fc = score_at(c), fd = score_at(d);
  for (int it = 0; it < 30 && b - a > 1e-4; ++it) {
    if (fc < fd) {
      b = d, d = c, fd = fc;
      c = b - phi * (b - a), fc = score_at(c);
    } else {
      a = c, c = d, fc = fd;
      d = a + phi * (b - a), fd = score_at(d);
    }
  }
  const double refined = fc < fd ? c : d;
  return fit_at(std::min(fc, fd) < best_score ? refined : best);
}

} // namespace

PSplineRegression::PSplineRegression(const Vector& x, const Vector& y, std::size_t n_basis)
    : fit_(fit_gcv(x, y, uniform_basis(x, y, n_basis, 3))) {}

PSplineRegression::PSplineRegression(const Vector& x, const Vector& y, std::size_t n_basis,
                                     double lambda, std::size_t degree, std::size_t penalty_order)
    : fit_(fit_fixed(x, y, uniform_basis(x, y, n_basis, degree), lambda,
                     penalty_order)) {}

PSplineRegression::PSplineRegression(const Vector& x, const Vector& y, const Vector& knots)
    : fit_(fit_gcv(x, y, knot_basis(x, y, knots, 3))) {}

PSplineRegression::PSplineRegression(const Vector& x, const Vector& y, const Vector& knots,
                                     double lambda, std::size_t degree, std::size_t penalty_order)
    : fit_(fit_fixed(x, y, knot_basis(x, y, knots, degree), lambda,
                     penalty_order)) {}

double PSplineRegression::gcv() const noexcept { return gcv_score(fit_); }

Vector PSplineRegression::predict(const Vector& x) const {
  const auto [lo, hi] = domain();
  Vector clamped = x;
  for (double& v : clamped) v = std::clamp(v, lo, hi);
  return fit_.predict(clamped.values());
}

} // namespace datamunge
