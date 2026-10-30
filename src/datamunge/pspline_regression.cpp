#include <datamunge/pspline_regression.hpp>

/// @file pspline_regression.cpp
/// @brief Extended-knot basis construction, bounded GCV search and P-spline diagnostic tables.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge {

/// @brief Internal basis construction, validation and GCV search.
namespace {

/// @brief Prefix identifying user-facing facade validation failures.
constexpr const char* kWho = "PSplineRegression";

/// @brief Validates equal-length, nonempty finite input vectors.
/// @param x Training coordinates.
/// @param y Paired responses.
/// @throws std::invalid_argument For mismatched/empty or non-finite data.
void check_xy(const Vector<>& x, const Vector<>& y) {
  if (x.size() != y.size()) {
    throw std::invalid_argument(std::string(kWho) + ": x and y must have the same length");
  }
  if (x.empty()) {
    throw std::invalid_argument(std::string(kWho) + ": inputs must not be empty");
  }
  for (std::size_t i=0;i<x.size();++i)
    if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
      throw std::invalid_argument(std::string(kWho)+": inputs must be finite");
}

// Eilers & Marx's knot layout: the breakpoints, plus `degree` more on each side continuing the
// first/last spacing (rather than the clamped, repeated boundary knots BSplineRegression uses).
// With evenly spaced breakpoints the coefficients of a polynomial of degree < penalty order are
// then themselves such a polynomial in the index, so the difference penalty leaves it untouched.
/// @brief Continues outer breakpoint spacings past both domain boundaries.
/// @param breaks At least two sorted distinct finite breakpoints.
/// @param degree Number of extra knots appended on each side.
/// @return Owned extended basis without repeated boundaries.
/// @throws std::invalid_argument For non-finite extended knots or invalid resulting basis.
fda::BSplineBasis extended_basis(std::vector<double> breaks, std::size_t degree) {
  const double h_lo = breaks[1] - breaks[0];
  const double h_hi = breaks[breaks.size() - 1] - breaks[breaks.size() - 2];
  std::vector<double> full;
  full.reserve(breaks.size() + 2 * degree);
  for (std::size_t k = degree; k > 0; --k) full.push_back(breaks.front() - k * h_lo);
  full.insert(full.end(), breaks.begin(), breaks.end());
  for (std::size_t k = 1; k <= degree; ++k) full.push_back(breaks.back() + k * h_hi);
  for (double knot : full)
    if (!std::isfinite(knot)) throw std::invalid_argument(std::string(kWho)+": extended knots must be finite");
  return fda::BSplineBasis(degree, std::move(full));
}

/// @brief Constructs exactly n_basis functions using equally spaced extended knots.
/// @param x Finite coordinates spanning a nonzero-width interval.
/// @param y Matching finite responses, validated with x.
/// @param n_basis Basis count, at least degree+1.
/// @param degree Segment polynomial degree.
/// @return Owned basis on min(x) to max(x).
/// @throws std::invalid_argument For invalid data, size, domain or extended knots.
fda::BSplineBasis uniform_basis(const Vector<>& x, const Vector<>& y, std::size_t n_basis,
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

/// @brief Sorts/deduplicates breakpoints, includes the data domain and extends outer spacings.
/// @param x Finite training coordinates.
/// @param y Matching finite responses.
/// @param knots Nonempty finite breakpoints.
/// @param degree Number of extra knots on each side.
/// @return Owned extended basis covering x and knots.
/// @throws std::invalid_argument For invalid data, knots or combined domain.
fda::BSplineBasis knot_basis(const Vector<>& x, const Vector<>& y, const Vector<>& knots,
                             std::size_t degree) {
  check_xy(x, y);
  if (knots.empty()) throw std::invalid_argument(std::string(kWho) + ": knots must not be empty");
  for (double knot : knots)
    if (!std::isfinite(knot)) throw std::invalid_argument(std::string(kWho)+": knots must be finite");
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

/// @brief Fits a fixed penalty, translating numerical failure into facade context.
/// @param x Training coordinates.
/// @param y Matching responses.
/// @param basis Fixed basis covering x.
/// @param lambda Finite nonnegative penalty multiplier.
/// @param penalty_order Positive order less than the basis size.
/// @return Owned FDA fit.
/// @throws std::invalid_argument For invalid data/options.
/// @throws std::runtime_error For singular/non-finite equations.
/// @throws std::overflow_error For overflowing difference weights/penalty.
fda::PSplineFitResult fit_fixed(const Vector<>& x, const Vector<>& y, const fda::BSplineBasis& basis,
                                double lambda, std::size_t penalty_order) {
  if (!std::isfinite(lambda) || !(lambda >= 0.0)) {
    throw std::invalid_argument(std::string(kWho) + ": lambda must be finite and nonnegative");
  }
  fda::PSplineOptions opts;
  opts.lambda = lambda;
  opts.penalty_order = penalty_order;
  try {
    return fda::fit_pspline(x.values(), y.values(), basis, opts);
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(kWho) + ": " + e.what());
  } catch (const std::overflow_error&) {
    throw;
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(std::string(kWho) + ": penalized system is singular (" + e.what() +
                             "); adjust lambda or the basis");
  }
}

// Minimizes GCV over log10(lambda): a coarse grid over [1e-8, 1e8], then golden-section search
// on the bracket around the best grid point.
/// @brief Coarse log-lambda scan with bounded golden-section refinement of its best bracket.
/// @param x Training coordinates.
/// @param y Matching responses.
/// @param basis Fixed cubic basis covering x.
/// @return Selected order-two fit from lambda in [1e-8,1e8].
/// @throws std::invalid_argument For invalid inputs.
/// @throws std::runtime_error If no candidate fits numerically.
fda::PSplineFitResult fit_gcv(const Vector<>& x, const Vector<>& y, const fda::BSplineBasis& basis) {
  /// @brief Default second differences for automatic smoothing selection.
  constexpr std::size_t kPenaltyOrder = 2;
  /// @brief Log10-lambda scan endpoints and coarse step size.
  constexpr double kLo = -8.0, kHi = 8.0, kStep = 0.5;

  /// @brief Fits at a log10-lambda candidate on the fixed basis.
  auto fit_at = [&](double log_lambda) -> fda::PSplineFitResult {
    return fit_fixed(x, y, basis, std::pow(10.0, log_lambda), kPenaltyOrder);
  };
  // A tiny lambda can leave the system numerically singular when basis functions lack data;
  // score those as +inf rather than failing the whole search.
  /// @brief Scores failed numerical candidates as infinity to keep the search running.
  auto score_at = [&](double log_lambda) {
    try {
      return fit_at(log_lambda).gcv();
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

PSplineRegression::PSplineRegression(const Vector<>& x, const Vector<>& y, std::size_t n_basis)
    : fit_(fit_gcv(x, y, uniform_basis(x, y, n_basis, 3))) {}

PSplineRegression::PSplineRegression(const Vector<>& x, const Vector<>& y, std::size_t n_basis,
                                     double lambda, std::size_t degree, std::size_t penalty_order)
    : fit_(fit_fixed(x, y, uniform_basis(x, y, n_basis, degree), lambda,
                     penalty_order)) {}

PSplineRegression::PSplineRegression(const Vector<>& x, const Vector<>& y, const Vector<>& knots)
    : fit_(fit_gcv(x, y, knot_basis(x, y, knots, 3))) {}

PSplineRegression::PSplineRegression(const Vector<>& x, const Vector<>& y, const Vector<>& knots,
                                     double lambda, std::size_t degree, std::size_t penalty_order)
    : fit_(fit_fixed(x, y, knot_basis(x, y, knots, degree), lambda,
                     penalty_order)) {}

double PSplineRegression::gcv() const noexcept { return fit_.gcv(); }

Vector<> PSplineRegression::predict(const Vector<>& x) const {
  const auto [lo, hi] = domain();
  Vector<> clamped = x;
  for (double& v : clamped) {
    if (!std::isfinite(v)) throw std::invalid_argument(std::string(kWho)+": prediction coordinates must be finite");
    v = std::clamp(v, lo, hi);
  }
  return fit_.predict(clamped.values());
}

linalg::DenseMatrix<double> PSplineRegression::basis_matrix(const Vector<>& x) const {
  const auto [lo, hi] = domain();
  auto clamped = x.values();
  for (double& v : clamped) {
    if (!std::isfinite(v)) throw std::invalid_argument(std::string(kWho)+": basis coordinates must be finite");
    v = std::clamp(v,lo,hi);
  }
  return fit_.basis.evaluate_dense(clamped);
}

Vector<> PSplineRegression::coefficient_differences() const {
  return fda::pspline_difference_matrix(basis_size(),penalty_order()) * fit_.coefficients;
}

dstruct::DataFrame PSplineRegression::summary() const {
  dstruct::DataFrame table;
  table.add_column("n",std::vector<double>{static_cast<double>(fit_.observations)});
  table.add_column("basis",std::vector<double>{static_cast<double>(basis_size())});
  table.add_column("degree",std::vector<double>{static_cast<double>(degree())});
  table.add_column("order",std::vector<double>{static_cast<double>(penalty_order())});
  table.add_column("lambda",std::vector<double>{lambda()});
  table.add_column("edf",std::vector<double>{effective_df()});
  table.add_column("rss",std::vector<double>{fit_.residual_sum_of_squares});
  table.add_column("roughness",std::vector<double>{fit_.penalty});
  table.add_column("objective",std::vector<double>{fit_.objective});
  table.add_column("r_squared",std::vector<double>{r_squared()});
  table.add_column("gcv",std::vector<double>{gcv()});
  table.add_column("loo_cv",std::vector<double>{fit_.loo_cv()});
  return table;
}

dstruct::DataFrame PSplineRegression::observation_details() const {
  std::vector<double> index(fit_.observations), response(fit_.observations);
  for (std::size_t i=0;i<index.size();++i) {
    index[i]=static_cast<double>(i+1); response[i]=fit_.fitted[i]+fit_.residuals[i];
  }
  dstruct::DataFrame table;
  table.add_column("observation",index); table.add_column("response",response);
  table.add_column("fitted",fit_.fitted); table.add_column("residual",fit_.residuals);
  table.add_column("leverage",fit_.leverage); table.add_column("loo_residual",fit_.loo_residuals());
  return table;
}

dstruct::DataFrame PSplineRegression::smoothing_path(const Vector<>& x, const Vector<>& y,
    std::size_t n_basis, const Vector<>& lambdas, std::size_t degree, std::size_t penalty_order) {
  if (lambdas.empty()) throw std::invalid_argument(std::string(kWho)+": lambda grid must not be empty");
  const auto basis=uniform_basis(x,y,n_basis,degree);
  std::vector<double> edf,rss,roughness,objective,gcv,loo;
  for (double lambda : lambdas) {
    const auto fit=fit_fixed(x,y,basis,lambda,penalty_order);
    edf.push_back(fit.effective_degrees_of_freedom); rss.push_back(fit.residual_sum_of_squares);
    roughness.push_back(fit.penalty); objective.push_back(fit.objective);
    gcv.push_back(fit.gcv()); loo.push_back(fit.loo_cv());
  }
  dstruct::DataFrame table;
  table.add_column("lambda",lambdas.values()); table.add_column("edf",edf);
  table.add_column("rss",rss); table.add_column("roughness",roughness);
  table.add_column("objective",objective); table.add_column("gcv",gcv); table.add_column("loo_cv",loo);
  return table;
}

} // namespace datamunge
