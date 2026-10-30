// SWIG view of datamunge/vector.hpp and datamunge/{b,p}spline_regression.hpp, shared by the
// language bindings (%include "datamunge/vector.i" after plot.hpp and datamunge.hpp).
//
// The C++ API is built around dm::Vector, whose arithmetic/math overloads make no sense in a
// language that already has native numeric vectors (numpy / R / ...). So instead of wrapping
// those headers, this file declares the binding-relevant subset with Vector spelled as
// std::vector<double>, which every binding already maps to its native vector type. The
// generated wrappers still call the real functions: Vector converts implicitly to and from
// std::vector<double>.

// (Relies on the including module's %{ #include "datamunge/datamunge.hpp" %} block.)

// RGB/RGBA is a plain aggregate, which SWIG only gives a no-argument constructor.
%extend datamunge::graphics::RGB {
  RGB() { return new datamunge::graphics::RGB(); }
  RGB(int r, int g, int b, double alpha = 1.0) {
    return new datamunge::graphics::RGB{r, g, b, alpha};
  }
}

namespace datamunge {

/// Reseeds the generator shared by unif() and rnorm(), like R's set.seed().
void set_seed(std::uint64_t seed);
/// n draws from Uniform[lower, upper).
std::vector<double> unif(double lower, double upper, std::size_t n);
/// Alias for unif(lower, upper, n), sharing its generator and validation.
std::vector<double> runif(double lower, double upper, std::size_t n);
/// n draws from Normal(mean, sd).
std::vector<double> rnorm(double mean, double sd, std::size_t n);
/// n evenly spaced values from `from` to `to` inclusive.
std::vector<double> seql(double from, double to, std::size_t n);

/// Least-squares B-spline regression (see bspline_regression.hpp).
class BSplineRegression {
 public:
  /// n_basis B-splines of the given degree with equally spaced knots over the range of x.
  BSplineRegression(const std::vector<double>& x, const std::vector<double>& y,
                    std::size_t n_basis, std::size_t degree = 3);
  /// Basis on the given knot locations, like knots= in R's splines::bs().
  BSplineRegression(const std::vector<double>& x, const std::vector<double>& y,
                    const std::vector<double>& knots, std::size_t degree = 3);

  std::vector<double> predict(const std::vector<double>& x) const;
  std::vector<double> coefficients() const;
  std::vector<double> fitted() const;
  std::vector<double> residuals() const;
  std::vector<double> knots() const;
  double              r_squared() const;
  std::size_t         basis_size() const;
  std::size_t         degree() const;
};

/// Penalized B-spline (P-spline) regression (see pspline_regression.hpp).
class PSplineRegression {
 public:
  /// n_basis cubic B-splines with equally spaced knots; lambda chosen by GCV.
  PSplineRegression(const std::vector<double>& x, const std::vector<double>& y,
                    std::size_t n_basis);
  /// n_basis B-splines with a fixed smoothing parameter lambda.
  PSplineRegression(const std::vector<double>& x, const std::vector<double>& y,
                    std::size_t n_basis, double lambda, std::size_t degree = 3,
                    std::size_t penalty_order = 2);
  /// Cubic basis on the given knot locations; lambda chosen by GCV.
  PSplineRegression(const std::vector<double>& x, const std::vector<double>& y,
                    const std::vector<double>& knots);
  /// Basis on the given knot locations with a fixed smoothing parameter lambda.
  PSplineRegression(const std::vector<double>& x, const std::vector<double>& y,
                    const std::vector<double>& knots, double lambda, std::size_t degree = 3,
                    std::size_t penalty_order = 2);

  std::vector<double> predict(const std::vector<double>& x) const;
  std::vector<double> coefficients() const;
  std::vector<double> fitted() const;
  std::vector<double> residuals() const;
  std::vector<double> knots() const;
  double              r_squared() const;
  double              lambda() const;
  double              effective_df() const;
  double              gcv() const;
  std::size_t         penalty_order() const;
  std::size_t         basis_size() const;
  std::size_t         degree() const;
};

} // namespace datamunge
