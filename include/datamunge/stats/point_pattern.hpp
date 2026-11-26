#pragma once

/// @file point_pattern.hpp
/// @brief Second-order summaries and Poisson intensity regression for planar point patterns in a
/// rectangular window.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Axis-aligned rectangular observation window.
/// @par Example
/// @code
/// const dm::stats::RectangleWindow w{.xmax = 2, .ymax = 1};
/// std::println("{}", w.area()); // 2
/// @endcode
struct RectangleWindow {
    double xmin{0}; ///< Left edge.
    double xmax{1}; ///< Right edge.
    double ymin{0}; ///< Bottom edge.
    double ymax{1}; ///< Top edge.

    /// @brief Area of the window.
    /// @return (xmax - xmin)(ymax - ymin).
    [[nodiscard]] double area() const { return (xmax - xmin) * (ymax - ymin); }
};

/// @brief Edge correction for second-order summaries.
enum class EdgeCorrection {
    none,        ///< No correction (biased downward at larger r).
    border,      ///< Reduced sample: only points at least r from the boundary act as centres.
    translation, ///< Ohser-Stoyan translation weights |W| / |W cap (W + x_j - x_i)|.
    isotropic,   ///< Ripley's weights: 1 / fraction of the circle through x_j centred at x_i inside W (capped at 100).
};

/// @brief Ripley's K function, the expected number of further points within r of a typical point
/// divided by the intensity.
///
/// With \f$\hat\lambda^2=n(n-1)/|W|^2\f$, the weighted estimators are
/// \f$\hat K(r)=\frac{|W|}{n(n-1)}\sum_{i\ne j}1\{d_{ij}\le r\}e_{ij}\f$; the border estimator is
/// \f$\sum_{i:b_i\ge r}\#\{j:d_{ij}\le r\}/(\hat\lambda\,\#\{i:b_i\ge r\})\f$ with \f$\hat\lambda=n/|W|\f$
/// (NaN when no point is r from the boundary). Under complete spatial randomness \f$K(r)=\pi r^2\f$;
/// larger values indicate clustering and smaller values regularity. Matches `spatstat.explore::Kest`
/// (evaluated on an evenly spaced r grid, where spatstat's binning is exact).
/// @param points n-by-2 point coordinates inside the window.
/// @param window Rectangular window.
/// @param r Distances at which to evaluate (nonnegative).
/// @param correction Edge correction.
/// @return K at each r.
/// @throws std::invalid_argument For fewer than two points, points outside the window, a degenerate
/// window, or negative or nonfinite r.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> p(3, 2, std::vector<double>{0.2, 0.2, 0.3, 0.2, 0.8, 0.8});
/// const auto k = dm::stats::ripley_k(p, {}, {0.05, 0.2}, dm::stats::EdgeCorrection::none);
/// std::println("{} {:.4f}", k[0], k[1]); // 0 0.3333
/// @endcode
std::vector<double> ripley_k(const linalg::DenseMatrix<double>& points, const RectangleWindow& window,
                             const std::vector<double>& r, EdgeCorrection correction = EdgeCorrection::isotropic);

/// @brief Besag's variance-stabilised L function \f$L(r)=\sqrt{K(r)/\pi}\f$ (L(r) = r under CSR).
/// @param points n-by-2 point coordinates.
/// @param window Rectangular window.
/// @param r Distances.
/// @param correction Edge correction.
/// @return L at each r.
/// @throws std::invalid_argument As for ripley_k().
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> p(3, 2, std::vector<double>{0.2, 0.2, 0.3, 0.2, 0.8, 0.8});
/// const auto l = dm::stats::besag_l(p, {}, {0.2}, dm::stats::EdgeCorrection::none);
/// std::println("{:.4f}", l[0]); // 0.3257
/// @endcode
std::vector<double> besag_l(const linalg::DenseMatrix<double>& points, const RectangleWindow& window,
                            const std::vector<double>& r, EdgeCorrection correction = EdgeCorrection::isotropic);

/// @brief Options for pair_correlation().
/// @par Example
/// @code
/// const dm::stats::PairCorrelationOptions o{.stoyan = 0.1};
/// std::println("{}", o.divide_by_distance); // false
/// @endcode
struct PairCorrelationOptions {
    EdgeCorrection correction{EdgeCorrection::translation}; ///< none, translation or isotropic.
    double bandwidth{std::numeric_limits<double>::quiet_NaN()}; ///< Kernel SD; NaN uses stoyan / sqrt(5 lambda).
    double stoyan{0.15};          ///< Stoyan's rule: Epanechnikov half-width stoyan / sqrt(lambda).
    bool divide_by_distance{false}; ///< Divide each pair by 2 pi d_ij (spatstat divisor "d") instead of 2 pi r.
};

/// @brief Kernel estimate of the pair correlation function \f$g(r)=K'(r)/(2\pi r)\f$.
///
/// \f$\hat g(r)=\frac{|W|}{n(n-1)}\sum_{i\ne j}\frac{\kappa_h(r-d_{ij})\,e_{ij}}{2\pi r}\f$ with the
/// Epanechnikov kernel of standard deviation h. g = 1 under CSR, g > 1 at distances where pairs are
/// more frequent than random, and g < 1 where they are inhibited. Matches `spatstat.explore::pcf`
/// with `zerocor = "none"` and `fast = FALSE` (exact kernel sums; spatstat's default smooths on a grid).
/// With divisor r the estimate at r = 0 is NaN.
/// @param points n-by-2 point coordinates.
/// @param window Rectangular window.
/// @param r Distances.
/// @param options Correction, bandwidth and divisor.
/// @return g at each r.
/// @throws std::invalid_argument As for ripley_k(), for the border correction, or a nonpositive
/// bandwidth.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> p(3, 2, std::vector<double>{0.2, 0.2, 0.3, 0.2, 0.8, 0.8});
/// const auto g = dm::stats::pair_correlation(p, {}, {0.1, 0.5}, {.bandwidth = 0.02});
/// std::println("{}", g[0] > g[1]); // true
/// @endcode
std::vector<double> pair_correlation(const linalg::DenseMatrix<double>& points, const RectangleWindow& window,
                                     const std::vector<double>& r, const PairCorrelationOptions& options = {});

/// @brief Covariate design row at a location: returns z(x, y), including a 1 for an intercept.
using PointCovariates = std::function<std::vector<double>(double x, double y)>;

/// @brief Berman-Turner quadrature: data and dummy points with integration weights.
/// @par Example
/// @code
/// dm::stats::QuadratureScheme q;
/// std::println("{}", q.weights.size()); // 0
/// @endcode
struct QuadratureScheme {
    linalg::DenseMatrix<double> points; ///< m-by-2 quadrature points: the n data points first, then the dummies.
    std::vector<double> weights;        ///< Weights summing to the window area.
    std::size_t data_points{};          ///< Number of data points n.
};

/// @brief Builds spatstat's default Berman-Turner quadrature scheme.
///
/// Dummy points are the centres of an nx-by-ny grid of tiles plus the four window corners; each
/// quadrature point (data or dummy) receives the area of its tile divided by the number of
/// quadrature points in that tile ("counting weights"). With nx = ny = 0 the grid size is
/// \f$\max(32,\,10\lceil 2\sqrt n/10\rceil)\f$ per side, as `spatstat.geom::quadscheme`.
/// @param points n-by-2 data points inside the window.
/// @param window Rectangular window.
/// @param nx Grid tiles along x (0 for the default).
/// @param ny Grid tiles along y (0 for the default).
/// @return The quadrature scheme.
/// @throws std::invalid_argument For points outside the window, a degenerate window, or only one of
/// nx and ny set.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> p(1, 2, std::vector<double>{0.3, 0.3});
/// const auto q = dm::stats::berman_turner_quadrature(p, {}, 2, 2);
/// std::println("{} {:.4f}", q.points.rows(), q.weights[0]); // 9 0.0833
/// @endcode
QuadratureScheme berman_turner_quadrature(const linalg::DenseMatrix<double>& points, const RectangleWindow& window,
                                          std::size_t nx = 0, std::size_t ny = 0);

/// @brief Options for poisson_process_regression().
/// @par Example
/// @code
/// const dm::stats::PoissonProcessOptions o{.grid_x = 64, .grid_y = 32};
/// std::println("{}", o.max_iterations); // 100
/// @endcode
struct PoissonProcessOptions {
    std::size_t grid_x{0};          ///< Quadrature grid along x (0: spatstat default).
    std::size_t grid_y{0};          ///< Quadrature grid along y (0: spatstat default).
    std::size_t max_iterations{100}; ///< Newton iterations.
    double tolerance{1e-10};        ///< Convergence tolerance on the relative log-likelihood change.
};

/// @brief Fit of a log-linear Poisson intensity.
/// @par Example
/// @code
/// dm::stats::PoissonProcessFit f;
/// std::println("{}", f.converged); // false
/// @endcode
struct PoissonProcessFit {
    std::vector<double> coefficients;       ///< beta of \f$\lambda(u)=\exp(\beta^\top z(u))\f$.
    std::vector<double> standard_errors;    ///< From the inverse Fisher information.
    linalg::DenseMatrix<double> covariance; ///< Inverse of \f$\sum_jw_j\lambda_jz_jz_j^\top\f$.
    double log_likelihood{};  ///< Berman-Turner log-likelihood \f$\sum_i\log\lambda(x_i)-\sum_jw_j\lambda(u_j)\f$.
    double aic{};             ///< -2 log_likelihood + 2p.
    double expected_count{};  ///< Fitted integral of the intensity over the window (equals n with an intercept).
    std::size_t iterations{}; ///< Newton iterations used.
    bool converged{false};    ///< Whether the tolerance was met.
};

/// @brief Maximum-likelihood regression of an inhomogeneous Poisson intensity on spatial covariates.
///
/// The Poisson log-likelihood \f$\sum_i\log\lambda(x_i)-\int_W\lambda(u)\,du\f$ with
/// \f$\lambda(u)=\exp(\beta^\top z(u))\f$ is approximated by Berman-Turner quadrature and maximised by
/// Newton's method with step halving (the objective is concave). Standard errors come from the
/// observed Fisher information. Matches `spatstat.model::ppm` with the same quadrature scheme
/// (coefficients, `vcov`, `logLik`, `AIC`). Use a finer grid when covariates vary on scales smaller
/// than the default tiles; the fit is otherwise exact for the quadrature, not for the integral.
/// @param points n-by-2 data points.
/// @param window Rectangular window.
/// @param covariates Design row z(x, y) (same length p everywhere; include a 1 for an intercept).
/// @param options Quadrature grid and Newton controls.
/// @return Coefficients, standard errors and log-likelihood.
/// @throws std::invalid_argument For invalid points or window, an empty pattern, or covariate rows of
/// inconsistent length or nonfinite values.
/// @throws std::runtime_error If the Fisher information is singular (collinear covariates).
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> p(4, 2, std::vector<double>{0.1, 0.2, 0.4, 0.7, 0.8, 0.3, 0.6, 0.9});
/// const auto f = dm::stats::poisson_process_regression(p, {}, [](double, double) {
///     return std::vector<double>{1.0};
/// });
/// std::println("{:.4f}", std::exp(f.coefficients[0])); // 4.0000
/// @endcode
PoissonProcessFit poisson_process_regression(const linalg::DenseMatrix<double>& points, const RectangleWindow& window,
                                             const PointCovariates& covariates,
                                             const PoissonProcessOptions& options = {});

/// @brief Fitted intensity \f$\exp(\beta^\top z(x,y))\f$ at a location.
/// @param fit Fitted model.
/// @param covariates The covariate function used for the fit.
/// @param x Location x.
/// @param y Location y.
/// @return The intensity.
/// @throws std::invalid_argument If the covariate row length does not match the fit.
/// @par Example
/// @code
/// dm::stats::PoissonProcessFit f;
/// f.coefficients = {std::log(50.0), 1.0};
/// const auto z = [](double x, double) { return std::vector<double>{1.0, x}; };
/// std::println("{:.3f}", dm::stats::poisson_process_intensity(f, z, 0.0, 0.5)); // 50.000
/// @endcode
double poisson_process_intensity(const PoissonProcessFit& fit, const PointCovariates& covariates, double x, double y);

} // namespace datamunge::stats
