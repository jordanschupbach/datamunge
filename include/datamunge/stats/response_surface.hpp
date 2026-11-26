#pragma once

/// @file response_surface.hpp
/// @brief Central composite and Box–Behnken designs, and second-order response surfaces with canonical analysis.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Rule for the axial (star) distance \f$\alpha\f$ of a central composite design.
/// @par Example
/// @code
/// const auto rule = dm::stats::AxialDistance::face_centered;
/// std::println("{}", rule == dm::stats::AxialDistance::rotatable); // false
/// @endcode
enum class AxialDistance {
    rotatable,     ///< \f$\alpha=F^{1/4}\f$ for F factorial runs: prediction variance depends only on distance from the center.
    face_centered, ///< \f$\alpha=1\f$: axial points on the faces of the cube, three levels per factor.
    spherical,     ///< \f$\alpha=\sqrt{k}\f$: axial and factorial points on the same sphere.
    custom,        ///< Use CentralCompositeOptions::alpha.
};

/// @brief Construction options for central_composite_design().
/// @par Example
/// @code
/// const dm::stats::CentralCompositeOptions options{.axial = dm::stats::AxialDistance::spherical};
/// std::println("{} {}", options.factorial_center_points, options.axial_center_points); // 2 2
/// @endcode
struct CentralCompositeOptions {
    AxialDistance axial{AxialDistance::rotatable}; ///< How \f$\alpha\f$ is chosen.
    double alpha{};                                ///< Axial distance when `axial` is custom; must be positive.

    /// Generator words for a \f$2^{k-q}\f$ fractional factorial portion (see
    /// fractional_factorial_design()); empty for a full \f$2^k\f$ portion.
    std::vector<std::string> generators;
    std::size_t factorial_center_points{2};        ///< Center runs placed after the factorial portion.
    std::size_t axial_center_points{2};            ///< Center runs placed after the axial portion.
};

/// @brief A central composite design in coded units.
/// @par Example
/// @code
/// const auto d = dm::stats::central_composite_design(2);
/// std::println("{} runs, alpha {:.4f}", d.runs.rows(), d.alpha); // 12 runs, alpha 1.4142
/// @endcode
struct CentralCompositeDesign {
    /// Runs by factors: the factorial portion in standard order, its center runs, the axial
    /// pairs \f$(-\alpha,+\alpha)\f$ for factor 1, 2, ..., then the remaining center runs.
    linalg::DenseMatrix<double> runs;
    double alpha{};                 ///< Axial distance used.
    std::size_t factorial_runs{};   ///< Number of \f$\pm1\f$ corner runs F.
};

/// @brief Central composite design for k factors in coded units.
///
/// Combines a two-level factorial portion, \f$2k\f$ axial runs at \f$\pm\alpha\f$ on each
/// axis and center runs, in the run order of `rsm::ccd(k, n0, oneblock = TRUE,
/// randomize = FALSE)`. The factorial portion must have resolution V or more (or be
/// full) for all second-order terms to be estimable.
/// @param factors Number of factors k, from 2 to 20.
/// @param options Axial rule, fractional generators and center points.
/// @return Run matrix, axial distance and number of factorial runs.
/// @throws std::invalid_argument For k outside 2..20, a nonpositive custom alpha, or invalid generators.
/// @par Example
/// @code
/// const auto d = dm::stats::central_composite_design(3);
/// std::println("{} {:.6f}", d.runs.rows(), d.alpha); // 18 1.681793
/// @endcode
/// @par References
/// - G. E. P. Box and K. B. Wilson (1951). "On the experimental attainment of optimum conditions".
///   Journal of the Royal Statistical Society B 13(1), 1-45. doi:10.1111/j.2517-6161.1951.tb00067.x
/// - R. V. Lenth (2009). "Response-surface methods in R, using rsm". Journal of Statistical
///   Software 32(7). doi:10.18637/jss.v032.i07
/// @see box_behnken_design(), fit_quadratic_surface()
[[nodiscard]] CentralCompositeDesign central_composite_design(std::size_t factors,
                                                              const CentralCompositeOptions& options = {});

/// @brief Box–Behnken design for k factors in coded units.
///
/// For every pair of factors \f$i<j\f$ (in lexicographic order) it runs the \f$2^2\f$
/// factorial in those two factors, first factor fastest, with all other factors at 0, then
/// adds center runs. Every factor takes three levels and no run sits at a corner of the
/// cube. For three factors this is the run order of `rsm::bbd(3, randomize = FALSE)`; for four
/// and five factors it is the same set of runs as `rsm::bbd` in a different order. For six
/// or more factors the classical designs use balanced incomplete blocks of factor triples
/// and are smaller than this pairs construction.
/// @param factors Number of factors k, from 3 to 20.
/// @param center_points Number of center runs appended.
/// @return \f$4\binom{k}{2}+n_0\f$ runs by k factors.
/// @throws std::invalid_argument For k outside 3..20.
/// @par Example
/// @code
/// const auto d = dm::stats::box_behnken_design(3, 3);
/// std::println("{} {} {}", d.rows(), d(4, 0), d(4, 2)); // 15 -1 -1
/// @endcode
/// @par References
/// G. E. P. Box and D. W. Behnken (1960). "Some new three level designs for the study of
/// quantitative variables". Technometrics 2(4), 455-475. doi:10.1080/00401706.1960.10489912
/// @see central_composite_design()
[[nodiscard]] linalg::DenseMatrix<double> box_behnken_design(std::size_t factors, std::size_t center_points = 3);

/// @brief Nature of the stationary point of a fitted quadratic surface.
/// @par Example
/// @code
/// const auto kind = dm::stats::StationaryKind::saddle;
/// std::println("{}", kind == dm::stats::StationaryKind::maximum); // false
/// @endcode
enum class StationaryKind {
    maximum, ///< All eigenvalues of B negative.
    minimum, ///< All eigenvalues of B positive.
    saddle,  ///< Eigenvalues of both signs.

    /// Some eigenvalue is numerically zero: a stationary ridge. The reported point is the
    /// pseudo-inverse solution, the stationary point closest to the origin.
    ridge,
};

/// @brief Least-squares second-order model \f$\hat y=b_0+x^\top b+x^\top Bx\f$ and its canonical analysis.
/// @par Example
/// @code
/// const dm::stats::QuadraticSurface surface;
/// std::println("{}", surface.coefficients.size()); // 0
/// @endcode
struct QuadraticSurface {
    /// Coefficients in the order of `rsm(y ~ SO(x1, ..., xk))`: intercept, linear terms,
    /// two-factor interactions \f$x_1x_2,x_1x_3,\dots,x_{k-1}x_k\f$, then pure quadratics \f$x_j^2\f$.
    std::vector<double> coefficients;
    double intercept{};                 ///< \f$b_0\f$.
    std::vector<double> linear;         ///< \f$b\f$, the first-order coefficients.

    /// Symmetric B: pure quadratic coefficients on the diagonal and half of each
    /// interaction coefficient off the diagonal.
    linalg::DenseMatrix<double> quadratic;
    double residual_sd{};               ///< \f$\sqrt{\mathrm{RSS}/(n-p)}\f$.
    double r_squared{};                 ///< Coefficient of determination.
    std::vector<double> stationary_point; ///< \f$x_s=-\tfrac12B^{-1}b\f$ in coded units.
    double stationary_response{};       ///< \f$\hat y(x_s)=b_0+\tfrac12x_s^\top b\f$.
    std::vector<double> eigenvalues;    ///< Eigenvalues of B in decreasing order.
    linalg::DenseMatrix<double> eigenvectors; ///< Canonical axes as columns, in the order of `eigenvalues`.
    StationaryKind kind{StationaryKind::ridge}; ///< Maximum, minimum, saddle or ridge.
};

/// @brief Fits a full second-order polynomial by least squares and locates its stationary point.
///
/// Canonical analysis writes \f$\hat y=\hat y_s+\sum_i\lambda_iw_i^2\f$ along the
/// eigenvectors of B centered at \f$x_s\f$; the eigenvalue signs classify the point. An
/// eigenvalue with \f$|\lambda|\le10^{-8}\max|\lambda|\f$ is treated as zero.
/// Matches `rsm::rsm(y ~ SO(...))` coefficients and `canonical(fit, threshold = 0)`.
/// @param x Runs by k factors (k from 1 to 20) in coded units.
/// @param y Response, one value per run, finite.
/// @return Coefficients, fit summaries and canonical analysis.
/// @throws std::invalid_argument For mismatched sizes, nonfinite values, k outside 1..20, or not more
/// runs than the \f$(k+1)(k+2)/2\f$ coefficients.
/// @throws std::runtime_error When the design cannot estimate every second-order term.
/// @par Example
/// @code
/// const auto d = dm::stats::central_composite_design(2, {.axial = dm::stats::AxialDistance::face_centered});
/// std::vector<double> y;
/// for (std::size_t r = 0; r < d.runs.rows(); ++r) {
///     const double a = d.runs(r, 0) - .2, b = d.runs(r, 1) + .3;
///     y.push_back(10 - a * a - 2 * b * b);
/// }
/// const auto s = dm::stats::fit_quadratic_surface(d.runs, y);
/// std::println("{:.2f} {:.2f} {:.2f}", s.stationary_point[0], s.stationary_point[1], s.stationary_response); // 0.20 -0.30 10.00
/// @endcode
/// @par References
/// - G. E. P. Box and N. R. Draper (2007). Response Surfaces, Mixtures, and Ridge Analyses,
///   2nd ed. Wiley, ch. 11. doi:10.1002/0470072768
/// - R. V. Lenth (2009), doi:10.18637/jss.v032.i07
/// @see predict_quadratic_surface(), central_composite_design(), box_behnken_design()
[[nodiscard]] QuadraticSurface fit_quadratic_surface(const linalg::DenseMatrix<double>& x,
                                                     const std::vector<double>& y);

/// @brief Evaluates a fitted second-order surface at a point.
/// @param surface Fitted surface.
/// @param x Point with one coordinate per factor, in coded units.
/// @return \f$b_0+x^\top b+x^\top Bx\f$.
/// @throws std::invalid_argument When `x` has the wrong length.
/// @par Example
/// @code
/// const auto d = dm::stats::box_behnken_design(3, 3);
/// std::vector<double> y;
/// for (std::size_t r = 0; r < d.rows(); ++r)
///     y.push_back(1 + d(r, 0) * d(r, 1) + d(r, 2) * d(r, 2));
/// const auto s = dm::stats::fit_quadratic_surface(d, y);
/// std::println("{:.2f}", dm::stats::predict_quadratic_surface(s, {1, 1, 1})); // 3.00
/// @endcode
/// @see fit_quadratic_surface()
[[nodiscard]] double predict_quadratic_surface(const QuadraticSurface& surface, const std::vector<double>& x);

} // namespace datamunge::stats
