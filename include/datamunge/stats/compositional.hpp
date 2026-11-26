#pragma once

/// @file compositional.hpp
/// @brief Compositional data analysis in the Aitchison geometry: closure, centered (clr), additive (alr) and
/// isometric (ilr) log-ratio coordinates with their inverses, orthonormal and sequential-binary-partition bases,
/// Aitchison distance, center, variation matrix and total variance, and replacement of rounded and count zeros.
///
/// Rows are compositions (observations), columns parts. Log-ratio functions require strictly positive parts:
/// replace zeros first with multiplicative_replacement() (values below a detection limit) or
/// count_zero_replacement() (zeros from finite counting).

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Sentinel for alr() and alr_inverse(): use the last part as the reference.
/// @par Example
/// @code
/// std::println("{}", dm::stats::last_part == std::numeric_limits<std::size_t>::max()); // true
/// @endcode
inline constexpr std::size_t last_part = std::numeric_limits<std::size_t>::max();

/// @brief Rescale each row to sum to `total`.
/// @param x Nonnegative finite matrix with positive row sums.
/// @param total Row total.
/// @return Closed matrix.
/// @throws std::invalid_argument On negative or nonfinite entries, a zero row, or total <= 0.
/// @par Example
/// @code
/// const auto c = dm::stats::closure(dm::linalg::DenseMatrix<double>(1, 3, {1, 1, 2}));
/// std::println("{}", c(0, 2)); // 0.5
/// @endcode
linalg::DenseMatrix<double> closure(const linalg::DenseMatrix<double>& x, double total = 1.0);

/// @brief Centered log-ratio \f$\mathrm{clr}(x)_j=\log x_j-\frac1D\sum_k\log x_k\f$ (rows sum to 0).
/// @param x Strictly positive compositions (any scale).
/// @return n x D coordinates.
/// @throws std::invalid_argument If any part is not strictly positive and finite.
/// @par Example
/// @code
/// const auto z = dm::stats::clr(dm::linalg::DenseMatrix<double>(1, 2, {1, std::exp(2.0)}));
/// std::println("{}", z(0, 1)); // 1
/// @endcode
linalg::DenseMatrix<double> clr(const linalg::DenseMatrix<double>& x);

/// @brief Inverse clr: closed \f$\exp(z)\f$.
/// @param z clr coordinates (rows are shifted to sum to 0 first).
/// @return Compositions closed to 1.
/// @throws std::invalid_argument On nonfinite input.
/// @par Example
/// @code
/// const auto x = dm::stats::clr_inverse(dm::linalg::DenseMatrix<double>(1, 2, {0, 0}));
/// std::println("{}", x(0, 0)); // 0.5
/// @endcode
linalg::DenseMatrix<double> clr_inverse(const linalg::DenseMatrix<double>& z);

/// @brief Additive log-ratio \f$\log(x_j/x_r)\f$ for every part j other than the reference r.
///
/// Not isometric: distances and angles in alr coordinates depend on the reference.
/// @param x Strictly positive compositions.
/// @param reference Index r of the denominator part (default: the last part, as in `compositions::alr`).
/// @return n x (D - 1) coordinates in the original part order with r removed.
/// @throws std::invalid_argument On nonpositive parts, D < 2 or reference out of range.
/// @par Example
/// @code
/// const auto z = dm::stats::alr(dm::linalg::DenseMatrix<double>(1, 3, {2, 1, 1}));
/// std::println("{:.4f}", z(0, 0)); // 0.6931
/// @endcode
linalg::DenseMatrix<double> alr(const linalg::DenseMatrix<double>& x, std::size_t reference = last_part);

/// @brief Inverse alr.
/// @param z alr coordinates (n x (D - 1)).
/// @param reference Reference part index in the D-part composition (default: last).
/// @return Compositions closed to 1.
/// @throws std::invalid_argument On nonfinite input or reference out of range.
/// @par Example
/// @code
/// const auto x = dm::stats::alr_inverse(dm::linalg::DenseMatrix<double>(1, 1, {0}));
/// std::println("{}", x(0, 1)); // 0.5
/// @endcode
linalg::DenseMatrix<double> alr_inverse(const linalg::DenseMatrix<double>& z, std::size_t reference = last_part);

/// @brief Built-in orthonormal ilr bases.
/// @par Example
/// @code
/// const auto b = dm::stats::IlrBasis::pivot;
/// std::println("{}", b == dm::stats::IlrBasis::pivot); // true
/// @endcode
enum class IlrBasis {
    helmert, ///< `compositions::ilrBase`: coordinate j contrasts part j+1 with parts 1..j.
    pivot,   ///< Pivot (robCompositions): coordinate j contrasts part j with all later parts; the first is
             ///< \f$\sqrt{(D-1)/D}\log(x_1/g(x_2,\dots,x_D))\f$.
};

/// @brief D x (D - 1) contrast matrix V with orthonormal columns orthogonal to the ones vector; ilr = clr V.
/// @param parts D >= 2.
/// @param basis Basis family.
/// @return V.
/// @throws std::invalid_argument If parts < 2.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::ilr_basis(2, dm::stats::IlrBasis::helmert)(1, 0)); // 0.7071
/// @endcode
linalg::DenseMatrix<double> ilr_basis(std::size_t parts, IlrBasis basis = IlrBasis::helmert);

/// @brief Balance basis from a sequential binary partition (Egozcue and Pawlowsky-Glahn 2005).
///
/// Column k of `partition` marks the parts in the numerator (+1), denominator (-1) or outside (0) of balance k.
/// With r numerator and s denominator parts the balance is \f$\sqrt{rs/(r+s)}\log(g(x_+)/g(x_-))\f$.
/// @param partition D x (D - 1) sign matrix; each column needs at least one +1 and one -1.
/// @return V (orthonormal when the partition is a valid sequential binary partition).
/// @throws std::invalid_argument On a wrong shape, entries outside {-1, 0, 1}, or a one-sided column.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> p(3, 2, {1, 1, -1, 1, 0, -1});
/// std::println("{:.4f}", dm::stats::balance_basis(p)(2, 1)); // -0.8165
/// @endcode
linalg::DenseMatrix<double> balance_basis(const linalg::DenseMatrix<double>& partition);

/// @brief Isometric log-ratio coordinates clr(x) V.
/// @param x Strictly positive compositions.
/// @param basis D x (D - 1) contrast matrix (from ilr_basis() or balance_basis()).
/// @return n x (D - 1) coordinates.
/// @throws std::invalid_argument On nonpositive parts or a basis of the wrong shape.
/// @par Example
/// @code
/// const auto z = dm::stats::ilr(dm::linalg::DenseMatrix<double>(1, 2, {1, std::exp(1.0)}),
///                               dm::stats::ilr_basis(2));
/// std::println("{:.4f}", z(0, 0)); // 0.7071
/// @endcode
linalg::DenseMatrix<double> ilr(const linalg::DenseMatrix<double>& x, const linalg::DenseMatrix<double>& basis);

/// @brief ilr coordinates in the Helmert basis (as `compositions::ilr`).
/// @param x Strictly positive compositions.
/// @return n x (D - 1) coordinates.
/// @throws std::invalid_argument On nonpositive parts or D < 2.
/// @par Example
/// @code
/// const auto z = dm::stats::ilr(dm::linalg::DenseMatrix<double>(1, 3, {1, 1, std::exp(1.0)}));
/// std::println("{:.4f}", z(0, 1)); // 0.8165
/// @endcode
linalg::DenseMatrix<double> ilr(const linalg::DenseMatrix<double>& x);

/// @brief Inverse ilr: clr_inverse(z V').
/// @param z n x (D - 1) coordinates.
/// @param basis The basis used for ilr().
/// @return Compositions closed to 1.
/// @throws std::invalid_argument On a shape mismatch or nonfinite input.
/// @par Example
/// @code
/// const auto x = dm::stats::ilr_inverse(dm::linalg::DenseMatrix<double>(1, 1, {0}), dm::stats::ilr_basis(2));
/// std::println("{}", x(0, 0)); // 0.5
/// @endcode
linalg::DenseMatrix<double> ilr_inverse(const linalg::DenseMatrix<double>& z, const linalg::DenseMatrix<double>& basis);

/// @brief Aitchison distance: Euclidean distance between clr vectors (scale invariant, subcompositionally
/// dominant).
/// @param a Strictly positive composition.
/// @param b Composition with the same number of parts.
/// @return Distance.
/// @throws std::invalid_argument On a size mismatch or nonpositive parts.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::aitchison_distance({1, 2, 3}, {10, 20, 30})); // 0.0000
/// @endcode
double aitchison_distance(const std::vector<double>& a, const std::vector<double>& b);

/// @brief All pairwise Aitchison distances between rows.
/// @param x Strictly positive compositions.
/// @return n x n symmetric distance matrix.
/// @throws std::invalid_argument On nonpositive parts.
/// @par Example
/// @code
/// const auto d = dm::stats::aitchison_distances(dm::linalg::DenseMatrix<double>(2, 2, {1, 1, 1, std::exp(1.0)}));
/// std::println("{:.4f}", d(0, 1)); // 0.7071
/// @endcode
linalg::DenseMatrix<double> aitchison_distances(const linalg::DenseMatrix<double>& x);

/// @brief Compositional center: closed column-wise geometric mean (the mean in clr space).
/// @param x Strictly positive compositions (n >= 1).
/// @return Center closed to 1.
/// @throws std::invalid_argument On nonpositive parts or no rows.
/// @par Example
/// @code
/// const auto g = dm::stats::compositional_center(dm::linalg::DenseMatrix<double>(2, 2, {1, 4, 4, 1}));
/// std::println("{}", g[0]); // 0.5
/// @endcode
std::vector<double> compositional_center(const linalg::DenseMatrix<double>& x);

/// @brief Variation matrix \f$\tau_{jk}=\mathrm{var}\log(x_j/x_k)\f$ (denominator n - 1), as `compositions::variation`.
/// @param x Strictly positive compositions (n >= 2).
/// @return D x D symmetric matrix with zero diagonal.
/// @throws std::invalid_argument On nonpositive parts or n < 2.
/// @par Example
/// @code
/// const auto t = dm::stats::variation_matrix(dm::linalg::DenseMatrix<double>(2, 2, {1, 1, 1, std::exp(1.0)}));
/// std::println("{}", t(0, 1)); // 0.5
/// @endcode
linalg::DenseMatrix<double> variation_matrix(const linalg::DenseMatrix<double>& x);

/// @brief Total variance: trace of the clr covariance, \f$\frac1{2D}\sum_{j,k}\tau_{jk}\f$ (`compositions::mvar`).
/// @param x Strictly positive compositions (n >= 2).
/// @return Total variance.
/// @throws std::invalid_argument On nonpositive parts or n < 2.
/// @par Example
/// @code
/// const auto v = dm::stats::total_variance(dm::linalg::DenseMatrix<double>(2, 2, {1, 1, 1, std::exp(1.0)}));
/// std::println("{}", v); // 0.25
/// @endcode
double total_variance(const linalg::DenseMatrix<double>& x);

/// @brief Multiplicative replacement of rounded zeros (values below a detection limit), as
/// `zCompositions::multRepl` (Martín-Fernández, Barceló-Vidal and Pawlowsky-Glahn 2003).
///
/// Each zero becomes \f$\delta=\text{fraction}\times\mathrm{DL}_j\f$ and the nonzero parts of the row are scaled
/// by \f$1-\sum\delta/c\f$, preserving their ratios. If every row has the same total c the result keeps that
/// total. Otherwise nonzero parts are kept and the replacements become \f$\delta/(1-\sum\delta/c_i)\f$
/// (the same compositions).
/// @param x Nonnegative data; zeros mark values below the detection limit.
/// @param detection_limits One limit per column (must be positive where the column has zeros).
/// @param fraction Fraction of the limit used, in (0, 1].
/// @return Matrix without zeros.
/// @throws std::invalid_argument On negative or nonfinite data, a size mismatch, nonpositive limits for a
/// zero, a row whose replacements reach its total, or fraction outside (0, 1].
/// @par Example
/// @code
/// const auto r = dm::stats::multiplicative_replacement(dm::linalg::DenseMatrix<double>(1, 3, {0, 0.5, 0.5}),
///                                                      {0.1, 0.1, 0.1});
/// std::println("{:.3f}", r(0, 0)); // 0.065
/// @endcode
linalg::DenseMatrix<double> multiplicative_replacement(const linalg::DenseMatrix<double>& x,
                                                       const std::vector<double>& detection_limits,
                                                       double fraction = 0.65);

/// @brief Prior for count_zero_replacement() (Martín-Fernández et al. 2015).
/// @par Example
/// @code
/// const auto m = dm::stats::CountZeroMethod::bayes_laplace;
/// std::println("{}", m == dm::stats::CountZeroMethod::bayes_laplace); // true
/// @endcode
enum class CountZeroMethod {
    geometric_bayesian,  ///< GBM: prior strength \f$s_i=1/g(t_i)\f$.
    square_root,         ///< SQ: \f$s_i=\sqrt{n_i}\f$.
    bayes_laplace,       ///< BL: \f$s_i=D\f$.
    count_zero_multiplicative, ///< CZM: zeros become fraction x threshold / \f$n_i\f$.
};

/// @brief Options for count_zero_replacement().
/// @par Example
/// @code
/// dm::stats::CountZeroOptions o{.adjust = false};
/// std::println("{}", o.adjust); // false
/// @endcode
struct CountZeroOptions {
    CountZeroMethod method{CountZeroMethod::geometric_bayesian}; ///< Prior.
    double fraction{0.65};  ///< Fraction for CZM and for adjusted replacements.
    double threshold{0.5};  ///< CZM count threshold.
    bool adjust{true};      ///< Cap a replacement above its column's minimum nonzero proportion at fraction x that minimum.
};

/// @brief Bayesian-multiplicative replacement of count zeros, as `zCompositions::cmultRepl` (output "prop").
///
/// The prior proportions \f$t_{ij}\f$ come from the column totals of the other rows; zeros become posterior
/// estimates \f$t_{ij}s_i/(n_i+s_i)\f$ and the nonzero proportions shrink multiplicatively.
/// @param counts Nonnegative counts (n >= 2 rows; no all-zero column).
/// @param options Options.
/// @return Proportions without zeros (rows sum to 1).
/// @throws std::invalid_argument On negative or nonfinite counts, fewer than 2 rows, a zero row, or a GBM prior
/// with a zero proportion.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> c(3, 2, {0, 10, 5, 5, 4, 6});
/// std::println("{}", dm::stats::count_zero_replacement(c)(0, 0) > 0); // true
/// @endcode
linalg::DenseMatrix<double> count_zero_replacement(const linalg::DenseMatrix<double>& counts,
                                                   const CountZeroOptions& options = {});

} // namespace datamunge::stats
