#pragma once

/// @file generators.hpp
/// @brief Synthetic datasets with known structure, for testing methods against ground truth:
///        a Swiss roll for manifold learning, and Breiman's two-class benchmarks (as R's
///        mlbench) with known Bayes error for classifiers.

#include <datamunge/dstruct/dataframe.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::datasets {

/// @brief Points on a Swiss roll: a rectangle rolled into a spiral in 3-D, as scikit-learn's
///        `make_swiss_roll()`.
///
/// For each point, \f$t=\tfrac{3\pi}2(1+2u)\f$ and \f$h=21v\f$ with \f$u,v\f$ uniform on
/// \f$[0,1)\f$, and the point is \f$(t\cos t,\ h,\ t\sin t)\f$ plus independent
/// \f$N(0,\text{noise}^2)\f$ errors on each coordinate. The roll makes 1.5 turns,
/// \f$t\in[4.71, 14.14)\f$, and successive layers are \f$2\pi\f$ apart radially. Unrolled, the
/// sheet is a flat rectangle with coordinates \f$(s,h)\f$, where
/// \f$s(t)=\int_0^t\sqrt{1+\tau^2}\,d\tau=\tfrac12\big(t\sqrt{1+t^2}+\operatorname{asinh}t\big)\f$
/// is the arc length along the spiral \f$(t\cos t,t\sin t)\f$; \f$s\f$ runs from 12.5 to
/// 101.9, so the rectangle is about 89 by 21. A method that unrolls the roll recovers
/// \f$(s,h)\f$ up to an affine map, and \f$t\f$ only up to the monotone map \f$s(t)\f$. Draws use
/// datamunge's Xoshiro256** engine, so the data is the same on every platform (but differs from
/// scikit-learn's numbers).
///
/// @param n  Number of points.
/// @param noise  Standard deviation of the coordinate noise (default 0).
/// @param seed  Random seed (default 0).
/// @return A DataFrame with numeric columns `x`, `y`, `z` (the 3-D point), `t` and `height` (its
///         intrinsic coordinates) and `arc_length` (\f$s(t)\f$, the flat coordinate along the
///         spiral).
/// @throws std::invalid_argument  If @p noise is negative.
///
/// @par Example
/// @code
/// const auto roll = dm::datasets::swiss_roll(500, 0.0, 1);
/// std::print("{} {}\n", roll.nrows(), roll.columns());  // 500 ["x", "y", "z", "t", "height", "arc_length"]
/// @endcode
///
/// @par References
/// - S. T. Roweis and L. K. Saul (2000). "Nonlinear dimensionality reduction by locally linear
///   embedding". Science 290(5500), 2323-2326. doi:10.1126/science.290.5500.2323
/// - scikit-learn `make_swiss_roll`:
///   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html
///
/// @see stats::LLE, stats::Isomap
dstruct::DataFrame swiss_roll(std::size_t n, double noise = 0.0, std::uint64_t seed = 0);

/// @brief Breiman's twonorm benchmark: two Gaussian classes with identity covariance and means
///        \f$\pm a\mathbf 1\f$, \f$a=2/\sqrt d\f$ (R's `mlbench.twonorm`).
///
/// Each row's class is "1" or "2" with probability 1/2 each. The means are 4 apart whatever
/// \f$d\f$ is, so the Bayes rule (the sign of \f$\sum_jx_j\f$) has error \f$\Phi(-2)=0.02275\f$.
/// @param n     Number of rows.
/// @param d     Number of predictors (at least 1, default 20).
/// @param seed  Random seed.
/// @return Columns `x1` ... `xd` and the string column `class`.
/// @throws std::invalid_argument  If @p d is 0.
///
/// @par Example
/// @code
/// const auto train = dm::datasets::twonorm(300, 20, 1);
/// std::print("{} x {}\n", train.nrows(), train.ncols()); // 300 x 21
/// @endcode
///
/// @par References
/// - L. Breiman (1996). "Bias, variance and arcing classifiers". Technical report 460,
///   Statistics Department, University of California, Berkeley.
/// - F. Leisch and E. Dimitriadou (2024). *mlbench: Machine Learning Benchmark Problems*. R package.
dstruct::DataFrame twonorm(std::size_t n, std::size_t d = 20, std::uint64_t seed = 0);

/// @brief Breiman's threenorm benchmark: class "1" is an equal mixture of
///        \f$N(a\mathbf 1,I)\f$ and \f$N(-a\mathbf 1,I)\f$, class "2" is \f$N((a,-a,a,\dots),I)\f$, with
///        \f$a=2/\sqrt d\f$ (R's `mlbench.threenorm`).
/// @param n     Number of rows; classes have probability 1/2 each.
/// @param d     Number of predictors (at least 1, default 20).
/// @param seed  Random seed.
/// @return Columns `x1` ... `xd` and the string column `class`.
/// @throws std::invalid_argument  If @p d is 0.
dstruct::DataFrame threenorm(std::size_t n, std::size_t d = 20, std::uint64_t seed = 0);

/// @brief Breiman's ringnorm benchmark: class "1" is \f$N(0,4I)\f$, class "2" is
///        \f$N(a\mathbf 1,I)\f$ with \f$a=1/\sqrt d\f$ (R's `mlbench.ringnorm`).
/// @param n     Number of rows; classes have probability 1/2 each.
/// @param d     Number of predictors (at least 1, default 20).
/// @param seed  Random seed.
/// @return Columns `x1` ... `xd` and the string column `class`.
/// @throws std::invalid_argument  If @p d is 0.
dstruct::DataFrame ringnorm(std::size_t n, std::size_t d = 20, std::uint64_t seed = 0);

/// @brief Points uniform on \f$[-1,1]^d\f$, class "1" inside the centred ball that fills half the
///        cube and "2" outside (R's `mlbench.circle`), each label flipped with probability
///        @p flip.
///
/// The radius solves \f$\pi^{d/2}r^d/\Gamma(d/2+1)=2^{d-1}\f$; in the plane it is
/// \f$\sqrt{2/\pi}=0.7979\f$. For \f$d\ge4\f$ the radius exceeds 1, the ball pokes out of the cube and class
/// "1" covers less than half of it (46% for \f$d=5\f$), as in mlbench. For \f$\text{flip}<1/2\f$ the Bayes rule is the ball itself and its error
/// is @p flip.
/// @param n     Number of rows.
/// @param d     Number of predictors (at least 1, default 2).
/// @param flip  Label-noise probability, in \f$[0,1]\f$ (default 0).
/// @param seed  Random seed.
/// @return Columns `x1` ... `xd` and the string column `class`.
/// @throws std::invalid_argument  If @p d is 0 or @p flip is outside \f$[0,1]\f$.
///
/// @par Example
/// @code
/// const auto disc = dm::datasets::circle(200, 2, 0.1, 3);
/// std::print("{}\n", disc.columns()); // ["x1", "x2", "class"]
/// @endcode
dstruct::DataFrame circle(std::size_t n, std::size_t d = 2, double flip = 0.0, std::uint64_t seed = 0);

} // namespace datamunge::datasets
