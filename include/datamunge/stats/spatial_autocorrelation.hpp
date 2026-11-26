#pragma once

/// @file spatial_autocorrelation.hpp
/// @brief Spatial weights and global Moran's I and Geary's C tests of spatial autocorrelation.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief Binary k-nearest-neighbour weights: w_ij = 1 when j is among the k nearest points of i.
///
/// Distances are Euclidean in the columns of `coordinates`; ties at the k-th distance are broken by
/// index. The matrix is generally asymmetric and has a zero diagonal, as `spdep::knn2nb`.
/// @param coordinates n-by-d point coordinates.
/// @param k Neighbours per point (1 <= k < n).
/// @return n-by-n weights.
/// @throws std::invalid_argument For k outside [1, n) or nonfinite coordinates.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> xy(4, 1, std::vector<double>{0, 1, 3, 7});
/// const auto w = dm::stats::knn_weights(xy, 1);
/// std::println("{} {}", w(0, 1), w(3, 2)); // 1 1
/// @endcode
linalg::DenseMatrix<double> knn_weights(const linalg::DenseMatrix<double>& coordinates, std::size_t k);

/// @brief Binary distance-band weights: w_ij = 1 when lower < d_ij <= upper (i != j), as `spdep::dnearneigh`.
/// @param coordinates n-by-d point coordinates.
/// @param lower Lower distance bound (exclusive; use 0 for all points within `upper`).
/// @param upper Upper distance bound (inclusive).
/// @return Symmetric n-by-n weights.
/// @throws std::invalid_argument For bounds with lower < 0 or upper <= lower, or nonfinite coordinates.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> xy(3, 1, std::vector<double>{0, 1, 3});
/// const auto w = dm::stats::distance_band_weights(xy, 0, 1.5);
/// std::println("{} {}", w(0, 1), w(1, 2)); // 1 0
/// @endcode
linalg::DenseMatrix<double> distance_band_weights(const linalg::DenseMatrix<double>& coordinates, double lower,
                                                  double upper);

/// @brief Divides each row by its sum (style "W"); rows summing to zero are left at zero.
/// @param weights Square weights.
/// @return Row-standardised weights.
/// @throws std::invalid_argument For a non-square matrix or negative weights.
/// @par Example
/// @code
/// const auto w = dm::stats::row_standardize(dm::linalg::DenseMatrix<double>(2, 2, std::vector<double>{0, 2, 1, 0}));
/// std::println("{} {}", w(0, 1), w(1, 0)); // 1 1
/// @endcode
linalg::DenseMatrix<double> row_standardize(const linalg::DenseMatrix<double>& weights);

/// @brief Direction of the alternative hypothesis.
enum class SpatialAlternative {
    positive,  ///< Positive autocorrelation (I above, C below its expectation); spdep "greater".
    negative,  ///< Negative autocorrelation; spdep "less".
    two_sided, ///< Either direction.
};

/// @brief Options for morans_i() and gearys_c().
/// @par Example
/// @code
/// const dm::stats::SpatialAutocorrelationOptions o{.permutations = 999};
/// std::println("{}", o.randomisation); // true
/// @endcode
struct SpatialAutocorrelationOptions {
    bool randomisation{true}; ///< Variance under randomisation (true) or normality (false).
    SpatialAlternative alternative{SpatialAlternative::positive}; ///< Alternative hypothesis.
    std::size_t permutations{0}; ///< Random permutations of the values for a Monte Carlo test (0 skips).
    std::uint64_t seed{1};       ///< Seed of the permutations.
};

/// @brief Result of a global spatial autocorrelation test.
/// @par Example
/// @code
/// dm::stats::SpatialAutocorrelationTest t{.statistic = 0.3, .expectation = -0.02};
/// std::println("{}", t.statistic > t.expectation); // true
/// @endcode
struct SpatialAutocorrelationTest {
    double statistic{};   ///< I or C.
    double expectation{}; ///< Null expectation: -1/(n-1) for I, 1 for C.
    double variance{};    ///< Null variance under the chosen assumption.
    double z{};           ///< Standard deviate, signed so that positive means positive autocorrelation.
    double p_value{};     ///< Normal-approximation p-value for the alternative.
    double permutation_p_value{}; ///< (1 + #{as or more extreme}) / (1 + permutations); NaN without permutations.
    double permutation_mean{};    ///< Mean of the permuted statistics (NaN without permutations).
    double permutation_sd{};      ///< SD of the permuted statistics (NaN without permutations).
};

/// @brief Global Moran's I test of spatial autocorrelation.
///
/// \f$I=\frac{n}{S_0}\frac{\sum_{ij}w_{ij}z_iz_j}{\sum_iz_i^2}\f$ with \f$z=x-\bar x\f$ and
/// \f$S_0=\sum_{ij}w_{ij}\f$. Under normality,
/// \f$\mathrm{Var}(I)=\frac{n^2S_1-nS_2+3S_0^2}{S_0^2(n^2-1)}-E(I)^2\f$, with
/// \f$S_1=\frac12\sum_{ij}(w_{ij}+w_{ji})^2\f$ and \f$S_2=\sum_i(w_{i\cdot}+w_{\cdot i})^2\f$. Under
/// randomisation the variance also depends on the kurtosis \f$K=n\sum z^4/(\sum z^2)^2\f$ (Cliff and
/// Ord 1981). The z-score uses \f$(I-E(I))/\sqrt{\mathrm{Var}(I)}\f$. Matches `spdep::moran.test`
/// (without islands); the permutation test follows `spdep::moran.mc` with the (1 + count) / (1 + nsim)
/// convention.
/// @param values Observed variable at the n locations.
/// @param weights n-by-n spatial weights with a zero diagonal (any style).
/// @param options Variance assumption, alternative and permutations.
/// @return The test.
/// @throws std::invalid_argument For fewer than four locations, mismatched sizes, nonfinite values,
/// constant values, a nonzero diagonal, or weights summing to zero.
/// @par Example
/// @code
/// std::vector<double> x;
/// for (int i = 0; i < 20; ++i)
///     x.push_back(i < 10 ? 1.0 + 0.1 * (i % 3) : 3.0 - 0.1 * (i % 2));
/// const dm::linalg::DenseMatrix<double> xy(20, 1, std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
///                                                                      11, 12, 13, 14, 15, 16, 17, 18, 19});
/// const auto w = dm::stats::row_standardize(dm::stats::knn_weights(xy, 2));
/// std::println("{}", dm::stats::morans_i(x, w).p_value < 0.001); // true
/// @endcode
SpatialAutocorrelationTest morans_i(const std::vector<double>& values, const linalg::DenseMatrix<double>& weights,
                                    const SpatialAutocorrelationOptions& options = {});

/// @brief Global Geary's C test of spatial autocorrelation.
///
/// \f$C=\frac{(n-1)\sum_{ij}w_{ij}(x_i-x_j)^2}{2S_0\sum_iz_i^2}\f$, with expectation 1; values below 1
/// indicate positive autocorrelation. The variances under normality and randomisation are those of
/// Cliff and Ord (1981), and the deviate is \f$(1-C)/\sqrt{\mathrm{Var}(C)}\f$. Matches
/// `spdep::geary.test` and `spdep::geary.mc`.
/// @param values Observed variable at the n locations.
/// @param weights n-by-n spatial weights with a zero diagonal.
/// @param options Variance assumption, alternative and permutations.
/// @return The test.
/// @throws std::invalid_argument As for morans_i().
/// @par Example
/// @code
/// const std::vector<double> x{1, 1.2, 1.1, 3, 3.1, 2.9, 1, 1.3, 3.2, 3};
/// const dm::linalg::DenseMatrix<double> xy(10, 1, std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
/// const auto c = dm::stats::gearys_c(x, dm::stats::distance_band_weights(xy, 0, 1));
/// std::println("{}", c.statistic < 1); // true
/// @endcode
SpatialAutocorrelationTest gearys_c(const std::vector<double>& values, const linalg::DenseMatrix<double>& weights,
                                    const SpatialAutocorrelationOptions& options = {});

/// @brief Quadrant of a location in the Moran scatterplot (own deviation against neighbours' lag).
enum class LisaQuadrant {
    high_high, ///< Above the mean, neighbours above the mean (hot-spot cluster).
    low_low,   ///< Below the mean, neighbours below the mean (cold-spot cluster).
    high_low,  ///< Above the mean, neighbours below (spatial outlier).
    low_high,  ///< Below the mean, neighbours above (spatial outlier).
};

/// @brief Options for local_morans_i() and local_getis_ord().
/// @par Example
/// @code
/// const dm::stats::LocalStatisticOptions o{.permutations = 499};
/// std::println("{}", o.conditional); // true
/// @endcode
struct LocalStatisticOptions {
    bool conditional{true}; ///< Local Moran: moments under conditional (true) or total randomisation.
    SpatialAlternative alternative{SpatialAlternative::two_sided}; ///< Alternative for the p-values.
    std::size_t permutations{0}; ///< Conditional permutations per location (0 skips).
    std::uint64_t seed{1};       ///< Seed of the permutations.
};

/// @brief A location-specific statistic with analytic and permutation inference.
/// @par Example
/// @code
/// dm::stats::LocalStatistic s{.statistic = 1.4, .expectation = -0.02};
/// std::println("{}", s.statistic > s.expectation); // true
/// @endcode
struct LocalStatistic {
    double statistic{};   ///< I_i or G_i.
    double expectation{}; ///< Null expectation.
    double variance{};    ///< Null variance.
    double z{};           ///< (statistic - expectation) / sqrt(variance).
    double p_value{};     ///< Normal-approximation p-value for the alternative.
    double permutation_p_value{}; ///< Conditional-permutation p-value (NaN without permutations); two-sided is folded.
    double permutation_mean{};    ///< Mean of the conditionally permuted statistic (NaN without permutations).
    double permutation_sd{};      ///< SD of the conditionally permuted statistic (NaN without permutations).
    LisaQuadrant quadrant{LisaQuadrant::high_high}; ///< Moran-scatterplot quadrant from z_i and its lag.
};

/// @brief Local Moran statistics (LISA, Anselin 1995).
///
/// \f$I_i=\frac{z_i}{m_2}\sum_jw_{ij}z_j\f$ with \f$m_2=\sum z^2/n\f$, so that
/// \f$\sum_iI_i\f$ is proportional to global I. Under conditional randomisation (z_i fixed, the
/// other values permuted) \f$E(I_i)=-z_i^2W_i/((n-1)m_2)\f$ and the variance of Sokal, Oden and
/// Thomson (1998); under total randomisation \f$E(I_i)=-W_i/(n-1)\f$. The conditional
/// permutation test reassigns the other n - 1 values to i's neighbours. The two-sided permutation
/// p-value is folded: \f$(1+\min(\#\ge,\#\le))/(1+R)\f$, as `spdep::localmoran_perm`. Analytic
/// values match `spdep::localmoran` (default `mlvar`, no islands). p-values are not adjusted for
/// the n simultaneous tests.
/// @param values Observed variable.
/// @param weights n-by-n weights with a zero diagonal.
/// @param options Conditional or total moments, alternative and permutations.
/// @return One statistic per location.
/// @throws std::invalid_argument As for morans_i().
/// @par Example
/// @code
/// const std::vector<double> x{1, 1.2, 0.9, 1.1, 5, 5.2, 4.9, 1, 1.1, 0.8};
/// const dm::linalg::DenseMatrix<double> xy(10, 1, std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
/// const auto w = dm::stats::row_standardize(dm::stats::distance_band_weights(xy, 0, 1));
/// const auto lisa = dm::stats::local_morans_i(x, w);
/// std::println("{}", lisa[5].quadrant == dm::stats::LisaQuadrant::high_high); // true
/// @endcode
std::vector<LocalStatistic> local_morans_i(const std::vector<double>& values,
                                           const linalg::DenseMatrix<double>& weights,
                                           const LocalStatisticOptions& options = {});

/// @brief Getis-Ord local G_i (or G_i* when the weights include the diagonal).
///
/// Without self-weights, \f$G_i=\sum_{j\ne i}w_{ij}x_j/\sum_{j\ne i}x_j\f$, with mean and variance
/// computed from the other n - 1 values. With `star` true, the diagonal weight w_ii is included in
/// the numerator and \f$G_i^*=\sum_jw_{ij}x_j/\sum_jx_j\f$ uses all n values (Ord and Getis 1995). The
/// z-scores and p-values match `spdep::localG` (`include.self` for G*). Large positive z indicates a
/// hot spot. The variable should be nonnegative for the ratio to be interpretable. The conditional
/// permutation test fixes x_i at its own location.
/// @param values Observed variable.
/// @param weights n-by-n weights; the diagonal must be zero unless `star` is true.
/// @param star Compute G_i* (self included) instead of G_i.
/// @param options Alternative and permutations (`conditional` is ignored).
/// @return One statistic per location (quadrant as for local_morans_i()).
/// @throws std::invalid_argument For mismatched sizes, nonfinite values, constant values, a nonzero
/// diagonal without `star`, or fewer than four locations.
/// @par Example
/// @code
/// const std::vector<double> x{1, 1.2, 0.9, 1.1, 5, 5.2, 4.9, 1, 1.1, 0.8};
/// const dm::linalg::DenseMatrix<double> xy(10, 1, std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
/// const auto g = dm::stats::local_getis_ord(x, dm::stats::distance_band_weights(xy, 0, 1));
/// std::println("{}", g[5].z > 1.96); // true
/// @endcode
std::vector<LocalStatistic> local_getis_ord(const std::vector<double>& values,
                                            const linalg::DenseMatrix<double>& weights, bool star = false,
                                            const LocalStatisticOptions& options = {});

} // namespace datamunge::stats
