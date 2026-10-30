#pragma once

/// @file knn_common.hpp
/// @brief Neighbour search shared by stats::KNNClassifier and stats::KNNRegressor: per-feature
///        scaling, exact leave-one-out rescaling, and the sorted distance list.

#include <datamunge/linalg/dense_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Distance used to rank neighbours, computed on the (standardised) predictors.
enum class DistanceMetric {
    Euclidean,  ///< \f$\lVert a-b\rVert_2=\sqrt{\sum_j(a_j-b_j)^2}\f$.
    Manhattan   ///< \f$\lVert a-b\rVert_1=\sum_j|a_j-b_j|\f$.
};

/// @brief The neighbours of a set of query points, as returned by the kneighbors() methods.
struct KNNNeighbors {
    /// @brief For each query, the training-row positions of its neighbours, nearest first.
    std::vector<std::vector<std::size_t>> index;
    /// @brief The matching distances, measured on the standardised predictors.
    std::vector<std::vector<double>> distance;
};

/// @brief Implementation helpers shared by the k-NN classes.
namespace detail {

/// @brief Column means and population standard deviations of the training predictors.
struct KnnScaling {
    std::vector<double> mean;   ///< Column means (0 when not standardising).
    std::vector<double> scale;  ///< Column standard deviations, divisor \f$n\f$ (1 when not standardising or constant).
    std::vector<double> ss;     ///< Centred column sums of squares, for the leave-one-out downdate.
};

/// @brief Computes the scaling of @p X.
/// @param X            Raw training predictors, one row per observation.
/// @param standardize  When false, every mean is 0 and every scale 1.
/// @return The scaling.
inline KnnScaling knn_scaling(const linalg::DenseMatrix<double>& X, bool standardize) {
    const std::size_t n = X.rows(), p = X.cols();
    KnnScaling s{std::vector<double>(p, 0.0), std::vector<double>(p, 1.0), std::vector<double>(p, 0.0)};
    if (!standardize || n == 0) return s;
    for (std::size_t j = 0; j < p; ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) sum += X(i, j);
        s.mean[j] = sum / static_cast<double>(n);
        double ss = 0.0;
        for (std::size_t i = 0; i < n; ++i) ss += (X(i, j) - s.mean[j]) * (X(i, j) - s.mean[j]);
        s.ss[j]          = ss;
        const double sd  = std::sqrt(ss / static_cast<double>(n));
        s.scale[j]       = sd < 1e-12 ? 1.0 : sd;
    }
    return s;
}

/// @brief Per-predictor multipliers \f$1/s_j\f$ that turn raw differences into standardised ones.
/// @param s  Scaling of the training rows.
/// @return One multiplier per predictor.
inline std::vector<double> knn_multipliers(const KnnScaling& s) {
    std::vector<double> m(s.scale.size());
    for (std::size_t j = 0; j < m.size(); ++j) m[j] = 1.0 / s.scale[j];
    return m;
}

/// @brief Multipliers \f$1/s_{-i,j}\f$ for the standardisation a fit without row @p i would use.
///
/// Removing \f$x_i\f$ downdates the centred sum of squares to
/// \f$SS_j-\tfrac{n}{n-1}(x_{ij}-\bar x_j)^2\f$, which gives \f$s_{-i,j}\f$ (divisor \f$n-1\f$, the
/// remaining count). The means cancel in differences, so only the scales matter.
/// @param s            Scaling of all rows.
/// @param X            Raw training predictors.
/// @param i            Row left out.
/// @param standardize  When false, every multiplier is 1.
/// @return One multiplier per predictor.
inline std::vector<double> knn_loo_multipliers(const KnnScaling& s, const linalg::DenseMatrix<double>& X,
                                               std::size_t i, bool standardize) {
    const std::size_t n = X.rows(), p = X.cols();
    std::vector<double> m(p, 1.0);
    if (!standardize) return m;
    for (std::size_t j = 0; j < p; ++j) {
        m[j] = 1.0 / s.scale[j];
        if (n < 3 || s.ss[j] / static_cast<double>(n) < 1e-24) continue;  // constant column: scale stays 1
        const double dev = X(i, j) - s.mean[j];
        const double ss  = std::max(0.0, s.ss[j] - static_cast<double>(n) / static_cast<double>(n - 1) * dev * dev);
        const double sd  = std::sqrt(ss / static_cast<double>(n - 1));
        m[j]             = sd < 1e-12 ? 1.0 : 1.0 / sd;
    }
    return m;
}

/// @brief The @p kk nearest training rows to @p query, sorted by distance and then row index.
///
/// Each difference is taken on the raw values and then multiplied, so equal raw differences
/// give exactly equal distances and ties are broken by row order, not by rounding.
/// @param X        Raw training predictors.
/// @param query    Raw query point.
/// @param metric   Distance.
/// @param kk       How many neighbours to return (at most the candidates available).
/// @param mult     Per-predictor multipliers (\f$1/s_j\f$), one per predictor.
/// @param exclude  A row to skip (leave-one-out), or `SIZE_MAX` for none.
/// @return Pairs (distance, row), nearest first.
inline std::vector<std::pair<double, std::size_t>> knn_nearest(const linalg::DenseMatrix<double>& X,
                                                               const std::vector<double>& query, DistanceMetric metric,
                                                               std::size_t kk, const std::vector<double>& mult,
                                                               std::size_t exclude = SIZE_MAX) {
    const std::size_t n = X.rows(), p = X.cols();
    std::vector<std::pair<double, std::size_t>> d;
    d.reserve(n);
    for (std::size_t r = 0; r < n; ++r) {
        if (r == exclude) continue;
        double s = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            const double diff = std::fabs(X(r, j) - query[j]) * mult[j];
            s += metric == DistanceMetric::Euclidean ? diff * diff : diff;
        }
        d.emplace_back(metric == DistanceMetric::Euclidean ? std::sqrt(s) : s, r);
    }
    kk = std::min(kk, d.size());
    std::partial_sort(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(kk), d.end());
    d.resize(kk);
    return d;
}

/// @brief Vote weight of a neighbour at distance @p d.
/// @param d         Distance.
/// @param weighted  Inverse-distance weighting (true) or uniform (false).
/// @return \f$1/(d+10^{-12})\f$ or 1; an exact match therefore outvotes every other neighbour.
inline double knn_weight(double d, bool weighted) { return weighted ? 1.0 / (d + 1e-12) : 1.0; }

} // namespace detail

} // namespace datamunge::stats
