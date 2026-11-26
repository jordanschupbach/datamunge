#pragma once

/// @file extremal_index.hpp
/// @brief Extremal index of a stationary series from inter-exceedance times: the Ferro-Segers intervals estimator,
/// the runs estimator, runs declustering and the Ferro-Segers cluster bootstrap.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief Extremal-index estimator.
/// @par Example
/// @code
/// const auto m = dm::stats::ExtremalIndexMethod::runs;
/// std::println("{}", m == dm::stats::ExtremalIndexMethod::runs); // true
/// @endcode
enum class ExtremalIndexMethod {
    intervals, ///< Ferro and Segers (2003): moment estimator from inter-exceedance times; no tuning parameter.
    runs,      ///< Smith and Weissman (1994): clusters / exceedances, clusters separated by more than `run_length`.
};

/// @brief Options for extremal_index().
/// @par Example
/// @code
/// dm::stats::ExtremalIndexOptions o{.bootstrap = 500};
/// std::println("{}", o.bootstrap); // 500
/// @endcode
struct ExtremalIndexOptions {
    ExtremalIndexMethod method{ExtremalIndexMethod::intervals}; ///< Estimator.
    std::size_t run_length{1};  ///< Runs method: a gap T > run_length between exceedance times starts a new cluster.
    std::size_t bootstrap{0};   ///< Cluster-bootstrap replicates for the interval (0: none).
    double confidence{0.95};    ///< Interval level.
    std::uint64_t seed{1};      ///< Bootstrap seed.
};

/// @brief An extremal-index estimate.
/// @par Example
/// @code
/// dm::stats::ExtremalIndex e{.theta = 0.5};
/// std::println("{}", e.theta); // 0.5
/// @endcode
struct ExtremalIndex {
    double threshold{};          ///< u.
    std::size_t observations{};  ///< n.
    std::size_t exceedances{};   ///< \f$N_u\f$.
    double theta{};              ///< \f$\hat\theta\in(0,1]\f$: reciprocal of the limiting mean cluster size.
    bool all_gaps_short{};       ///< Intervals: all gaps were <= 2, so the \f$\hat\theta_1\f$ form was used.
    std::size_t clusters{};      ///< Number of clusters (intervals: \f$\lceil\hat\theta N_u\rceil\f$ adjusted for ties).
    std::size_t run_length{};    ///< Run length r defining the clusters (intervals: estimated; 0 when
                                 ///< \f$\hat\theta=1\f$, every exceedance its own cluster, where extRemes
                                 ///< reports the smallest gap instead).
    double lower{};              ///< Bootstrap percentile interval (NaN without bootstrap or with one cluster).
    double upper{};              ///< Upper limit.
    std::size_t bootstrap{};     ///< Replicates used.
};

/// @brief Estimate the extremal index of `data` at `threshold`.
///
/// The intervals estimator (Ferro and Segers 2003, the form used by `extRemes::extremalindex`) is
/// \f$\hat\theta=\min\{1,\,2(\sum(T_i-1))^2/((N-1)\sum(T_i-1)(T_i-2))\}\f$ for inter-exceedance times \f$T_i\f$
/// (with \f$T_i,\,T_i^2\f$ in place of \f$T_i-1,\,(T_i-1)(T_i-2)\f$ when no gap exceeds 2). Its clusters are split
/// at the \f$\lceil\hat\theta N\rceil-1\f$ largest gaps. The bootstrap resamples whole clusters and, separately,
/// the inter-cluster gaps (Ferro and Segers 2003, section 4).
/// @param data A stationary series in time order (finite).
/// @param threshold u; at least 2 observations must exceed it.
/// @param options Options.
/// @return The estimate.
/// @throws std::invalid_argument With fewer than 2 exceedances, nonfinite data, run_length 0 or confidence
/// outside (0, 1).
/// @par Example
/// @code
/// const std::vector<double> x{0, 5, 6, 0, 0, 0, 7, 0, 0, 0, 8, 9, 0, 0, 0, 6};
/// std::println("{:.2f}", dm::stats::extremal_index(x, 1, {.method = dm::stats::ExtremalIndexMethod::runs}).theta);
/// // 0.67
/// @endcode
ExtremalIndex extremal_index(const std::vector<double>& data, double threshold, const ExtremalIndexOptions& options = {});

/// @brief Clusters of exceedances and their maxima.
/// @par Example
/// @code
/// dm::stats::Declustered d;
/// std::println("{}", d.cluster_maxima.size()); // 0
/// @endcode
struct Declustered {
    std::vector<std::size_t> exceedance_index;      ///< Time index of each exceedance.
    std::vector<std::size_t> cluster;               ///< Cluster number (0-based) of each exceedance.
    std::vector<double> cluster_maxima;             ///< Largest value in each cluster.
    std::vector<std::size_t> cluster_maximum_index; ///< Time index of each cluster maximum (first if tied).
};

/// @brief Runs declustering: exceedances whose times differ by more than `run_length` start a new cluster
/// (run_length 0: every exceedance is a cluster).
///
/// Use the run length of an intervals estimate for the Ferro-Segers declustering scheme.
/// @param data Series in time order.
/// @param threshold u.
/// @param run_length r.
/// @return Clusters.
/// @throws std::invalid_argument On nonfinite data.
/// @par Example
/// @code
/// const std::vector<double> x{0, 5, 6, 0, 0, 0, 7, 0, 0, 0, 8, 9, 0, 0, 0, 6};
/// std::println("{}", dm::stats::decluster(x, 1, 1).cluster_maxima.size()); // 4
/// @endcode
Declustered decluster(const std::vector<double>& data, double threshold, std::size_t run_length);

} // namespace datamunge::stats
