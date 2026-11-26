#pragma once

/// @file tail_index.hpp
/// @brief Tail-index (extreme value index) estimation from upper order statistics: Hill, Pickands and the
/// Dekkers-Einmahl-de Haan moment estimators, paths over the number k of order statistics, and Weissman
/// extrapolated quantiles.

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Tail-index estimator.
/// @par Example
/// @code
/// const auto m = dm::stats::TailIndexMethod::pickands;
/// std::println("{}", m == dm::stats::TailIndexMethod::pickands); // true
/// @endcode
enum class TailIndexMethod {
    hill,     ///< \f$\frac1k\sum_{i=1}^k\log X_{n-i+1:n}-\log X_{n-k:n}\f$: efficient, only for \f$\xi>0\f$ and positive data.
    pickands, ///< \f$\log\bigl[(X_{n-k+1}-X_{n-2k+1})/(X_{n-2k+1}-X_{n-4k+1})\bigr]/\log2\f$: any \f$\xi\f$, location-scale invariant, high variance.
    moment,   ///< Dekkers, Einmahl and de Haan (1989): \f$M_1+1-\tfrac12(1-M_1^2/M_2)^{-1}\f$, any \f$\xi\f$, positive data.
};

/// @brief A tail-index estimate at one k.
/// @par Example
/// @code
/// dm::stats::TailIndexEstimate e{.k = 50};
/// std::println("{}", e.k); // 50
/// @endcode
struct TailIndexEstimate {
    std::size_t k{};         ///< Number of upper order statistics used.
    double threshold{};      ///< The order statistic \f$X_{n-k:n}\f$ the estimate conditions on.
    double estimate{};       ///< \f$\hat\xi\f$ (tail index \f$\alpha=1/\xi\f$ for Pareto-type tails).
    double standard_error{}; ///< Asymptotic standard error at \f$\hat\xi\f$ (ignores bias).
    double lower{};          ///< Normal interval.
    double upper{};          ///< Upper limit.
};

/// @brief Estimate the extreme value index from the k largest observations.
/// @param data Observations (finite; Hill and moment need \f$X_{n-k:n}>0\f$).
/// @param k Number of upper order statistics: 1 <= k < n (Pickands: 4k <= n; moment: k >= 2).
/// @param method Estimator.
/// @param confidence Interval level.
/// @return The estimate.
/// @throws std::invalid_argument If k is out of range, the data are nonfinite, or the threshold is not positive
/// for Hill or moment.
/// @par Example
/// @code
/// const std::vector<double> x{1, 2, 4, 8, 16, 32, 64, 128};
/// std::println("{:.4f}", dm::stats::tail_index(x, 3, dm::stats::TailIndexMethod::hill).estimate); // 1.3863
/// @endcode
TailIndexEstimate tail_index(const std::vector<double>& data, std::size_t k, TailIndexMethod method,
                             double confidence = 0.95);

/// @brief Estimates for every k in [k_min, k_max] (the "Hill plot" and its analogues).
/// @param data Observations.
/// @param method Estimator.
/// @param k_min Smallest k.
/// @param k_max Largest k (clipped to the method's admissible maximum).
/// @param confidence Interval level.
/// @return One estimate per admissible k (k values with a nonpositive threshold are skipped for Hill/moment).
/// @throws std::invalid_argument If k_min is 0 (1 for moment) or exceeds k_max, or the data are nonfinite.
/// @par Example
/// @code
/// const std::vector<double> x{1, 2, 4, 8, 16, 32, 64, 128};
/// std::println("{}", dm::stats::tail_index_path(x, dm::stats::TailIndexMethod::hill, 1, 5).size()); // 5
/// @endcode
std::vector<TailIndexEstimate> tail_index_path(const std::vector<double>& data, TailIndexMethod method,
                                               std::size_t k_min, std::size_t k_max, double confidence = 0.95);

/// @brief Weissman (1978) extrapolated quantile \f$\hat Q(1-p)=X_{n-k:n}\bigl((k+1)/((n+1)p)\bigr)^{\hat\xi}\f$
/// with the Hill estimate at k (`ReIns::Weissman.q`).
/// @param data Observations.
/// @param k Number of upper order statistics.
/// @param p Exceedance probability in (0, 1), typically below 1/n.
/// @return The quantile estimate.
/// @throws std::invalid_argument As for tail_index(), or p outside (0, 1).
/// @par Example
/// @code
/// const std::vector<double> x{1, 2, 4, 8, 16, 32, 64, 128};
/// std::println("{}", dm::stats::weissman_quantile(x, 3, 0.01) > 128); // true
/// @endcode
double weissman_quantile(const std::vector<double>& data, std::size_t k, double p);

} // namespace datamunge::stats
