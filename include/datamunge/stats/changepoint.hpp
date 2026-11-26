#pragma once

/// @file changepoint.hpp
/// @brief Penalized multiple changepoint detection by PELT (pruned exact linear time).

#include <cstddef>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Segment cost (twice the negative maximized log likelihood), as R's `changepoint` package.
/// @par Example
/// @code
/// const auto c = dm::stats::ChangeCost::normal_mean;
/// std::println("{}", c == dm::stats::ChangeCost::poisson); // false
/// @endcode
enum class ChangeCost {
    normal_mean,          ///< Change in mean, known noise variance: \f$\sum(x-\bar x)^2/\sigma^2\f$ (`cpt.mean`).
    normal_variance,      ///< Change in variance about a fixed mean: \f$m(\ln2\pi+\ln\hat\sigma^2+1)\f$ (`cpt.var`).
    normal_mean_variance, ///< Change in both: as above with segment means (`cpt.meanvar`).
    poisson,              ///< Change in a Poisson rate, counts: \f$2S\ln(m/S)\f$ with segment sum S (`test.stat = "Poisson"`).
    exponential,          ///< Change in an exponential rate, positive data: \f$2m\ln(S/m)\f$ (`test.stat = "Exponential"`).
};

/// @brief Penalty per added changepoint. With \f$d\f$ parameters changing at each changepoint
/// (2 for normal_mean_variance, otherwise 1), these are `changepoint`'s `penalty` choices.
/// @par Example
/// @code
/// const auto p = dm::stats::ChangePenalty::mbic;
/// std::println("{}", p == dm::stats::ChangePenalty::bic); // false
/// @endcode
enum class ChangePenalty {
    mbic,         ///< Modified BIC (Zhang and Siegmund): \f$(d+2)\ln n\f$ plus \f$\ln m\f$ for each segment of length m.
    bic,          ///< \f$(d+1)\ln n\f$.
    aic,          ///< \f$2(d+1)\f$.
    hannan_quinn, ///< \f$2(d+1)\ln\ln n\f$.
    manual,       ///< `penalty_value`.
    none,         ///< Zero (every admissible split).
};

/// @brief Options for pelt_changepoints().
/// @par Example
/// @code
/// const dm::stats::PeltOptions options{.cost = dm::stats::ChangeCost::normal_mean_variance};
/// std::println("{}", options.min_segment_length); // 0
/// @endcode
struct PeltOptions {
    ChangeCost cost{ChangeCost::normal_mean};      ///< Segment model.
    ChangePenalty penalty{ChangePenalty::mbic};    ///< Penalty rule.
    double penalty_value{0};                       ///< Penalty for ChangePenalty::manual.
    std::size_t min_segment_length{0};             ///< Minimum segment length (0: 1 for normal_mean, otherwise 2).
    double noise_variance{1};                      ///< Known \f$\sigma^2\f$ for normal_mean (R assumes 1).
    std::optional<double> known_mean{};            ///< Fixed mean for normal_variance (default: the overall mean).
};

/// @brief Segmentation found by PELT.
/// @par Example
/// @code
/// const dm::stats::ChangepointResult r;
/// std::println("{}", r.changepoints.size()); // 0
/// @endcode
struct ChangepointResult {
    /// Zero-based index of the first observation of each new segment, increasing. Equal to R's
    /// `cpts()` (the one-based last index of each segment but the final one).
    std::vector<std::size_t> changepoints;
    std::vector<double> means;     ///< Mean of each segment.
    std::vector<double> variances; ///< Maximum-likelihood variance (divisor m) of each segment.
    double penalty{};              ///< Penalty per changepoint that was used.
    double cost{};                 ///< Minimized total: segment costs plus penalties (with R's \f$-\beta\f$ offset).
};

/// @brief Optimal penalized segmentation by PELT, as R's `changepoint::cpt.mean`, `cpt.var` and
/// `cpt.meanvar` with `method = "PELT"`.
///
/// Minimizes \f$\sum_{j}\big[C(x_{\tau_{j-1}+1:\tau_j})+\beta\big]\f$ over all segmentations with
/// segments of at least the minimum length, by optimal partitioning with Killick, Fearnhead and
/// Eckley's pruning: a candidate last changepoint \f$\tau\f$ is dropped once
/// \f$F(\tau)+C(x_{\tau+1:t})>F(t)\f$, which is exact for these costs. Expected cost is linear
/// in n when the number of changepoints grows with n. The normal mean cost assumes the noise
/// variance given in the options; scale the data (for example by a MAD estimate from first
/// differences) when it is unknown.
/// @param x Series (finite; nonnegative integers for poisson, positive for exponential).
/// @param options Cost, penalty, minimum segment length and known parameters.
/// @return Changepoints, per-segment means and variances, the penalty and the minimized cost.
/// @throws std::invalid_argument For empty or invalid data, a negative or nonfinite manual penalty,
/// a minimum segment length of 1 for a variance cost, or a nonpositive noise variance.
/// @par Example
/// @code
/// std::vector<double> x;
/// for (int t = 0; t < 300; ++t)
///     x.push_back((t < 100 ? 0.0 : t < 200 ? 3.0 : 1.0) + ((t * 37) % 11 - 5) / 5.0);
/// const auto r = dm::stats::pelt_changepoints(x);
/// std::println("{} {}", r.changepoints[0], r.changepoints[1]); // 100 200
/// @endcode
/// @par References
/// - R. Killick, P. Fearnhead and I. A. Eckley (2012). "Optimal detection of changepoints with a
///   linear computational cost." JASA 107(500), 1590-1598. doi:10.1080/01621459.2012.737745
/// - N. R. Zhang and D. O. Siegmund (2007). "A modified Bayes information criterion with
///   applications to the analysis of comparative genomic hybridization data." Biometrics 63(1),
///   22-32. doi:10.1111/j.1541-0420.2006.00662.x
[[nodiscard]] ChangepointResult pelt_changepoints(const std::vector<double>& x, const PeltOptions& options = {});

} // namespace datamunge::stats
