#pragma once

/// @file turnbull.hpp
/// @brief Turnbull nonparametric maximum likelihood estimate of a distribution from interval-censored data.

#include <datamunge/stats/aft.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Options for turnbull().
/// @par Example
/// @code
/// const dm::stats::TurnbullOptions o{.tolerance = 1e-8};
/// std::println("{}", o.max_iterations); // 100000
/// @endcode
struct TurnbullOptions {
    double tolerance{1e-10};          ///< Stop when the largest KKT violation \f$\max_j d_j/n-1\f$ falls below this.
    std::size_t max_iterations{100000}; ///< Self-consistency iteration limit.
};

/// @brief An innermost (Turnbull) interval with the probability mass assigned to it.
/// @par Example
/// @code
/// const dm::stats::TurnbullInterval i{.lower = 4, .upper = 5, .mass = .1};
/// std::println("{}", i.lower_closed); // false
/// @endcode
struct TurnbullInterval {
    double lower{};             ///< Left end.
    double upper{};             ///< Right end.
    bool lower_closed{false};   ///< True for a single exact point [t, t]; otherwise (lower, upper].
    double mass{};              ///< Estimated probability that the event lies in the interval.
};

/// @brief Turnbull estimate: innermost intervals, masses and survival between them.
/// @par Example
/// @code
/// const dm::stats::TurnbullEstimate e;
/// std::println("{}", e.converged); // false
/// @endcode
struct TurnbullEstimate {
    std::vector<TurnbullInterval> intervals; ///< Innermost intervals with positive mass (the support), increasing.
    std::size_t candidate_intervals{};       ///< Number of innermost intervals before dropping zero masses.
    double log_likelihood{};                 ///< \f$\sum_i\log\sum_{j\subseteq I_i}p_j\f$.
    double kkt_violation{};                  ///< \f$\max_j(d_j/n-1)\f$, at most 0 up to tolerance at the maximum.
    std::size_t iterations{};                ///< Iterations used.
    bool converged{false};                   ///< True if the tolerance was met.

    /// @brief Survival probability outside the innermost intervals, where it is uniquely determined.
    ///
    /// Inside an interval with positive mass the NPMLE does not say where the mass lies; the
    /// value returned there is the survival before the interval (an upper bound).
    /// @param time Time.
    /// @return \f$1-\sum\{p_j:\text{upper}_j\le t\}\f$, adjusted for an exact point at t.
    /// @par Example
    /// @code
    /// using C = dm::stats::CensoredTime;
    /// const auto e = dm::stats::turnbull({C::interval(0.5, 1), C::exact(2), C::right(3)});
    /// std::println("{:.4f}", e.survival(2.5)); // 0.3333
    /// @endcode
    [[nodiscard]] double survival(double time) const;
};

/// @brief Turnbull's nonparametric maximum likelihood estimate for interval-censored event times, as R's `interval::icfit`.
///
/// Observations are (lower, upper] with exact times [t, t], right censoring (t, ∞) and left
/// censoring (0, t]. The NPMLE puts all its mass on the innermost intervals: a left end followed
/// by a right end with no other end between them. Masses are found by self-consistency
/// (EM) iterations \f$p_j\leftarrow n^{-1}\sum_i\alpha_{ij}p_j/\sum_k\alpha_{ik}p_k\f$, stopped
/// when the Kuhn–Tucker conditions \f$d_j=\sum_i\alpha_{ij}/\sum_k\alpha_{ik}p_k\le n\f$ hold to the
/// tolerance. Because EM drives masses towards zero only geometrically, each convergence is
/// followed by dropping intervals with \f$d_j<n(1-10^{-6})\f$ (which cannot carry mass at the
/// maximum) and iterating again, reviving any dropped interval whose \f$d_j\f$ exceeds \f$n\f$.
/// Zero-mass intervals are omitted, as `icfit` does, so the result lists the support. With only exact and right-censored
/// data the result is the Kaplan–Meier estimate.
/// @param times Exact or censored times.
/// @param options Tolerance and iteration limit.
/// @return Innermost intervals with masses, log likelihood and convergence diagnostics.
/// @throws std::invalid_argument For empty input or an invalid censored time.
/// @par Example
/// @code
/// using C = dm::stats::CensoredTime;
/// const auto e = dm::stats::turnbull({C::interval(1, 3), C::interval(2, 5), C::left(2), C::right(4)});
/// std::println("{}", e.intervals.size()); // 2
/// @endcode
/// @par References
/// - B. W. Turnbull (1976). "The empirical distribution function with arbitrarily grouped,
///   censored and truncated data." JRSS B 38(3), 290-295. doi:10.1111/j.2517-6161.1976.tb01597.x
/// - M. P. Fay and P. A. Shaw (2010). "Exact and asymptotic weighted logrank tests for interval
///   censored data: the interval R package." Journal of Statistical Software 36(2).
///   doi:10.18637/jss.v036.i02
/// @see fit_aft(), kaplan_meier()
[[nodiscard]] TurnbullEstimate turnbull(const std::vector<CensoredTime>& times, const TurnbullOptions& options = {});

} // namespace datamunge::stats
