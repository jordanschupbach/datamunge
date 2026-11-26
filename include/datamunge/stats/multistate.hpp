#pragma once

/// @file multistate.hpp
/// @brief Aalen–Johansen estimation of state occupation and transition probabilities, competing-risk cumulative incidence and Gray's test.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief One interval of a subject's follow-up in a multi-state process.
/// @par Example
/// @code
/// const dm::stats::MultiStateRecord r{.start = 0, .stop = 3, .from = 0, .to = 1};
/// std::println("{}", r.to != r.from); // true
/// @endcode
struct MultiStateRecord {
    double start{0}; ///< Entry time; the record is at risk on (start, stop].
    double stop{0};  ///< Exit time, greater than start.
    int from{0};     ///< State occupied during the interval.
    int to{0};       ///< State entered at stop; equal to from when the interval ends by censoring.
    int id{-1};      ///< Subject identifier for clustering influence (negative: each record is its own subject).
};

/// @brief Aalen–Johansen estimate on the grid of distinct exit times.
/// @par Example
/// @code
/// const dm::stats::AalenJohansenEstimate e;
/// std::println("{}", e.states); // 0
/// @endcode
struct AalenJohansenEstimate {
    std::size_t states{};                        ///< Number of states.
    std::vector<double> initial;                 ///< Initial state distribution \f$p(0)\f$.
    std::vector<double> times;                   ///< Distinct exit times (events or censorings), increasing.
    linalg::DenseMatrix<double> probabilities;   ///< Times by states: \f$p_k(t)\f$.
    linalg::DenseMatrix<double> standard_errors; ///< Infinitesimal-jackknife standard errors of \f$p_k(t)\f$.
    linalg::DenseMatrix<double> at_risk;         ///< Times by states: number at risk in each state just before t.
    std::vector<linalg::DenseMatrix<double>> increments; ///< Per time, the transition-intensity increments \f$d\hat A(t)\f$.

    /// @brief State occupation probabilities at a time (right-continuous).
    /// @param time Time.
    /// @return \f$p(t)\f$, the initial distribution before the first time.
    /// @par Example
    /// @code
    /// const auto e = dm::stats::cumulative_incidence({{1, 1}, {2, 2}, {3, 0}});
    /// std::println("{:.4f}", e.state_probabilities(1.5)[1]); // 0.3333
    /// @endcode
    [[nodiscard]] std::vector<double> state_probabilities(double time) const;

    /// @brief Aalen–Johansen transition matrix \f$\hat P(s,t)=\prod_{s<u\le t}\{I+d\hat A(u)\}\f$.
    /// @param s Start time.
    /// @param t End time, at least s.
    /// @return States by states; entry (h, k) estimates P(in k at t | in h at s).
    /// @throws std::invalid_argument If t < s.
    /// @par Example
    /// @code
    /// const auto e = dm::stats::cumulative_incidence({{1, 1}, {2, 2}, {3, 0}});
    /// std::println("{:.4f}", e.transition_matrix(0, 2)(0, 2)); // 0.3333
    /// @endcode
    [[nodiscard]] linalg::DenseMatrix<double> transition_matrix(double s, double t) const;
};

/// @brief Aalen–Johansen estimator for a Markov multi-state process, as R's `survfit` with a factor status.
///
/// Transition intensities are estimated by Nelson–Aalen increments \f$d\hat A_{hk}(t)=dN_{hk}(t)/Y_h(t)\f$,
/// where \f$Y_h(t)\f$ counts records in state h with start < t ≤ stop. State occupation
/// probabilities follow \f$p(t)=p(t-)\{I+d\hat A(t)\}\f$ from the distribution of states among records
/// starting at the earliest entry time. Standard errors use the infinitesimal jackknife, summing
/// each subject's influence over its records (no small-sample factor). States are numbered 0 to K−1.
/// @param records Intervals with their states.
/// @param states Number of states K (0: one more than the largest state used).
/// @return Probabilities, standard errors, risk sets and intensity increments at each exit time.
/// @throws std::invalid_argument For empty input, invalid intervals or states.
/// @par Example
/// @code
/// const auto e = dm::stats::aalen_johansen({{0, 1, 0, 1}, {0, 2, 0, 0}, {1, 3, 1, 2}, {0, 4, 0, 2}});
/// std::println("{}", e.states); // 3
/// @endcode
/// @par References
/// - O. O. Aalen and S. Johansen (1978). "An empirical transition matrix for non-homogeneous Markov
///   chains based on censored observations." Scandinavian Journal of Statistics 5(3), 141-150.
/// - T. M. Therneau, C. Crowson and E. Atkinson (2024). "Multi-state models and competing risks."
///   Vignette of the R survival package.
/// @see cumulative_incidence(), nelson_aalen()
[[nodiscard]] AalenJohansenEstimate aalen_johansen(const std::vector<MultiStateRecord>& records,
                                                   std::size_t states = 0);

/// @brief A right-censored time with the cause of failure (0 for censoring).
/// @par Example
/// @code
/// const dm::stats::CompetingRiskObservation o{.time = 3, .cause = 2};
/// std::println("{}", o.cause); // 2
/// @endcode
struct CompetingRiskObservation {
    double time{}; ///< Follow-up time.
    int cause{0};  ///< 0 for censoring, otherwise the cause 1, 2, ...
};

/// @brief Competing-risks cumulative incidence by Aalen–Johansen, as `survfit(Surv(time, factor(cause)) ~ 1)`.
///
/// State 0 is event-free and state k is failure from cause k, so `probabilities(t, k)` is the
/// cumulative incidence of cause k and `probabilities(t, 0)` the overall survival.
/// @param observations Times and causes.
/// @return The Aalen–Johansen estimate with one absorbing state per cause.
/// @throws std::invalid_argument For empty input, negative causes or invalid times.
/// @par Example
/// @code
/// const auto e = dm::stats::cumulative_incidence({{1, 1}, {2, 2}, {2, 0}, {3, 1}, {4, 0}});
/// std::println("{:.4f}", e.probabilities(e.times.size() - 1, 1)); // 0.5000
/// @endcode
/// @see aalen_johansen()
[[nodiscard]] AalenJohansenEstimate cumulative_incidence(const std::vector<CompetingRiskObservation>& observations);

/// @brief Options for gray_test().
/// @par Example
/// @code
/// const dm::stats::GrayTestOptions o{.rho = 1};
/// std::println("{}", o.strata.empty()); // true
/// @endcode
struct GrayTestOptions {
    double rho{0};             ///< Weight exponent on \f$\{1-\hat F_k(t-)\}\f$ (0 is the usual test).
    std::vector<int> strata{}; ///< Optional stratum label per observation (empty: one stratum).
};

/// @brief Gray's test for one cause.
/// @par Example
/// @code
/// const dm::stats::GrayTest t;
/// std::println("{}", t.cause); // 0
/// @endcode
struct GrayTest {
    int cause{};                         ///< Cause tested.
    std::vector<double> score;           ///< Score for groups 2..G (in increasing label order), summed over strata.
    linalg::DenseMatrix<double> variance; ///< Its estimated covariance.
    double statistic{};                  ///< \f$s^\top V^{-1}s\f$.
    std::size_t degrees_of_freedom{};    ///< Groups minus one.
    double p_value{1};                   ///< Upper chi-square tail.
};

/// @brief Gray's K-sample test of equal cumulative incidence for each cause, as `cmprsk::cuminc`.
///
/// For cause k, compares the subdistribution hazards of the groups, weighting by
/// \f$\{1-\hat F_k(t-)\}^\rho\f$ on the pooled cumulative incidence. Failures from other causes stay in
/// the subdistribution risk set with weights from each group's censoring-adjusted survival. The
/// variance is Gray's (1988) estimator. Strata sum scores and variances over within-stratum tests.
/// This is a port of cmprsk's `crst` routine.
/// @param observations Times and causes (0 censored).
/// @param groups Group label per observation (at least two groups).
/// @param options Weight exponent and optional strata.
/// @return One test per cause present, in increasing cause order.
/// @throws std::invalid_argument For mismatched sizes, fewer than two groups, invalid data or no failures.
/// @throws std::runtime_error If a score variance matrix is singular.
/// @par Example
/// @code
/// const std::vector<dm::stats::CompetingRiskObservation> obs{{1, 1}, {2, 2}, {3, 1}, {4, 0},
///                                                           {2, 2}, {3, 2}, {5, 1}, {6, 0}};
/// const auto t = dm::stats::gray_test(obs, {0, 0, 0, 0, 1, 1, 1, 1});
/// std::println("{} {:.3f}", t.size(), t[0].statistic); // 2 2.203
/// @endcode
/// @par References
/// - R. J. Gray (1988). "A class of K-sample tests for comparing the cumulative incidence of a
///   competing risk." Annals of Statistics 16(3), 1141-1154. doi:10.1214/aos/1176350951
/// @see cumulative_incidence()
[[nodiscard]] std::vector<GrayTest> gray_test(const std::vector<CompetingRiskObservation>& observations,
                                              const std::vector<int>& groups, const GrayTestOptions& options = {});

} // namespace datamunge::stats
