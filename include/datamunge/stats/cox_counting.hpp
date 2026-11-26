#pragma once

/// @file cox_counting.hpp
/// @brief Cox regression on counting-process (start, stop] records (delayed entry, time-varying covariates, strata) and proportional-hazards diagnostics.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/survival.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief One (start, stop] interval of a subject's follow-up with covariates constant on it.
/// @par Example
/// @code
/// const dm::stats::CountingRecord r{.start = 0, .stop = 2.5, .event = true};
/// std::println("{}", r.stop - r.start); // 2.5
/// @endcode
struct CountingRecord {
    double start{0};    ///< Entry time; the record is at risk on (start, stop].
    double stop{0};     ///< Exit time, greater than start.
    bool event{false};  ///< True if the interval ends with an event.
};

/// @brief A chi-square test with its degrees of freedom and upper-tail p-value.
/// @par Example
/// @code
/// const dm::stats::ChiSquareTest t{.statistic = 3.84, .degrees_of_freedom = 1};
/// std::println("{}", t.degrees_of_freedom); // 1
/// @endcode
struct ChiSquareTest {
    double statistic{};               ///< Chi-square statistic.
    std::size_t degrees_of_freedom{}; ///< Degrees of freedom.
    double p_value{1};                ///< Upper-tail p-value.
};

/// @brief Options for fit_cox_counting().
/// @par Example
/// @code
/// const dm::stats::CoxCountingOptions o{.ties = dm::stats::CoxTies::breslow};
/// std::println("{}", o.max_iterations); // 30
/// @endcode
struct CoxCountingOptions {
    CoxTies ties{CoxTies::efron};  ///< Tie approximation and baseline hazard increments.
    std::size_t max_iterations{30}; ///< Maximum Newton updates.
    double tolerance{1e-9};        ///< Relative change in the log partial likelihood that stops iteration.
    std::vector<int> strata{};     ///< Optional stratum label per record (empty: one stratum).
    std::vector<int> clusters{};   ///< Optional subject/cluster label per record; enables the robust variance.
};

/// @brief Baseline cumulative hazard of one stratum at the reference covariates.
/// @par Example
/// @code
/// const dm::stats::CoxBaseline b{.stratum = 2};
/// std::println("{}", b.event_times.empty()); // true
/// @endcode
struct CoxBaseline {
    int stratum{};                         ///< Stratum label (0 when unstratified).
    std::vector<double> event_times;       ///< Distinct event times in the stratum, increasing.
    std::vector<double> cumulative_hazard; ///< \f$H_{0s}(t)\f$ at each event time.
};

/// @brief Fitted counting-process Cox model, laid out like R's `coxph` summary.
/// @par Example
/// @code
/// const dm::stats::CoxCountingFit fit;
/// std::println("{}", fit.coefficients.size()); // 0
/// @endcode
struct CoxCountingFit {
    std::vector<double> coefficients;        ///< Log hazard ratios per unit of each covariate column.
    std::vector<double> standard_errors;     ///< Square roots of the diagonal of the inverse information.
    linalg::DenseMatrix<double> covariance;  ///< Inverse observed information at the estimate.
    std::vector<double> reference;           ///< Column means of the covariate rows, defining the baseline.
    double log_likelihood{};                 ///< Log partial likelihood at the estimate.
    double null_log_likelihood{};            ///< Log partial likelihood at zero.
    ChiSquareTest likelihood_ratio;          ///< \f$2(\ell(\hat\beta)-\ell(0))\f$.
    ChiSquareTest wald;                      ///< \f$\hat\beta^\top I(\hat\beta)\hat\beta\f$.
    ChiSquareTest score;                     ///< \f$U(0)^\top I(0)^{-1}U(0)\f$.
    std::size_t iterations{};                ///< Newton updates taken.
    bool converged{false};                   ///< True if the tolerance was met.
    std::size_t events{};                    ///< Number of events.
    std::vector<CoxBaseline> baselines;      ///< One baseline per stratum, by increasing label.
    linalg::DenseMatrix<double> robust_covariance; ///< Cluster-robust sandwich \f$D^\top D\f$ from summed dfbeta residuals (empty without clusters).
    std::vector<double> robust_standard_errors;    ///< Square roots of its diagonal (empty without clusters).
    ChiSquareTest robust_wald;                     ///< \f$\hat\beta^\top V_{\mathrm{robust}}^{-1}\hat\beta\f$ (zero without clusters).
    std::size_t clusters{};                        ///< Number of distinct clusters (zero without clusters).

    /// @brief Survival beyond a time for a covariate profile in a stratum, \f$\exp\{-H_{0s}(t)e^{(x-\bar x)^\top\hat\beta}\}\f$.
    /// @param time Nonnegative time.
    /// @param covariates One value per coefficient.
    /// @param stratum Stratum label (0 when unstratified).
    /// @return Predicted survival probability.
    /// @throws std::invalid_argument For a negative time, wrong covariate length or unknown stratum.
    /// @par Example
    /// @code
    /// const auto fit = dm::stats::fit_cox_counting(
    ///     {{0, 1, true}, {0, 2, true}, {0, 3, false}, {1, 4, true}},
    ///     dm::linalg::DenseMatrix<double>(4, 1, {1, 0, 1, 0}));
    /// std::println("{}", fit.predict_survival(0, {1})); // 1
    /// @endcode
    [[nodiscard]] double predict_survival(double time, const std::vector<double>& covariates,
                                          int stratum = 0) const;
};

/// @brief Fits a Cox model to counting-process records, as R's `coxph(Surv(start, stop, event) ~ x)`.
///
/// A record is in the risk set at event time \f$t\f$ when start \f$<t\le\f$ stop, so delayed entry,
/// gaps and covariates that change over time are expressed by splitting each subject's follow-up
/// into records. Ties use Efron's or Breslow's approximation. Risk-set sums are updated in a
/// sweep over sorted stop and start times (recomputed from scratch if subtraction would lose
/// precision), so each likelihood evaluation costs \f$O(np^2)\f$ after an \f$O(n\log n)\f$ sort.
/// With strata, each stratum has its own baseline hazard and risk sets, and the coefficients are
/// shared, as `coxph(... + strata(s))`; the likelihood is the sum over strata.
/// The maximizer is found by Newton's method with step halving from \f$\beta=0\f$. Subjects
/// split into several records need no identifier for the estimate: records act independently in
/// the partial likelihood. For recurrent events (the Andersen–Gill model) or other correlated
/// records, pass `clusters` to also get the robust variance of Lin and Wei (1989), as
/// `coxph(... + cluster(id))`. Each record's score residual (with Efron's tie fractions, as
/// survival's `coxscore`) times the inverse information gives its dfbeta. These are summed within
/// clusters, and the robust covariance is \f$D^\top D\f$.
/// @param records One interval per row of covariates.
/// @param covariates Rows of covariate values (no intercept), finite.
/// @param options Ties, iteration limit, tolerance, optional strata and optional clusters.
/// @return Coefficients, model-based (and, with clusters, robust) covariance, likelihood, the global tests and the baseline hazard.
/// @throws std::invalid_argument For mismatched sizes (including strata and clusters), invalid intervals or covariates, or no events.
/// @throws std::runtime_error If the information matrix is singular (for example a constant
/// covariate or monotone likelihood).
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_cox_counting(
///     {{0, 1, true}, {0, 2, true}, {0, 3, false}, {1, 4, true}, {0.5, 5, true}},
///     dm::linalg::DenseMatrix<double>(5, 1, {1, 0, 1, 0, 0}));
/// std::println("{:.3f}", fit.coefficients[0]); // 0.549
/// @endcode
/// @par References
/// - P. K. Andersen and R. D. Gill (1982). "Cox's regression model for counting processes: a large
///   sample study." Annals of Statistics 10(4), 1100-1120. doi:10.1214/aos/1176345976
/// - T. M. Therneau and P. M. Grambsch (2000). Modeling Survival Data: Extending the Cox Model.
///   Springer. doi:10.1007/978-1-4757-3294-8
/// - D. Y. Lin and L. J. Wei (1989). "The robust inference for the Cox proportional hazards model."
///   JASA 84(408), 1074-1078. doi:10.1080/01621459.1989.10478874
/// @see cox_ph(), nelson_aalen()
[[nodiscard]] CoxCountingFit fit_cox_counting(const std::vector<CountingRecord>& records,
                                              const linalg::DenseMatrix<double>& covariates,
                                              const CoxCountingOptions& options = {});

/// @brief Time scale g(t) on which proportional-hazards tests look for a trend, as `cox.zph(transform =)`.
/// @par Example
/// @code
/// const auto g = dm::stats::TimeTransform::km;
/// std::println("{}", g == dm::stats::TimeTransform::rank); // false
/// @endcode
enum class TimeTransform {
    km,       ///< \f$1-\hat S(t-)\f$ from the pooled Kaplan–Meier curve (R's default).
    rank,     ///< Rank of the time among all records (average ranks for ties).
    identity, ///< The time itself.
    log,      ///< Natural logarithm of the time.
};

/// @brief Score tests of proportional hazards and scaled Schoenfeld residuals, as R's `cox.zph`.
/// @par Example
/// @code
/// const dm::stats::ProportionalHazardsTest t;
/// std::println("{}", t.terms.size()); // 0
/// @endcode
struct ProportionalHazardsTest {
    std::vector<ChiSquareTest> terms;           ///< One 1-df test per covariate column.
    ChiSquareTest global;                       ///< Joint test of all columns.
    std::vector<std::size_t> event_records;     ///< Record index of each residual row (rows by stratum, time, index).
    std::vector<int> event_strata;              ///< Stratum label of each row.
    std::vector<double> event_times;            ///< Event time of each row.
    std::vector<double> transformed_times;      ///< \f$g(t)\f$ of each row, before centering.
    linalg::DenseMatrix<double> schoenfeld;     ///< Schoenfeld residuals \f$x_i-\bar x(t_i)\f$, events by covariates.
    linalg::DenseMatrix<double> scaled_schoenfeld; ///< \f$\hat\beta+d\,I^{-1}r_i\f$; NaN where a column is constant in the stratum.
};

/// @brief Tests the proportional-hazards assumption of a fitted counting-process Cox model, as R's `cox.zph`.
///
/// For each covariate, a score test at \f$(\hat\beta,0)\f$ of adding the time-varying term
/// \f$\gamma\,x\,g(t)\f$, where \f$g\f$ is the transformed time centered over events. It uses the joint
/// information of \f$\beta\f$ and \f$\gamma\f$, so each test adjusts for estimating \f$\beta\f$. The global
/// test adds all such terms. The scaled Schoenfeld residuals \f$\hat\beta_k+\{d\,I^{-1}r_i\}_k\f$ have
/// expectation close to \f$\beta_k(t_i)\f$, so smoothing them against \f$g(t)\f$ shows how a coefficient
/// drifts over time (Grambsch and Therneau 1994).
/// @param records Intervals used in the fit.
/// @param covariates Covariate rows used in the fit.
/// @param fit Result of fit_cox_counting() on the same data and options.
/// @param options The options used for the fit (ties and strata).
/// @param transform Time scale for the tests and residuals.
/// @return Per-term and global tests, residuals and their event times.
/// @throws std::invalid_argument For inconsistent inputs, or the log transform with a stop time at or below 0.
/// @throws std::runtime_error If the joint information is singular.
/// @par Example
/// @code
/// const std::vector<dm::stats::CountingRecord> r{{0, 1, true}, {0, 2, true}, {0, 3, false},
///                                                {0, 4, true}, {0, 5, true}, {0, 6, true}};
/// const dm::linalg::DenseMatrix<double> x(6, 1, {1, 0, 1, 0, 1, 0});
/// const auto t = dm::stats::test_proportional_hazards(r, x, dm::stats::fit_cox_counting(r, x));
/// std::println("{:.3f}", t.global.statistic); // 0.022
/// @endcode
/// @par References
/// - P. M. Grambsch and T. M. Therneau (1994). "Proportional hazards tests and diagnostics based on
///   weighted residuals." Biometrika 81(3), 515-526. doi:10.1093/biomet/81.3.515
/// - D. Schoenfeld (1982). "Partial residuals for the proportional hazards regression model."
///   Biometrika 69(1), 239-241. doi:10.1093/biomet/69.1.239
/// @see fit_cox_counting()
[[nodiscard]] ProportionalHazardsTest test_proportional_hazards(const std::vector<CountingRecord>& records,
                                                                const linalg::DenseMatrix<double>& covariates,
                                                                const CoxCountingFit& fit,
                                                                const CoxCountingOptions& options = {},
                                                                TimeTransform transform = TimeTransform::km);

} // namespace datamunge::stats
