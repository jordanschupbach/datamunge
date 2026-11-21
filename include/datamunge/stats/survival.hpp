#pragma once

/// @file survival.hpp
/// @brief Right-censored survival estimates, rank scores and proportional hazards regression.

#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief A single right-censored follow-up record.
/// @par Example
/// @code
/// const dm::stats::SurvivalObservation censored{2.0, false};
/// @endcode
/// @see kaplan_meier()
struct SurvivalObservation {
    double time{0}; ///< Finite nonnegative elapsed time.
    bool event{false}; ///< True for an observed event, false for right censoring.
};

/// @brief Kaplan–Meier estimate at a distinct observed time.
/// @par Example
/// @code
/// const auto point = dm::stats::kaplan_meier({{1, true}, {2, false}}).points.front();
/// std::print("{}\n", point.survival); // 0.5
/// @endcode
/// @see KaplanMeierEstimate
struct KaplanMeierPoint {
    double time; ///< Follow-up time.
    std::size_t at_risk; ///< Number at risk immediately before this time.
    std::size_t events; ///< Observed events at this time.
    std::size_t censored; ///< Censored records at this time.
    double survival; ///< Right-continuous product-limit survival estimate.
    double standard_error; ///< Greenwood standard error, zero at survival zero or one.
};

/// @brief Owned product-limit results in ascending time order.
/// @par Example
/// @code
/// const auto fit = dm::stats::kaplan_meier({{1, true}, {2, false}});
/// std::print("{}\n", fit.median_time.value()); // 1
/// @endcode
/// @see kaplan_meier(), kaplan_meier_plot()
struct KaplanMeierEstimate {
    std::vector<KaplanMeierPoint> points; ///< One point per distinct event or censoring time.
    std::optional<double> median_time; ///< First time survival is at most 0.5, if reached.
};

/// @brief Estimates survival by the Kaplan–Meier product limit and Greenwood standard error.
///
/// Tied events and censorings share the same risk set. Events reduce survival,
/// then all records at the time leave the risk set. Assumes independent,
/// noninformative right censoring. Uses O(n log n) time and O(n) space.
/// @param observations Nonempty right-censored follow-up records.
/// @return Owned survival curve and optional median.
/// @throws std::invalid_argument If input is empty or a time is negative or nonfinite.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto fit = dm::stats::kaplan_meier({{1, true}, {2, false}, {3, true}});
/// std::print("{}\n", fit.points.back().survival); // 0
/// @endcode
/// @par References
/// - Kaplan and Meier (1958), Nonparametric Estimation from Incomplete Observations.
///   https://doi.org/10.1080/01621459.1958.10501452
/// @see kaplan_meier_plot(), prentice_wilcoxon_scores()
KaplanMeierEstimate kaplan_meier(const std::vector<SurvivalObservation>& observations);

/// @brief Plots one or more Kaplan–Meier curves with censoring markers.
/// @param estimates Nonempty curves returned by kaplan_meier().
/// @param labels One legend label per curve, or empty for generated labels.
/// @return Editable native step plot of survival probabilities.
/// @throws std::invalid_argument For empty curves or a label count mismatch.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto plot = dm::stats::kaplan_meier_plot({dm::stats::kaplan_meier({{1, true}, {2, false}})});
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/survival/html/plot.survfit.html
/// @see kaplan_meier()
graphics::RPlot kaplan_meier_plot(const std::vector<KaplanMeierEstimate>& estimates,
                                  const std::vector<std::string>& labels = {});

/// @brief Computes Prentice generalized Wilcoxon scores from pooled follow-up.
///
/// At event times updates the weight by multiplying by risk divided by
/// risk plus events. A censoring gets the accumulated weighted event fraction.
/// An event gets the same cumulative fraction minus the current weight.
/// Tied times use the shared risk set just before that time.
/// @param observations Nonempty finite nonnegative follow-up records.
/// @return One score per input record, in input order. Positive scores favour longer survival.
/// @throws std::invalid_argument If input is empty or times are invalid.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto scores = dm::stats::prentice_wilcoxon_scores({{1, true}, {2, false}});
/// std::print("{:.3f}\n", scores.front()); // -0.333
/// @endcode
/// @par References
/// - https://cran.r-project.org/web/packages/coin/refman/coin.html
/// @see prentice_wilcoxon_test(), kaplan_meier()
std::vector<double> prentice_wilcoxon_scores(const std::vector<SurvivalObservation>& observations);

/// @brief Compares independent right-censored groups using Prentice–Wilcoxon scores.
///
/// Pools observations to calculate scores. Uses the variance of a uniformly
/// permuted first-group score sum and a normal tail. Conditional calibration
/// assumes exchangeability of the observed follow-up records under the null.
/// @param x Nonempty first group of valid right-censored records.
/// @param y Nonempty independent second group of valid right-censored records.
/// @param alternative TwoSided (default), Less or Greater for shorter or longer first-group survival.
/// @return First-group score sum, group sizes in parameter1 and parameter2, and approximate p-value.
/// @throws std::invalid_argument For invalid data or alternative, or zero pooled score variance.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto result = dm::stats::prentice_wilcoxon_test({{1, true}, {2, true}}, {{3, true}, {4, false}});
/// @endcode
/// @par References
/// - https://cran.r-project.org/web/packages/coin/refman/coin.html
/// @see prentice_wilcoxon_scores(), kaplan_meier()
HypothesisTestResult prentice_wilcoxon_test(const std::vector<SurvivalObservation>& x,
                                           const std::vector<SurvivalObservation>& y,
                                           Alternative alternative = Alternative::TwoSided);

/// @brief Approximation used when multiple events share the same time.
/// @par Example
/// @code
/// const auto ties = dm::stats::CoxTies::breslow;
/// @endcode
/// @see cox_ph()
enum class CoxTies {
    efron, ///< Progressively removes fractions of the observed deaths from the risk set.
    breslow, ///< Uses the full risk set for every event at the same time.
};

/// @brief Fitted Cox regression and its baseline hazard at the covariate means.
/// @par Example
/// @code
/// const auto fit = dm::stats::cox_ph({{1, true}, {2, true}, {3, false}, {4, true}},
///                                    {{1}, {0}, {1}, {0}});
/// const auto survival = fit.predict_survival(2, {1});
/// @endcode
/// @see cox_ph()
struct CoxPHFit {
    CoxTies ties{}; ///< Approximation used for tied events and baseline increments.
    std::vector<double> coefficients; ///< Log hazard ratios per unit increase in each input column.
    std::vector<double> standard_errors; ///< Model-based standard errors in original covariate units.
    std::vector<double> reference; ///< Training covariate means defining the baseline.
    std::vector<double> event_times; ///< Distinct event times in ascending order.
    std::vector<double> log_baseline_cumulative_hazard; ///< Log cumulative hazard at each event time.
    double log_likelihood{}; ///< Maximized partial log likelihood.
    double null_log_likelihood{}; ///< Partial log likelihood with all coefficients zero.
    std::size_t iterations{}; ///< Number of accepted Newton updates.

    /// @brief Predicts the right-continuous survival probability for one profile.
    /// Returns one before the first event and uses the last estimated hazard after the last event.
    /// @param time Finite nonnegative follow-up time.
    /// @param covariates Finite values in training column order and original units.
    /// @return Predicted probability of surviving beyond time.
    /// @throws std::invalid_argument If time or covariates are invalid.
    /// @par Example
    /// @code
    /// const auto fit = dm::stats::cox_ph({{1, true}, {2, true}, {3, false}, {4, true}},
    ///                                    {{1}, {0}, {1}, {0}});
    /// std::println("{}", fit.predict_survival(0, {1})); // 1
    /// @endcode
    /// @see cox_ph()
    double predict_survival(double time, const std::vector<double>& covariates) const;
};

/// @brief Fits a Cox proportional hazards regression to right-censored observations, like R's coxph().
///
/// Each subject enters the risk set at time zero and has fixed covariates.
/// Subjects censored at an event time remain at risk at that time. Supply
/// numeric columns with no intercept. The likelihood is optimized by damped
/// Newton iteration and the standard errors use the inverse observed information.
/// @param observations Nonempty follow-up records with finite nonnegative times and at least one event.
/// @param covariates One finite numeric row per observation, with a common positive column count.
/// @param ties Efron by default or Breslow for ties and baseline hazard increments.
/// @return Owned fit in the original covariate units, with a baseline at the column means.
/// @throws std::invalid_argument For malformed input or absent events.
/// @throws std::runtime_error For singular information or failure to find a finite optimum within 50 updates.
///
/// @par Example
/// @code
/// const auto fit = dm::stats::cox_ph({{1, true}, {2, true}, {3, false}, {4, true}},
///                                    {{1}, {0}, {1}, {0}});
/// const auto hazard_ratio = std::exp(fit.coefficients[0]);
/// @endcode
/// @par References
/// - Cox (1972), Regression Models and Life-Tables. https://doi.org/10.1111/j.2517-6161.1972.tb00899.x
/// - Efron (1977), The Efficiency of Cox's Likelihood Function for Censored Data. https://doi.org/10.1080/01621459.1977.10480613
/// @see CoxPHFit, kaplan_meier()
CoxPHFit cox_ph(const std::vector<SurvivalObservation>& observations,
                const std::vector<std::vector<double>>& covariates, CoxTies ties = CoxTies::efron);

} // namespace datamunge::stats
