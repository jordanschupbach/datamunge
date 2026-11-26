#pragma once

/// @file msm.hpp
/// @brief Marginal structural models: time-varying inverse probability weights and weighted outcome fits.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::causal {

/// @brief Pooled logistic models for one time-varying process (treatment or censoring).
///
/// Rows are person-time intervals, grouped by individual and ordered by time within individual.
/// The denominator model conditions on the full measured history (including time-varying
/// confounders); the numerator model on baseline covariates and time only, which stabilizes the weights.
/// @par Example
/// @code
/// const dm::causal::LongitudinalWeightModel m{.event = {0, 1}};
/// std::println("{}", m.event.size()); // 2
/// @endcode
struct LongitudinalWeightModel {
    std::vector<int> event;                  ///< Modeled indicator (0 or 1) of each row, e.g. treatment started or censored.
    linalg::DenseMatrix<double> numerator;   ///< Numerator covariates (zero columns: intercept only).
    linalg::DenseMatrix<double> denominator; ///< Denominator covariates.
};

/// @brief Rows that contribute to a weight model.
enum class WeightRiskSet {
    all_rows,           ///< Every row (treatment can switch on and off).
    until_first_event,  ///< Rows up to and including each individual's first event (monotone treatment or censoring).
};

/// @brief Options for longitudinal_weights().
/// @par Example
/// @code
/// const dm::causal::LongitudinalWeightOptions o{.truncation = 0.01};
/// std::println("{}", o.truncation); // 0.01
/// @endcode
struct LongitudinalWeightOptions {
    WeightRiskSet risk_set{WeightRiskSet::until_first_event}; ///< Which rows are modeled and contribute factors.
    bool stabilized{true};  ///< Use the numerator model; false gives unstabilized weights (numerator 1).
    double truncation{0.0}; ///< Truncate weights at the p and 1 - p quantiles (type 7); 0 disables.
};

/// @brief Cumulative inverse probability weights for one process.
/// @par Example
/// @code
/// const dm::causal::LongitudinalWeights w{.weights = {1.0, 0.8}};
/// std::println("{}", w.weights[1]); // 0.8
/// @endcode
struct LongitudinalWeights {
    std::vector<double> weights;                  ///< Cumulative product of the per-row factors within each individual.
    std::vector<double> truncated;                ///< Weights after quantile truncation (equal to weights when disabled).
    std::vector<double> numerator_probability;    ///< Fitted P(event = 1) from the numerator model (NaN outside the risk set).
    std::vector<double> denominator_probability;  ///< Fitted P(event = 1) from the denominator model (NaN outside the risk set).
    std::vector<bool> at_risk;                    ///< Rows used in the models.
    std::vector<double> numerator_coefficients;   ///< Intercept and slopes of the numerator model (empty if unstabilized).
    std::vector<double> denominator_coefficients; ///< Intercept and slopes of the denominator model.
    double mean{};                                ///< Mean of the (untruncated) weights; near 1 for well-specified stabilized weights.
    double min{};                                 ///< Smallest weight.
    double max{};                                 ///< Largest weight.
};

/// @brief Stabilized inverse probability weights for a time-varying treatment or censoring process.
///
/// Fits pooled logistic regressions of the event on the numerator and denominator covariates over the
/// risk-set rows. Each risk-set row contributes the factor \f$p_{num}/p_{den}\f$ when the event occurs
/// and \f$(1-p_{num})/(1-p_{den})\f$ otherwise; rows outside the risk set contribute 1. The weight of a
/// row is the cumulative product of factors up to and including it, as in R's `ipwtm(family =
/// "binomial")`. Multiply the treatment and censoring weights to obtain the weights of a marginal
/// structural model.
/// @param id Individual of each row; each individual's rows must be contiguous and in time order.
/// @param model Event indicators and covariate matrices.
/// @param options Risk set, stabilization and truncation.
/// @return Weights, truncated weights, fitted probabilities and model coefficients.
/// @throws std::invalid_argument For mismatched sizes, non-contiguous individuals, invalid indicators,
/// non-finite covariates or a truncation outside [0, 0.5).
/// @throws std::runtime_error When a logistic fit fails or separates the outcomes.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> l(7, 1, {0, 1, 0, 1, 0, 1, 0});
/// const auto w = dm::causal::longitudinal_weights(
///     {1, 1, 1, 2, 2, 2, 2},
///     {.event = {0, 0, 1, 0, 0, 1, 0}, .numerator = dm::linalg::DenseMatrix<double>(7, 0), .denominator = l});
/// std::println("{} {}", w.at_risk[6], w.weights[6] == w.weights[5]); // false true
/// @endcode
/// @par References
/// - J. M. Robins, M. A. Hernan and B. Brumback (2000). Marginal structural models and causal
///   inference in epidemiology. Epidemiology 11(5), 550-560. doi:10.1097/00001648-200009000-00011
/// - S. R. Cole and M. A. Hernan (2008). Constructing inverse probability weights for marginal
///   structural models. American Journal of Epidemiology 168(6), 656-664. doi:10.1093/aje/kwn164
/// - W. M. van der Wal and R. B. Geskus (2011). ipw: An R package for inverse probability weighting.
///   Journal of Statistical Software 43(13). doi:10.18637/jss.v043.i13
/// @see fit_marginal_structural_model()
[[nodiscard]] LongitudinalWeights longitudinal_weights(const std::vector<std::size_t>& id,
                                                       const LongitudinalWeightModel& model,
                                                       const LongitudinalWeightOptions& options = {});

/// @brief Outcome model family of a marginal structural model.
enum class MsmFamily {
    logistic, ///< Pooled logistic regression (discrete-time hazard for person-time event data).
    linear,   ///< Linear regression of a continuous outcome.
};

/// @brief Weighted marginal structural model fit with individual-clustered standard errors.
/// @par Example
/// @code
/// const dm::causal::MsmFit f{.coefficients = {0.1, -0.5}};
/// std::println("{}", f.coefficients[1]); // -0.5
/// @endcode
struct MsmFit {
    std::vector<double> coefficients;    ///< Intercept followed by one coefficient per design column.
    std::vector<double> standard_errors; ///< Cluster-robust standard errors (sandwich vcovCL, HC0 with G/(G-1)).
    std::vector<double> z_values;        ///< Coefficient over standard error.
    std::vector<double> p_values;        ///< Two-sided normal p-values.
    std::vector<double> lower;           ///< Lower normal confidence limits.
    std::vector<double> upper;           ///< Upper normal confidence limits.
    std::size_t clusters{};              ///< Number of individuals.
    std::size_t iterations{};            ///< IRLS iterations (1 for linear).
    bool converged{false};               ///< True when IRLS converged.
};

/// @brief Fits a weighted marginal structural model and clusters its standard errors by individual.
///
/// Solves the weighted (pooled) logistic or least-squares estimating equations
/// \f$\sum_i w_i x_i\{y_i-\mu(x_i^\top\beta)\}=0\f$ and uses the sandwich covariance clustered on the
/// individual, which accounts both for repeated rows and for the weights (treated as known, which is
/// conservative for estimated weights).
/// @param outcome Outcome of each row (0/1 for logistic).
/// @param design Covariates of the structural model in columns (an intercept is added).
/// @param weights Nonnegative weights, typically treatment times censoring weights.
/// @param id Individual of each row (any order).
/// @param family Logistic or linear.
/// @param confidence Level of the normal confidence intervals.
/// @return Coefficients with cluster-robust inference.
/// @throws std::invalid_argument For mismatched sizes, invalid values or fewer than two individuals.
/// @throws std::runtime_error For a rank-deficient weighted design or IRLS failure.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> a(6, 1, {0, 0, 0, 1, 1, 1});
/// const auto f = dm::causal::fit_marginal_structural_model({1, 2, 3, 3, 4, 5}, a, {1, 1, 1, 1, 1, 1},
///                                                          {1, 2, 3, 4, 5, 6},
///                                                          dm::causal::MsmFamily::linear);
/// std::println("{:.1f} {:.1f}", f.coefficients[0], f.coefficients[1]); // 2.0 2.0
/// @endcode
/// @see longitudinal_weights()
[[nodiscard]] MsmFit fit_marginal_structural_model(const std::vector<double>& outcome,
                                                   const linalg::DenseMatrix<double>& design,
                                                   const std::vector<double>& weights,
                                                   const std::vector<std::size_t>& id, MsmFamily family,
                                                   double confidence = 0.95);

} // namespace datamunge::causal
