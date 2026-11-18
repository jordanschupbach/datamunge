#pragma once

/// @file effects.hpp
/// @brief Average treatment effects for binary exposures and quasi-experiments.

#include <datamunge/linalg/dense_matrix.hpp>
#include <vector>

namespace datamunge::causal {

/// @brief Point estimates and overlap diagnostics for a binary-treatment analysis.
struct BinaryATE {
    double difference_in_means{}; ///< Unadjusted treated-minus-control mean.
    double g_computation{}; ///< Average contrast of the two outcome predictions.
    double ipw{}; ///< Inverse propensity weighted contrast with weights normalized within each arm.
    double aipw{}; ///< Augmented inverse probability weighted contrast.
    double min_propensity{}; ///< Smallest estimated probability of treatment.
    double max_propensity{}; ///< Largest estimated probability of treatment.
};

/// @brief Estimates an average treatment effect from supplied nuisance predictions.
/// @param outcome Finite observed outcomes.
/// @param treatment Treatment indicators (0 or 1), with both arms represented.
/// @param propensity Treatment probabilities strictly between zero and one.
/// @param control_mean Predicted outcomes under control, for all rows.
/// @param treated_mean Predicted outcomes under treatment, for all rows.
/// @return Raw and adjusted point estimates and propensity extrema.
/// @throws std::invalid_argument Invalid sizes, indicators or values.
/// @par Example
/// @code
/// const auto fit = dm::causal::binary_ate({1, 3}, {0, 1}, {.5, .5}, {1, 1}, {3, 3});
/// @endcode
/// @par References
/// - https://doi.org/10.1111/j.1541-0420.2005.00377.x
/// @see fit_binary_ate()
[[nodiscard]] BinaryATE binary_ate(const std::vector<double>& outcome, const std::vector<int>& treatment,
                                    const std::vector<double>& propensity,
                                    const std::vector<double>& control_mean,
                                    const std::vector<double>& treated_mean);

/// @brief Fits logistic propensity and arm-specific linear outcome models on the supplied rows.
/// @param outcome Finite observed outcomes.
/// @param treatment Binary indicators with both arms represented.
/// @param adjustment Finite numeric adjustment covariates in columns (empty for intercept-only fits).
/// @return Point estimates conditional on consistency, positivity and no unmeasured confounding.
/// @throws std::invalid_argument Invalid data.
/// @throws std::runtime_error Rank-deficient regressions or unstable propensity fit.
/// @par Example
/// @code
/// const auto fit = dm::causal::fit_binary_ate({1, 2, 3, 4}, {0, 0, 1, 1});
/// @endcode
/// @see binary_ate(), DAG::backdoor_adjustment()
[[nodiscard]] BinaryATE fit_binary_ate(const std::vector<double>& outcome, const std::vector<int>& treatment,
                                        const linalg::DenseMatrix<double>& adjustment = {});

/// @brief Estimates the treated effect from a two-period panel by difference-in-differences.
/// @param before Finite pretreatment outcomes.
/// @param after Finite posttreatment outcomes in the same unit order.
/// @param treatment Group indicators with both groups represented.
/// @return Difference of mean changes, identified under parallel trends and no anticipation.
/// @throws std::invalid_argument Invalid inputs.
/// @par Example
/// @code
/// const double effect = dm::causal::difference_in_differences({1, 2}, {3, 7}, {0, 1});
/// @endcode
/// @par References
/// - https://doi.org/10.1016/j.jeconom.2020.12.001 (two-group, two-period setup)
/// @see fit_binary_ate()
[[nodiscard]] double difference_in_differences(const std::vector<double>& before, const std::vector<double>& after,
                                                const std::vector<int>& treatment);

/// @brief Point estimate and relevance diagnostics for two-stage least squares.
struct InstrumentalEffect {
    double effect{}; ///< Coefficient of the endogenous exposure.
    double first_stage_f{}; ///< Classical partial F statistic for excluded instruments.
};

/// @brief Fits one endogenous exposure using instruments and optional exogenous controls.
/// @param outcome Finite responses.
/// @param exposure Finite endogenous regressor.
/// @param instruments Finite excluded instruments (one or more columns).
/// @param controls Finite included exogenous covariates in columns, or empty.
/// @return Exposure coefficient and classical first-stage partial F statistic.
/// @throws std::invalid_argument Invalid inputs or insufficient residual degrees of freedom.
/// @throws std::runtime_error Numerically rank-deficient regression.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> z(4, 1, {0, 0, 1, 1});
/// const auto fit = dm::causal::two_stage_least_squares({1, 3, 3, 7}, {0, 1, 1, 3}, z);
/// @endcode
/// @par References
/// - https://dash.harvard.edu/entities/publication/73120378-82c1-6bd4-e053-0100007fdf3b
/// @see fit_binary_ate()
[[nodiscard]] InstrumentalEffect two_stage_least_squares(const std::vector<double>& outcome,
    const std::vector<double>& exposure, const linalg::DenseMatrix<double>& instruments,
    const linalg::DenseMatrix<double>& controls = {});

} // namespace datamunge::causal
