#pragma once

/// @file arellano_bond.hpp
/// @brief Difference GMM and residual serial correlation tests for dynamic panels.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Timing assumption for each supplied level regressor.
/// @par Example
/// @code
/// const auto timing = dm::stats::ArellanoBondRegressor::Predetermined;
/// std::println("{}", timing == dm::stats::ArellanoBondRegressor::Predetermined); // true
/// @endcode
enum class ArellanoBondRegressor {
    StrictlyExogenous, ///< Instrument its first difference with itself.
    Predetermined,     ///< Instrument its first difference with levels lagged at least one period.
    Endogenous         ///< Instrument its first difference with levels lagged at least two periods.
};

/// @brief GMM weighting step used for the returned coefficients.
/// @par Example
/// @code
/// const auto step = dm::stats::ArellanoBondStep::Two;
/// std::println("{}", step == dm::stats::ArellanoBondStep::Two); // true
/// @endcode
enum class ArellanoBondStep {
    One, ///< Use the covariance implied by differencing spherical level errors.
    Two  ///< Reweight using first-step moments and apply Windmeijer's covariance correction.
};

/// @brief Instrument selection, numerical rank and residual test controls.
/// @par Example
/// @code
/// const dm::stats::ArellanoBondOptions options{.maximum_lag=3};
/// std::println("{}", options.maximum_lag); // 3
/// @endcode
struct ArellanoBondOptions {
    ArellanoBondStep step{ArellanoBondStep::One}; ///< Weighting step used for the estimates.
    bool collapse{true};                        ///< Share lagged instruments across equation periods.
    bool time_effects{false};                   ///< Include a separate differenced intercept for every usable period.
    std::size_t minimum_outcome_lag{2};         ///< Earliest allowed outcome-level instrument lag, at least two.
    std::size_t maximum_lag{0};                 ///< Largest lagged-level instrument lag, or zero for all.
    std::size_t max_serial_order{2};            ///< Largest residual test lag, or zero to skip testing.
    double relative_tolerance{1e-10};           ///< Relative numerical rank cutoff for weights and moment design.
    std::vector<ArellanoBondRegressor> regressor_types; ///< Timing of each supplied regressor. Empty uses strict exogeneity.
};

/// @brief Arellano–Bond serial correlation test on the first-step differenced residuals.
/// @par Example
/// @code
/// const dm::stats::ArellanoBondSerialTest test{2,150,-.2,.84};
/// std::println("{}", test.lag); // 2
/// @endcode
struct ArellanoBondSerialTest {
    std::size_t lag;               ///< Separation in actual panel periods.
    std::size_t pairs;             ///< Within-unit residual pairs used.
    double statistic;              ///< Signed normal test statistic.
    double p_value;                ///< Two-sided normal tail probability.
};

/// @brief Hansen test of overidentifying restrictions using the second-step weight.
/// @par Example
/// @code
/// const dm::stats::ArellanoBondOveridentification test{2.0, 2, .3679};
/// std::println("{}", test.df); // 2
/// @endcode
struct ArellanoBondOveridentification {
    double statistic; ///< Weighted quadratic form of the fitted moments.
    std::size_t df;   ///< Instrument rank minus the number of coefficients.
    double p_value;   ///< Upper chi-square tail probability.
};

/// @brief Coefficient estimates, differenced fitted values and specification tests.
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_arellano_bond({1,2,2.2,3, 0,1,2,3.3, 2,1,2.5,2.8, 4,2,4,3.5},
///     {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4}, {0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3});
/// std::println("{}", fit.equation_rows.size()); // 8
/// @endcode
struct ArellanoBondFit {
    std::vector<double> coefficients;         ///< Lagged outcome, supplied regressors, then optional sorted period effects.
    linalg::DenseMatrix<double> covariance;  ///< Cluster-robust coefficient covariance.
    std::vector<std::int64_t> time_levels;    ///< Coefficient labels for the optional period effects.
    std::vector<std::size_t> equation_rows;   ///< Original row index for each differenced equation, in original row order.
    std::vector<double> fitted;               ///< Fitted first differences ordered by equation_rows.
    std::vector<double> residuals;            ///< First differences minus fitted values.
    std::size_t units;                        ///< Units with at least one usable equation.
    std::size_t instrument_columns;          ///< Instrument columns before numerical rank truncation.
    std::size_t instrument_rank;             ///< Retained instrument weight rank for the selected step.
    std::vector<ArellanoBondSerialTest> serial_correlation; ///< Tests with observed pairs and positive estimated variance, in lag order.
    std::optional<ArellanoBondOveridentification> hansen;   ///< Second-step test when the instrument rank exceeds the coefficient count.
};

/// @brief Fits a first-differenced panel model with one lagged outcome and additional regressors.
///
/// Each equation needs outcomes at three consecutive integer periods of the
/// same unit. Gaps can remove equations and eligible instrument levels.
/// Earlier eligible levels remain usable across gaps. The first step uses
/// twice the identity minus the adjacent-period off-diagonals for the
/// transformed error covariance. The second step uses the first-step
/// unit score cross-products. Serial correlation tests use first-step
/// residuals and account for first-step coefficient estimation.
/// @param predictors Finite level regressors, one row per outcome, without a constant or a lagged-outcome column.
/// @param response Finite observed level outcomes.
/// @param unit Unit label for each row.
/// @param time Integer period label for each row, unique within each unit.
/// @param options Instrument choices, rank tolerance and test controls.
/// @return GMM estimates and diagnostics.
/// @throws std::invalid_argument Invalid controls, shapes, nonfinite data, duplicate periods or too few usable equations.
/// @throws std::domain_error The retained moments do not identify all coefficients, or the estimated second-step weight is singular.
/// @throws std::length_error A dense workspace exceeds storage capacity.
/// @throws std::runtime_error A required numerical factorization failed to converge.
/// @throws std::overflow_error A fitted result exceeds the finite reporting range.
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_arellano_bond(dm::linalg::DenseMatrix<double>(16,0),
///     {1,2,2.2,3,0,1,2,3.3,2,1,2.5,2.8,4,2,4,3.5},
///     {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4}, {0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3});
/// std::println("{}", fit.units); // 4
/// @endcode
/// @par References
/// - Arellano and Bond (1991), https://doi.org/10.2307/2297968
/// - Windmeijer (2005), https://doi.org/10.1016/j.jeconom.2004.02.005
[[nodiscard]] ArellanoBondFit fit_arellano_bond(const linalg::DenseMatrix<double>& predictors,
    const std::vector<double>& response, const std::vector<std::int64_t>& unit, const std::vector<std::int64_t>& time,
    const ArellanoBondOptions& options = {});

/// @brief Fits a dynamic panel with a lagged outcome and optional period effects.
/// @param response Finite observed level outcomes.
/// @param unit Unit label for each row.
/// @param time Integer period label for each row, unique within each unit.
/// @param options Instrument choices, rank tolerance and test controls.
/// @return GMM estimates and diagnostics, as in the model with supplied regressors.
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_arellano_bond({1,2,2.2,3,0,1,2,3.3,2,1,2.5,2.8,4,2,4,3.5},
///     {1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4}, {0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3});
/// std::println("{}", fit.coefficients.size()); // 1
/// @endcode
[[nodiscard]] ArellanoBondFit fit_arellano_bond(const std::vector<double>& response,
    const std::vector<std::int64_t>& unit, const std::vector<std::int64_t>& time,
    const ArellanoBondOptions& options = {});

} // namespace datamunge::stats
