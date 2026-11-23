#pragma once

/// @file panel_fixed_effects.hpp
/// @brief Linear regression with absorbed unit and period fixed effects.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief Panel dimensions used to form the clustered sandwich covariance.
/// @par Example
/// @code
/// const auto cluster = dm::stats::PanelClustering::UnitAndTime;
/// std::println("{}", cluster == dm::stats::PanelClustering::UnitAndTime); // true
/// @endcode
enum class PanelClustering {
    Unit,        ///< Sum scores within units.
    Time,        ///< Sum scores within periods.
    UnitAndTime  ///< Add unit and period covariances and subtract their joint-cell covariance.
};

/// @brief Clustering and numerical controls for fixed-effect absorption.
/// @par Example
/// @code
/// const dm::stats::PanelFixedEffectsOptions options{.clustering = dm::stats::PanelClustering::Time};
/// std::println("{}", options.finite_sample_adjustment); // true
/// @endcode
struct PanelFixedEffectsOptions {
    PanelClustering clustering{PanelClustering::Unit}; ///< Which panel dimensions are clusters.
    bool finite_sample_adjustment{true}; ///< Multiply each component by G/(G-1) times (n-1)/residual_df.
    double tolerance{1e-13};              ///< Relative residual tolerance for the fixed-effect solve.
    std::size_t max_iterations{0};       ///< Iteration limit per column. Zero uses max(1000, number of periods).
};

/// @brief Clustered Wald t test for a fixed-effects coefficient contrast.
/// @par Example
/// @code
/// const dm::stats::PanelContrastResult test{2.0, 0.5, 4.0, 8, 0.00395};
/// std::println("{:.1f}", test.standard_error); // 0.5
/// @endcode
struct PanelContrastResult {
    double estimate;          ///< Estimated linear combination of slopes.
    double standard_error;    ///< Cluster-robust standard error.
    double t_statistic;       ///< Signed t statistic for the contrast's deviation from the null value.
    std::size_t df;           ///< Cluster-count degrees of freedom used by the reference distribution.
    double p_value;           ///< Two-sided p-value.
};

/// @brief Fitted slopes, effects, and clustered inference for a two-way fixed-effects panel.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(9,1,{0,1,0,1,0,2,0,2,1});
/// const auto fit = dm::stats::fit_two_way_fixed_effects(x, {1,4,3,5,2,8,4,7,6},
///     {1,1,1,2,2,2,3,3,3}, {1,2,3,1,2,3,1,2,3});
/// std::println("{}", fit.residual_df); // 3
/// @endcode
struct PanelFixedEffectsFit {
    std::vector<double> coefficients;           ///< Slopes in predictor-column order.
    linalg::DenseMatrix<double> covariance;     ///< Cluster sandwich covariance, possibly indefinite for two-way clustering.
    std::vector<double> fitted;                 ///< Fitted outcomes in input-row order, including both effects.
    std::vector<double> residuals;              ///< Outcome minus fitted value in input-row order.
    std::vector<std::int64_t> unit_levels;      ///< Ascending observed unit labels.
    std::vector<std::int64_t> time_levels;      ///< Ascending observed period labels.
    std::vector<double> unit_effects;           ///< Unit intercepts in unit_levels order.
    std::vector<double> time_effects;           ///< Period intercepts in time_levels order, with the first period in each component set to zero.
    std::size_t connected_components;           ///< Components of the observed unit-by-period incidence graph.
    std::size_t fixed_effect_rank;              ///< Number of units plus number of periods minus connected_components.
    std::size_t residual_df;                    ///< Observations minus the absorbed-effect rank and number of slopes.
    std::size_t cluster_df;                     ///< Smallest selected marginal cluster count minus one.

    /// @brief Tests a linear combination of slopes using the clustered covariance.
    /// @param weights Finite weights in slope-column order.
    /// @param null_value Finite null value for the weighted slope sum.
    /// @return Estimate, standard error, t statistic, degrees of freedom and two-sided p-value.
    /// @throws std::invalid_argument An incompatible contrast or nonfinite null value.
    /// @throws std::domain_error The contrast variance is nonpositive.
    /// @throws std::overflow_error An estimate or standard error exceeds finite double range.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(9,1,{0,1,0,1,0,2,0,2,1});
    /// const auto fit = dm::stats::fit_two_way_fixed_effects(x, {1,4,3,5,2,8,4,7,6},
    ///     {1,1,1,2,2,2,3,3,3}, {1,2,3,1,2,3,1,2,3});
    /// std::println("{}", fit.test_contrast({1}).df); // 2
    /// @endcode
    [[nodiscard]] PanelContrastResult test_contrast(const std::vector<double>& weights, double null_value = 0) const;
};

/// @brief Fits an additive unit and period fixed-effects regression with clustered covariance.
///
/// Pass the observed panel rows, in any order. The model absorbs the intercept
/// and both sets of effects using the observed unit-period incidence matrix,
/// including unbalanced panels and repeated observations per cell. Predictors
/// need full column rank after absorption. Unit and period effects are identified
/// up to a constant within each connected component.
/// @param predictors Finite slope predictors without an intercept.
/// @param response Finite outcomes in input-row order.
/// @param unit Unit label for each outcome.
/// @param time Period label for each outcome.
/// @param options Clustering and fixed-effect solver controls.
/// @return Slopes, fitted outcomes, effects, covariance and degrees of freedom.
/// @throws std::invalid_argument Incompatible or nonfinite data, invalid controls, inadequate degrees of freedom or aliased predictors.
/// @throws std::length_error A working matrix exceeds storage capacity.
/// @throws std::runtime_error A numerical solver failed to converge.
/// @throws std::overflow_error A fitted result exceeds finite double range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(9,1,{0,1,0,1,0,2,0,2,1});
/// const auto fit = dm::stats::fit_two_way_fixed_effects(x, {1,4,3,5,2,8,4,7,6},
///     {1,1,1,2,2,2,3,3,3}, {1,2,3,1,2,3,1,2,3});
/// std::println("{}", fit.fixed_effect_rank); // 5
/// @endcode
/// @par References
/// - Gaure (2013), https://doi.org/10.1016/j.csda.2013.03.024
/// - Cameron, Gelbach and Miller (2011), https://doi.org/10.1198/jbes.2010.07136
[[nodiscard]] PanelFixedEffectsFit fit_two_way_fixed_effects(const linalg::DenseMatrix<double>& predictors,
    const std::vector<double>& response, const std::vector<std::int64_t>& unit, const std::vector<std::int64_t>& time,
    const PanelFixedEffectsOptions& options = {});

} // namespace datamunge::stats
