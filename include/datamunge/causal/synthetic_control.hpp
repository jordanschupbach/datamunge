#pragma once

/// @file synthetic_control.hpp
/// @brief Synthetic control estimates (Abadie, Diamond and Hainmueller) with placebo inference.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::causal {

/// @brief Predictors and outcomes of one treated unit and its donor pool.
///
/// Rows of the outcome matrices are periods; columns of the donor matrices are donor units.
/// @par Example
/// @code
/// dm::causal::SyntheticControlData d;
/// d.treatment_period = 10;
/// std::println("{}", d.fit_periods.empty()); // true
/// @endcode
struct SyntheticControlData {
    std::vector<double> treated_predictors;           ///< k predictor values of the treated unit.
    linalg::DenseMatrix<double> donor_predictors;     ///< k-by-J predictor values of the donors.
    std::vector<double> treated_outcome;              ///< T outcome values of the treated unit.
    linalg::DenseMatrix<double> donor_outcomes;       ///< T-by-J outcome values of the donors.
    std::vector<std::size_t> fit_periods;             ///< Pre-treatment rows whose fit chooses V.
    std::size_t treatment_period{};                   ///< First post-treatment row.
};

/// @brief Options for synthetic_control().
/// @par Example
/// @code
/// const dm::causal::SyntheticControlOptions o{.restarts = 5};
/// std::println("{}", o.restarts); // 5
/// @endcode
struct SyntheticControlOptions {
    std::vector<double> predictor_weights; ///< Fixed nonnegative V diagonal; empty optimises V.
    std::size_t restarts{3};               ///< Nelder-Mead restarts from each starting V.
    std::size_t max_iterations{4000};      ///< Nelder-Mead iterations per run.
};

/// @brief A fitted synthetic control.
/// @par Example
/// @code
/// dm::causal::SyntheticControlFit f;
/// f.pre_rmspe = 0.1;
/// f.post_rmspe = 0.5;
/// std::println("{}", f.post_rmspe / f.pre_rmspe); // 5
/// @endcode
struct SyntheticControlFit {
    std::vector<double> weights;             ///< Donor weights W (nonnegative, summing to 1).
    std::vector<double> predictor_weights;   ///< Predictor importance V (nonnegative, summing to 1).
    std::vector<double> synthetic_predictors; ///< Predictor values of the synthetic unit, X0 W.
    std::vector<double> synthetic;           ///< Synthetic outcome path, Y0 W, for every period.
    std::vector<double> gap;                 ///< Treated minus synthetic outcome for every period.
    double predictor_loss{};  ///< V-weighted squared predictor discrepancy on the standardised scale.
    double fit_mspe{};        ///< Mean squared outcome gap over the fit periods.
    double pre_rmspe{};       ///< Root mean squared gap over all rows before the treatment period.
    double post_rmspe{};      ///< Root mean squared gap from the treatment period on.
    double rmspe_ratio{};     ///< post_rmspe / pre_rmspe.
};

/// @brief Synthetic control for one treated unit.
///
/// Each predictor is divided by its standard deviation across all units. For predictor weights V
/// the donor weights solve
/// \f[\min_W (X_1 - X_0W)^\top V (X_1 - X_0W)\quad\text{s.t. } W \ge 0,\ \textstyle\sum W = 1,\f]
/// which an exact active-set method solves. Unless V is given, V is chosen to minimise the mean
/// squared outcome gap over the fit periods, by Nelder-Mead from equal weights and from
/// regression-based weights (as in R's `Synth`), keeping the best.
/// @param data Treated unit, donors, fit periods and treatment period.
/// @param options Fixed V or search controls.
/// @return The fit.
/// @throws std::invalid_argument For inconsistent shapes, nonfinite values, fewer than two donors,
/// a constant predictor, a bad V, or periods out of range.
/// @par Example
/// @code
/// dm::causal::SyntheticControlData d;
/// d.treated_predictors = {2.0, 3.0};
/// d.donor_predictors = dm::linalg::DenseMatrix<double>(2, 3, {1, 3, 5, 2, 4, 9});
/// d.treated_outcome = {2, 3, 5};
/// d.donor_outcomes = dm::linalg::DenseMatrix<double>(3, 3, {1, 3, 6, 2, 4, 8, 3, 3, 9});
/// d.fit_periods = {0, 1};
/// d.treatment_period = 2;
/// const auto f = dm::causal::synthetic_control(d, {.predictor_weights = {1, 1}});
/// std::println("{:.2f} {:.2f}", f.weights[0], f.weights[1]); // 0.50 0.50
/// @endcode
SyntheticControlFit synthetic_control(const SyntheticControlData& data, const SyntheticControlOptions& options = {});

/// @brief Result of synthetic_control_placebos().
/// @par Example
/// @code
/// dm::causal::SyntheticControlPlacebos p;
/// p.p_value = 1.0 / 17;
/// std::println("{:.3f}", p.p_value); // 0.059
/// @endcode
struct SyntheticControlPlacebos {
    SyntheticControlFit treated;                ///< Fit for the actual treated unit.
    std::vector<SyntheticControlFit> placebos;  ///< Fit treating donor j as treated, other donors as pool.
    std::vector<std::size_t> included;          ///< Placebos passing the pre-treatment fit filter.
    double p_value{}; ///< (1 + included placebos with ratio >= treated ratio) / (1 + included).
};

/// @brief In-space placebo inference: refit with each donor in turn as the treated unit.
///
/// Each donor is treated as if it had been treated, with the remaining donors (never the real
/// treated unit) as its pool. The permutation p-value ranks the treated unit's post/pre RMSPE
/// ratio among the included placebos. Placebos whose pre-treatment MSPE exceeds
/// `max_pre_mspe_ratio` times the treated unit's are excluded, since a poor pre-treatment fit makes
/// post-treatment gaps uninformative.
/// @param data Treated unit, donors and periods.
/// @param options Options applied to every fit.
/// @param max_pre_mspe_ratio Exclusion threshold on relative pre-treatment MSPE (infinity keeps all).
/// @return The treated fit, placebo fits and p-value.
/// @throws std::invalid_argument As synthetic_control(), or fewer than three donors.
/// @par Example
/// @code
/// dm::causal::SyntheticControlData d;
/// d.treated_predictors = {2.0, 3.0};
/// d.donor_predictors = dm::linalg::DenseMatrix<double>(2, 3, {1, 3, 5, 2, 4, 9});
/// d.treated_outcome = {2, 3, 9};
/// d.donor_outcomes = dm::linalg::DenseMatrix<double>(3, 3, {1, 3, 6, 2, 4, 8, 3, 3, 9});
/// d.fit_periods = {0, 1};
/// d.treatment_period = 2;
/// const auto p = dm::causal::synthetic_control_placebos(d, {.predictor_weights = {1, 1}});
/// std::println("{}", p.placebos.size()); // 3
/// @endcode
SyntheticControlPlacebos synthetic_control_placebos(
    const SyntheticControlData& data, const SyntheticControlOptions& options = {},
    double max_pre_mspe_ratio = std::numeric_limits<double>::infinity());

} // namespace datamunge::causal
