#pragma once

/// @file lm.hpp
/// @brief R-`lm()`-style linear models: ordinary and weighted least squares fit from a
///        DataFrame/formula or dense/CSR matrix, with inference, ANOVA, prediction intervals, influence
///        diagnostics and R's four diagnostic plots.
///
/// The model is \f$y = X\beta + \varepsilon\f$ with \f$\varepsilon\sim N(0,\sigma^2 W^{-1})\f$,
/// where the formula builds the design matrix \f$X\f$ (factor columns use treatment coding by
/// default) and \f$W\f$ is the identity unless a weights column is given. The estimate is
/// \f$\hat\beta=(X^\top WX)^{-1}X^\top Wy\f$; every other quantity (standard errors, t and F
/// tests, intervals, leverage, Cook's distance) follows R's `summary.lm`, `anova.lm`,
/// `predict.lm` and `influence.measures`, so results can be checked against R directly.
///
/// @par References
/// - J. M. Chambers (1992). "Linear models". Chapter 4 of *Statistical Models in S*,
///   J. M. Chambers and T. J. Hastie (eds.), Wadsworth & Brooks/Cole.
/// - R `lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lm.html

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/regression.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>
#include <datamunge/stats/marginal_means.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace datamunge::random {
class RandomGenerator;
}

namespace datamunge::stats {

/// @brief Fitting options for LM, like the extra arguments of R's `lm()`.
struct LmOptions {
    /// Name of a numeric column of strictly positive case weights; set it for weighted least
    /// squares (R's `weights =`). Unset (the default) fits ordinary least squares.
    std::optional<std::string> weights_column;
    std::unordered_map<std::string, FactorContrast> contrasts{}; ///< Coding by categorical predictor name.
};

/// @brief Which interval LM::predict() adds around each fitted value, like R's
///        `predict.lm(interval = ...)`.
enum class PredictionInterval {
    None,       ///< Point predictions only.
    Confidence, ///< Interval for the mean response \f$x_0^\top\beta\f$.
    Prediction  ///< Interval for a new observation \f$x_0^\top\beta+\varepsilon\f$ (wider).
};

/// @brief Predictions from LM::predict(), one entry per row of the new data.
///
/// Rows with a null predictor get NaN in every vector.
struct LmPrediction {
    std::vector<double> fit;    ///< Predicted mean \f$x_0^\top\hat\beta\f$.
    std::vector<double> se_fit; ///< Standard error used for the interval (empty for PredictionInterval::None).
    std::vector<double> lower;  ///< Lower interval bound (empty for PredictionInterval::None).
    std::vector<double> upper;  ///< Upper interval bound (empty for PredictionInterval::None).
};

/// @brief One row of an ANOVA table from LM::anova() or LM::marginal_anova().
///
/// The final row is "Residuals", whose f_value and p_value are 0 (R prints them as blank).
struct AnovaRow {
    std::string term;                  ///< Term label, or "Residuals".
    std::size_t degrees_of_freedom{0}; ///< Number of tested coefficients (residual df on the last row).
    double      sum_sq{0.0};           ///< Sum of squares for the term test (RSS on the last row).
    double      mean_sq{0.0};          ///< sum_sq / degrees_of_freedom.
    double      f_value{0.0};          ///< mean_sq / \f$\hat\sigma^2\f$ of the full model.
    double      p_value{0.0};          ///< Upper tail of the F distribution at f_value.
};

/// @brief Adjustments for marginal tests of linear-model terms.
/// @par Example
/// @code
/// const auto type = dm::stats::MarginalAnovaType::III;
/// std::println("{}", type == dm::stats::MarginalAnovaType::III); // true
/// @endcode
/// @see LM::marginal_anova()
enum class MarginalAnovaType {
    II,  ///< Adjust for terms that do not contain the tested term.
    III, ///< Adjust for all other terms under the fitted coding.
};

/// @brief Result of the nested-model F-test `anova(reduced, full)`, like the two-row table of
///        R's `anova(fit0, fit1)`.
struct NestedAnova {
    std::size_t reduced_df{0};  ///< Residual degrees of freedom of the smaller model.
    std::size_t full_df{0};     ///< Residual degrees of freedom of the larger model.
    double      reduced_rss{0}; ///< Residual sum of squares of the smaller model.
    double      full_rss{0};    ///< Residual sum of squares of the larger model.
    std::size_t df{0};          ///< reduced_df - full_df: the number of constraints tested.
    double      sum_sq{0};      ///< reduced_rss - full_rss.
    double      f_value{0};     ///< (sum_sq / df) / (full_rss / full_df).
    double      p_value{0};     ///< Upper tail of \f$F_{df,\,full\_df}\f$ at f_value.
};

/// @brief Residual scaling for heteroscedasticity-consistent LM covariance.
/// @par Example
/// @code
/// const auto type = dm::stats::HcCovariance::HC3;
/// std::println("{}", type == dm::stats::HcCovariance::HC3); // true
/// @endcode
/// @par References
/// - MacKinnon and White (1985), https://doi.org/10.1016/0304-4076(85)90158-7
/// @see LM::heteroscedastic_covariance()
enum class HcCovariance {
    HC0, ///< Squared residuals with no degrees-of-freedom correction.
    HC1, ///< HC0 multiplied by n / (n - p).
    HC2, ///< Squared residuals divided by one minus leverage.
    HC3, ///< Squared residuals divided by the square of one minus leverage.
};

/// @brief An observed group contributing to a clustered covariance component.
/// @par Example
/// @code
/// const dm::stats::ClusterCovarianceGroup group{{10, 4}, 2};
/// std::println("{}", group.observations); // 2
/// @endcode
/// @see ClusterCovarianceComponent
struct ClusterCovarianceGroup {
    std::vector<std::int64_t> labels; ///< Observed cluster labels in component dimension order.
    std::size_t observations;        ///< Fitted rows in this group.
};

/// @brief One subset term in the multiway inclusion-exclusion sum.
/// @par Example
/// @code
/// const dm::stats::ClusterCovarianceComponent joint{{0, 1}, {{{10, 4}, 2}}, -1};
/// std::println("{}", joint.multiplier); // -1
/// @endcode
/// @see ClusterCovarianceGroup, LM::cluster_robust_covariance()
struct ClusterCovarianceComponent {
    std::vector<std::size_t> dimensions;       ///< Zero-based indices of the selected cluster dimensions.
    std::vector<ClusterCovarianceGroup> groups; ///< Observed joint clusters, in lexicographic label order.
    double multiplier;                         ///< Inclusion-exclusion sign times the applicable finite-sample factor.
};

/// @brief Coefficient covariance and the observed components of its clustered sum.
/// @par Example
/// @code
/// const dm::stats::LM model({0, 1, 0}, dm::Mat<>{{0}, {1}, {2}});
/// const auto fit = model.cluster_robust_covariance({{10, 10, 11}});
/// std::println("{}", fit.covariance.rows()); // 2
/// @endcode
/// @see LM::cluster_robust_covariance()
struct ClusterCovarianceResult {
    linalg::DenseMatrix<double> covariance;            ///< Covariance in model coefficient order.
    std::vector<ClusterCovarianceComponent> components; ///< Subsets in ascending dimension bit-mask order.
};

/// @brief An R-`lm()`-style linear model fit by ordinary or weighted least squares from a
///        DataFrame and a formula.
///
/// The constructor resolves the formula against the data (numeric columns enter as they are,
/// string columns as treatment-coded factors with the first level sorted alphabetically as the
/// baseline), drops every row with a null value in a column the formula uses (R's `na.omit`),
/// fits, and computes every summary and diagnostic once. All accessors are then cheap and
/// the object is immutable, so it can be shared between threads.
///
/// Per-observation vectors (fitted values, residuals, leverage, ...) have one entry per
/// *fitted* row, in data order, skipping dropped rows.
///
/// @par Example
/// @code
/// const auto model = dm::stats::LM(dm::formula("len ~ dose + supp"), dm::datasets::tooth_growth());
/// std::print("{}\n", model.coefficient_names());  // ["(Intercept)", "dose", "suppVC"]
/// std::print("{:.4f} {:.4f}\n", model.coefficients()[1], model.r_squared());  // 9.7636 0.7038
/// @endcode
///
/// @par References
/// - J. M. Chambers (1992). "Linear models". Chapter 4 of *Statistical Models in S*,
///   Wadsworth & Brooks/Cole.
/// - R `lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lm.html
///
/// @see Formula, GLM, LMM, anova(const LM&, const LM&)
class LM {
 public:
    /// @brief Fits `formula` to `data`, like R's `lm(formula, data, weights)`.
    ///
    /// @param data  Data frame holding every column the formula names (and the weights column).
    /// @param formula  Model formula such as `"y ~ x1 + x2 + group"`; `- 1` removes the intercept.
    /// @param options  Fitting options; set LmOptions::weights_column for weighted least squares.
    /// @throws std::invalid_argument  If the formula names a missing column, the weights column
    ///         is missing, null in a fitted row, or not strictly positive.
    /// @throws std::runtime_error  If the design matrix is rank deficient.
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const dm::stats::LM model(data, "y ~ x");
    /// std::print("{:.2f} {:.2f}\n", model.coefficients()[0], model.coefficients()[1]);  // 0.60 0.80
    /// @endcode
    ///
    /// @par References
    /// - R `lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lm.html
    ///
    /// @see LM(const Formula&, const dstruct::DataFrame&, LmOptions)
    LM(const dstruct::DataFrame& data, const std::string& formula, LmOptions options = {});

    /// @brief Fit a numeric response directly on dense predictor columns with Householder QR.
    /// Owns the fitted design. Adds an intercept by default; predictors must exclude it.
    /// Unlike the formula interface, nonfinite values are rejected rather than dropped.
    /// @param y Finite response values, one per predictor row.
    /// @param predictors Dense double predictors, in fixed column order.
    /// @param intercept Add a constant coefficient, default true.
    /// @throws std::invalid_argument For nonfinite values, incompatible shapes or n <= p.
    /// @throws std::runtime_error For rank-deficient predictors.
    /// @par Example
    /// @code
    /// const dm::stats::LM fit({1, 3, 5, 7}, dm::Mat<>{{0}, {1}, {2}, {3}});
    /// std::print("{:.0f}\n", fit.predict(dm::Mat<>{{4}})[0]); // 9
    /// @endcode
    /// @see LM(const std::vector<double>&, const SpMat<>&, bool, linalg::SparseLinearRegressionOptions)
    LM(const std::vector<double>& y, const Mat<>& predictors, bool intercept = true);

    /// @brief Fit a numeric response on CSR predictors using native sparse regression.
    /// Retains CSR storage, including any added intercept. Uses Jacobi-preconditioned
    /// conjugate gradients on normal equations, whose conditioning squares that of X.
    /// Inference uses a dense p-by-p Gram/covariance matrix; no dense n-by-p design
    /// or n-by-n orthogonal factor is created. Requires full column rank and n > p.
    /// @param y Finite response values, one per predictor row.
    /// @param predictors CSR double predictors, excluding any requested intercept.
    /// @param intercept Add a constant coefficient, default true.
    /// @param options Sparse solver settings: default relative residual tolerance
    /// 1e-12, 1000 iterations, and Jacobi preconditioning. Applied also to ANOVA refits.
    /// @throws std::invalid_argument For nonfinite values, invalid solver tolerance or incompatible/insufficient rows.
    /// @throws std::runtime_error For rank deficiency or failure to converge.
    /// @par Example
    /// @code
    /// const dm::stats::LM fit({1, 3, 5, 7}, dm::SpMat<>{{0}, {1}, {2}, {3}});
    /// std::print("{:.0f}\n", fit.predict(dm::SpMat<>{{4}})[0]); // 9
    /// @endcode
    /// @see linalg::sparse_linear_regression()
    LM(const std::vector<double>& y, const SpMat<>& predictors, bool intercept = true,
       linalg::SparseLinearRegressionOptions options = {{1e-12, 1000}, true});

    /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
    ///
    /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
    /// converts to Formula implicitly, so `LM("y ~ x", data)` also works.
    /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
    /// @param data  Data frame holding every column the formula names.
    /// @param options  Same as in the `(data, formula, ...)` constructor.
    /// @throws std::invalid_argument  As the `(data, formula, ...)` constructor.
    /// @throws std::runtime_error  As the `(data, formula, ...)` constructor.
    ///
    /// @par Example
    /// See LM.
    LM(const Formula& formula, const dstruct::DataFrame& data, LmOptions options = {})
        : LM(data, formula.text(), std::move(options)) {}

    /// @brief The formula as written.
    /// @return The formula text.
    [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
    /// @brief Whether the model has an intercept column.
    /// @return False only when the formula removed it with `- 1` or `+ 0`.
    [[nodiscard]] bool has_intercept() const { return design_.has_intercept; }
    /// @brief Number of fitted rows \f$n\f$ (after dropping rows with nulls).
    /// @return \f$n\f$.
    [[nodiscard]] std::size_t observations() const { return observations_; }
    /// @brief Number of coefficients \f$p\f$ (columns of the design matrix).
    /// @return \f$p\f$.
    [[nodiscard]] std::size_t rank() const { return coefficients_.size(); }
    /// @brief Residual degrees of freedom \f$n-p\f$.
    /// @return \f$n-p\f$.
    [[nodiscard]] std::size_t degrees_of_freedom() const { return degrees_of_freedom_; }

    /// @brief Data-frame row (0-based) of each fitted row, so per-observation vectors can be
    ///        matched back to the data; rows with nulls are absent.
    /// @return One row index per fitted row, increasing.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("body_mass_g ~ flipper_length_mm"), dm::datasets::penguins());
    /// std::print("{} {}\n", model.observations(), model.fitted_rows()[3]);  // 342 4
    /// @endcode
    [[nodiscard]] const std::vector<std::size_t>& fitted_rows() const { return fitted_rows_; }

    /// @brief Estimated coefficients \f$\hat\beta\f$, in the order of coefficient_names().
    /// @return One estimate per design column.
    [[nodiscard]] const std::vector<double>& coefficients() const { return coefficients_; }
    /// @brief Coefficient names as R prints them: `"(Intercept)"`, numeric column names, and
    ///        `column` + `level` for each factor contrast column (e.g. `"speciesGentoo"`).
    /// @return One name per coefficient.
    [[nodiscard]] const std::vector<std::string>& coefficient_names() const { return design_.coefficient_names; }
    /// @brief Fitted values \f$\hat y = X\hat\beta\f$, one per fitted row.
    /// @return The fitted values.
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    /// @brief Raw residuals \f$y-\hat y\f$, one per fitted row (unweighted even for WLS, as in R).
    /// @return The residuals.
    [[nodiscard]] const std::vector<double>& residuals() const { return residuals_; }
    /// @brief Standard errors \f$\hat\sigma\sqrt{[(X^\top WX)^{-1}]_{jj}}\f$ of the coefficients.
    /// @return One standard error per coefficient.
    [[nodiscard]] const std::vector<double>& standard_errors() const { return standard_errors_; }
    /// @brief t statistics \f$\hat\beta_j/\mathrm{SE}_j\f$ for \f$H_0:\beta_j=0\f$.
    /// @return One t value per coefficient.
    [[nodiscard]] const std::vector<double>& t_values() const { return t_values_; }
    /// @brief Two-sided p-values of the t statistics on \f$n-p\f$ degrees of freedom.
    /// @return One p-value per coefficient.
    [[nodiscard]] const std::vector<double>& p_values() const { return p_values_; }
    /// @brief Estimated covariance matrix \f$\hat\sigma^2(X^\top WX)^{-1}\f$ of \f$\hat\beta\f$
    ///        (R's `vcov()`).
    /// @return The \f$p\times p\f$ covariance matrix.
    [[nodiscard]] const linalg::DenseMatrix<double>& covariance() const { return covariance_; }

    /// @brief Computes an HC0, HC1, HC2 or HC3 covariance from the fitted regression scores.
    /// Uses weighted scores and weighted leverage when the fit has case weights.
    /// @param estimator The residual and leverage correction, default HC3.
    /// @return Symmetric covariance in coefficient order.
    /// @throws std::invalid_argument Invalid estimator.
    /// @throws std::domain_error HC2 or HC3 encounter leverage numerically equal to one.
    /// @throws std::overflow_error A covariance entry exceeds finite double range.
    /// @par Example
    /// @code
    /// const dm::stats::LM model({0, 1, 0}, dm::Mat<>{{0}, {1}, {2}});
    /// std::println("{:.6f}", model.heteroscedastic_covariance(dm::stats::HcCovariance::HC0)(1, 1)); // 0.055556
    /// @endcode
    /// @par References
    /// - MacKinnon and White (1985), https://doi.org/10.1016/0304-4076(85)90158-7
    /// @see newey_west_covariance(), cluster_robust_covariance()
    [[nodiscard]] linalg::DenseMatrix<double> heteroscedastic_covariance(
        HcCovariance estimator = HcCovariance::HC3) const;

    /// @brief Computes Newey–West coefficient covariance with Bartlett lag weights.
    /// Scores follow fitted-row order. Formula fits omit dropped rows before
    /// counting lags. A maximum lag of zero yields HC0 or HC1, depending on adjustment.
    /// @param max_lag Largest included lag, strictly less than the number of fitted rows.
    /// @param adjust_degrees_of_freedom Multiply by n / (n - p) when true.
    /// @return Symmetric covariance in coefficient order.
    /// @throws std::invalid_argument Lag exceeds the available fitted row pairs.
    /// @throws std::overflow_error A covariance entry exceeds finite double range.
    /// @par Example
    /// @code
    /// const dm::stats::LM model({0, 1, 0}, dm::Mat<>{{0}, {1}, {2}});
    /// std::println("{:.6f}", model.newey_west_covariance(1)(1, 1)); // 0.055556
    /// @endcode
    /// @par References
    /// - Newey and West (1987), https://doi.org/10.2307/1913610
    /// @see heteroscedastic_covariance()
    [[nodiscard]] linalg::DenseMatrix<double> newey_west_covariance(
        std::size_t max_lag, bool adjust_degrees_of_freedom = false) const;

    /// @brief Computes a one-way or multiway clustered sandwich covariance.
    /// Sums fitted scores within the observed joint cluster of each nonempty
    /// subset of dimensions. Odd-sized subsets add their component and even-sized
    /// subsets subtract it. The resulting multiway matrix can be indefinite.
    /// @param labels One vector per clustering dimension, each in fitted_rows() order.
    /// @param finite_sample_adjustment Apply G/(G-1) times (n-1)/(n-p) separately
    /// to each subset with G observed groups.
    /// @return Covariance and the labels and counts contributing to every subset term.
    /// @throws std::invalid_argument Empty dimensions or a label vector of the wrong length.
    /// @throws std::domain_error An adjusted component has fewer than two observed groups.
    /// @throws std::length_error The number of subsets exceeds the index range.
    /// @throws std::overflow_error A covariance entry exceeds finite double range.
    /// @par Example
    /// @code
    /// const dm::stats::LM model({0, 1, 0}, dm::Mat<>{{0}, {1}, {2}});
    /// std::println("{}", model.cluster_robust_covariance({{10, 10, 11}}).components.front().groups.size()); // 2
    /// @endcode
    /// @par References
    /// - Cameron, Gelbach and Miller (2011), https://doi.org/10.1198/jbes.2010.07136
    /// @see ClusterCovarianceResult, heteroscedastic_covariance()
    [[nodiscard]] ClusterCovarianceResult cluster_robust_covariance(
        const std::vector<std::vector<std::int64_t>>& labels, bool finite_sample_adjustment = false) const;

    /// @brief Tests variance dependence on supplied predictors by Breusch–Pagan regression.
    /// Regresses squared residuals on an intercept and the supplied predictors. For WLS,
    /// residuals are multiplied by the square root of their case weights before squaring.
    /// @param variance_predictors Finite matrix with rows in fitted_rows() order. The intercept is automatic.
    /// @param studentized Use Koenker's n times R-squared statistic when true, the default.
    /// Otherwise use the Gaussian Breusch–Pagan LM statistic.
    /// @return Chi-square statistic and upper-tail p-value, parameter1 equal to the numerical
    /// predictor rank after removing the intercept, and estimate1 equal to the auxiliary R-squared.
    /// @throws std::invalid_argument Invalid predictor matrix.
    /// @throws std::domain_error The residual or auxiliary fit gives undefined inference.
    /// @par Example
    /// @code
    /// const dm::stats::LM fit({0, 2, 1, 5, 4}, dm::Mat<>{{0}, {1}, {2}, {3}, {4}});
    /// std::println("{}", fit.breusch_pagan(dm::linalg::DenseMatrix<double>(5, 1, {0, 1, 2, 3, 4})).parameter1); // 1
    /// @endcode
    /// @par References
    /// - Breusch and Pagan (1979), https://doi.org/10.2307/1911963
    /// - Koenker (1981), https://doi.org/10.1016/0304-4076(81)90062-2
    /// @see white_heteroscedasticity(), heteroscedastic_covariance()
    [[nodiscard]] HypothesisTestResult breusch_pagan(
        const linalg::DenseMatrix<double>& variance_predictors, bool studentized = true) const;

    /// @brief Tests variance dependence on predictors, squares and pairwise products.
    /// Regresses squared working residuals on an intercept and the expanded predictor
    /// set. Numerical rank determines the degrees of freedom, including with indicator columns.
    /// @param variance_predictors Finite matrix with rows in fitted_rows() order. The intercept is automatic.
    /// @return n times auxiliary R-squared and its chi-square upper-tail p-value, with
    /// parameter1 equal to the numerical predictor rank and estimate1 the auxiliary R-squared.
    /// @throws std::invalid_argument Invalid predictor matrix.
    /// @throws std::domain_error The residual or auxiliary fit gives undefined inference.
    /// @throws std::length_error The number of auxiliary columns exceeds the index range.
    /// @par Example
    /// @code
    /// const dm::stats::LM fit({0, 2, 1, 5, 4}, dm::Mat<>{{0}, {1}, {2}, {3}, {4}});
    /// std::println("{}", fit.white_heteroscedasticity(dm::linalg::DenseMatrix<double>(5, 1, {0, 1, 2, 3, 4})).parameter1); // 2
    /// @endcode
    /// @par References
    /// - White (1980), https://doi.org/10.2307/1912934
    /// @see breusch_pagan()
    [[nodiscard]] HypothesisTestResult white_heteroscedasticity(
        const linalg::DenseMatrix<double>& variance_predictors) const;

    /// @brief Tests powers of the fitted values by the Ramsey RESET partial F statistic.
    /// Retains case weights and tests additional estimable columns jointly.
    /// @param powers Distinct powers of at least two. The default tests powers two and three.
    /// @return F statistic and upper-tail p-value, with numerator and denominator degrees of
    /// freedom in parameter1 and parameter2.
    /// @throws std::invalid_argument Invalid powers.
    /// @throws std::domain_error The residual or augmented design gives undefined inference.
    /// @throws std::length_error The augmented column count exceeds the index range.
    /// @par Example
    /// @code
    /// const dm::stats::LM fit({0, 2, 1, 5, 4}, dm::Mat<>{{0}, {1}, {2}, {3}, {4}});
    /// std::println("{}", fit.ramsey_reset({2}).parameter2); // 2
    /// @endcode
    /// @par References
    /// - Ramsey (1969), https://doi.org/10.1111/j.2517-6161.1969.tb00796.x
    /// @see breusch_pagan(), white_heteroscedasticity()
    [[nodiscard]] HypothesisTestResult ramsey_reset(const std::vector<unsigned>& powers = {2, 3}) const;

    /// @brief Tests lag-one residual correlation using a design-conditional Durbin–Watson null.
    /// Simulates independent Gaussian errors in the residual space of the fitted
    /// design. For WLS, both observed and simulated residuals are in weighted coordinates.
    /// Successive fitted rows define neighbors, with omitted rows skipped.
    /// @param rng Caller-owned random generator advanced by the null draws.
    /// @param simulations Positive number of null draws, default 9999.
    /// @param alternative Greater for positive correlation, Less for negative correlation,
    /// or TwoSided for twice the smaller tail capped at one.
    /// @return Durbin–Watson ratio and Monte Carlo p-value, using inclusive tails and
    /// the plus-one correction. Parameter1 is the number of null draws.
    /// @throws std::invalid_argument Invalid simulation count or alternative.
    /// @throws std::domain_error The observed statistic or numerical residual space is undefined.
    /// @par Example
    /// @code
    /// dm::random::SplitMix64 rng(3);
    /// const dm::stats::LM fit({1, 0, 0}, dm::Mat<>(3, 0));
    /// std::println("{:.1f}", fit.durbin_watson(rng, 99).statistic); // 1.5
    /// @endcode
    /// @par References
    /// - Durbin and Watson (1950), https://doi.org/10.1093/biomet/37.3-4.409
    /// @see breusch_godfrey()
    [[nodiscard]] HypothesisTestResult durbin_watson(random::RandomGenerator& rng, std::size_t simulations = 9999,
                                                    Alternative alternative = Alternative::TwoSided) const;

    /// @brief Tests lagged residuals jointly by a Breusch–Godfrey auxiliary regression.
    /// Augments the fitted design with lags of the working residual. Entries before
    /// the first fitted row use zero. WLS uses weighted design and residual coordinates.
    /// @param lags Positive maximum lag, less than the number of fitted rows.
    /// @param use_f Use the approximate partial F reference when true. The default uses
    /// the asymptotic chi-square LM reference.
    /// @return Statistic and upper-tail p-value. Parameter1 is the number of estimable
    /// lag terms, parameter2 is the denominator df for the F form, and estimate1 is
    /// the auxiliary proportion of residual sum of squares explained.
    /// @throws std::invalid_argument Invalid lag count.
    /// @throws std::domain_error The residuals are numerically zero or the auxiliary fit has undefined inference.
    /// @throws std::length_error The augmented matrix size exceeds the index range.
    /// @par Example
    /// @code
    /// const dm::stats::LM fit({1, 0, 1, 2, 4}, dm::Mat<>{{0}, {1}, {2}, {3}, {4}});
    /// std::println("{}", fit.breusch_godfrey().parameter1); // 1
    /// @endcode
    /// @par References
    /// - Breusch (1978), https://doi.org/10.1111/j.1467-8454.1978.tb00635.x
    /// - Godfrey (1978), https://doi.org/10.2307/1913829
    /// @see durbin_watson()
    [[nodiscard]] HypothesisTestResult breusch_godfrey(std::size_t lags = 1, bool use_f = false) const;

    /// @brief Coefficient of determination \f$R^2=1-\mathrm{RSS}/\mathrm{TSS}\f$ (weighted for
    ///        WLS; TSS about 0 for a model without intercept, as in R).
    /// @return \f$R^2\f$.
    [[nodiscard]] double r_squared() const { return r_squared_; }
    /// @brief Adjusted \f$R^2 = 1-(1-R^2)(n-1)/(n-p)\f$.
    /// @return Adjusted \f$R^2\f$.
    [[nodiscard]] double adjusted_r_squared() const { return adjusted_r_squared_; }
    /// @brief Residual standard error \f$\hat\sigma=\sqrt{\mathrm{RSS}/(n-p)}\f$.
    /// @return \f$\hat\sigma\f$.
    [[nodiscard]] double sigma() const { return sigma_; }
    /// @brief Overall F statistic for "every coefficient except the intercept is zero".
    /// @return The F statistic, or NaN when the model has no slope or no residual df.
    [[nodiscard]] double f_statistic() const { return f_statistic_; }
    /// @brief p-value of f_statistic().
    /// @return The p-value, or NaN as f_statistic().
    [[nodiscard]] double f_p_value() const { return f_p_value_; }

    /// @brief Residual sum of squares \f$\sum_i w_i(y_i-\hat y_i)^2\f$ (R's `deviance()`).
    ///
    /// @return The (weighted) RSS.
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// std::print("{:.1f}\n", dm::stats::LM(data, "y ~ x").rss());  // 3.6
    /// @endcode
    ///
    /// @par References
    /// - R `deviance()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/deviance.html
    ///
    /// @see sigma(), log_likelihood()
    [[nodiscard]] double rss() const { return sigma2_ * static_cast<double>(degrees_of_freedom_); }

    /// @brief Maximised Gaussian log-likelihood, as R's `logLik.lm`.
    ///
    /// \f$\ell=\tfrac12\big(\sum_i\log w_i-n(\log 2\pi+1-\log n+\log\mathrm{RSS})\big)\f$, using
    /// the maximum-likelihood variance \f$\mathrm{RSS}/n\f$ (not \f$\hat\sigma^2\f$).
    ///
    /// @return \f$\ell\f$.
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const dm::stats::LM model(data, "y ~ x");
    /// std::print("{:.4f} {:.4f} {:.4f}\n", model.log_likelihood(), model.aic(), model.bic());
    /// // -6.2734 18.5469 17.3752
    /// @endcode
    ///
    /// @par References
    /// - R `logLik()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/logLik.html
    ///
    /// @see aic(), bic()
    [[nodiscard]] double log_likelihood() const;

    /// @brief Akaike information criterion \f$-2\ell+2k\f$ with \f$k=p+1\f$ parameters
    ///        (the coefficients and \f$\sigma\f$), as R's `AIC()`.
    ///
    /// Lower is better; only differences between models fit to the same rows are meaningful.
    ///
    /// @return The AIC.
    ///
    /// @par Example
    /// See log_likelihood().
    ///
    /// @par References
    /// - H. Akaike (1974). "A new look at the statistical model identification". IEEE
    ///   Transactions on Automatic Control 19(6), 716-723. doi:10.1109/TAC.1974.1100705
    ///
    /// @see bic(), log_likelihood()
    [[nodiscard]] double aic() const;

    /// @brief Bayesian (Schwarz) information criterion \f$-2\ell+k\log n\f$ with \f$k=p+1\f$,
    ///        as R's `BIC()`.
    ///
    /// @return The BIC.
    ///
    /// @par Example
    /// See log_likelihood().
    ///
    /// @par References
    /// - G. Schwarz (1978). "Estimating the dimension of a model". The Annals of Statistics
    ///   6(2), 461-464. doi:10.1214/aos/1176344136
    ///
    /// @see aic(), log_likelihood()
    [[nodiscard]] double bic() const;

    /// @brief A two-sided confidence interval for one coefficient.
    struct Interval {
        double lower; ///< Lower bound.
        double upper; ///< Upper bound.
    };

    /// @brief Confidence intervals \f$\hat\beta_j\pm t_{n-p,\,1-\alpha/2}\,\mathrm{SE}_j\f$ for
    ///        every coefficient, like R's `confint()`.
    ///
    /// @param level  Confidence level in (0, 1) (default 0.95).
    /// @return One interval per coefficient.
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const auto ci = dm::stats::LM(data, "y ~ x").confidence_intervals()[1];
    /// std::print("{:.4f} {:.4f}\n", ci.lower, ci.upper);  // -0.3024 1.9024
    /// @endcode
    ///
    /// @par References
    /// - R `confint()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/confint.html
    ///
    /// @see coefficient_frame()
    [[nodiscard]] std::vector<Interval> confidence_intervals(double level = 0.95) const;

    /// @brief The coefficient table of R's `summary(fit)` plus `confint(fit)`, as a DataFrame.
    ///
    /// Columns: `term` (string), `estimate`, `std_error`, `t_value`, `p_value`, `lower`,
    /// `upper` (numeric), one row per coefficient. Handy for `to_md()`.
    ///
    /// @param level  Confidence level of the `lower`/`upper` columns (default 0.95).
    /// @return The table.
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const auto table = dm::stats::LM(data, "y ~ x").coefficient_frame();
    /// std::print("{} {}\n", table.nrows(), table.columns());
    /// // 2 ["term", "estimate", "std_error", "t_value", "p_value", "lower", "upper"]
    /// @endcode
    ///
    /// @par References
    /// - R `summary.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.lm.html
    ///
    /// @see confidence_intervals(), summary()
    [[nodiscard]] dstruct::DataFrame coefficient_frame(double level = 0.95) const;

    /// @brief Leverages \f$h_{ii}\f$: the diagonal of the hat matrix
    ///        \f$H=W^{1/2}X(X^\top WX)^{-1}X^\top W^{1/2}\f$ (R's `hatvalues()`).
    ///
    /// Each lies in [0, 1) and they sum to \f$p\f$.
    /// @return One leverage per fitted row.
    [[nodiscard]] const std::vector<double>& leverage() const { return leverage_; }
    /// @brief Standardized residuals \f$\sqrt{w_i}\,e_i/(\hat\sigma\sqrt{1-h_{ii}})\f$
    ///        (R's `rstandard()`).
    /// @return One standardized residual per fitted row.
    [[nodiscard]] const std::vector<double>& standardized_residuals() const { return standardized_residuals_; }
    /// @brief Externally studentized residuals, each scaled by \f$\hat\sigma_{(i)}\f$ from the
    ///        fit without row \f$i\f$ (R's `rstudent()`); \f$t_{n-p-1}\f$ under the model.
    /// @return One studentized residual per fitted row.
    [[nodiscard]] const std::vector<double>& studentized_residuals() const { return studentized_residuals_; }
    /// @brief Cook's distances \f$D_i=\frac{r_i^2}{p}\frac{h_{ii}}{1-h_{ii}}\f$ for standardized
    ///        residual \f$r_i\f$ (R's `cooks.distance()`).
    /// @return One distance per fitted row.
    [[nodiscard]] const std::vector<double>& cooks_distance() const { return cooks_distance_; }

    /// @brief Sequential (type I) analysis of variance, like R's `anova(fit)`.
    ///
    /// Terms are added in formula order; each row's sum of squares is the drop in RSS when its
    /// columns join the model, tested against the full model's \f$\hat\sigma^2\f$. The rows'
    /// sums of squares add up to the total (about the mean when there is an intercept). The
    /// last row is "Residuals". Refits one prefix model per term.
    ///
    /// @return One row per formula term, then the "Residuals" row; empty for a model with no terms.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("len ~ dose + supp"), dm::datasets::tooth_growth());
    /// for (const auto& row : model.anova())
    ///     std::print("{} {} {:.2f}\n", row.term, row.degrees_of_freedom, row.sum_sq);
    /// // dose 1 2224.30
    /// // supp 1 205.35
    /// // Residuals 57 1022.56
    /// @endcode
    ///
    /// @par References
    /// - R `anova.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/anova.lm.html
    ///
    /// @see anova_frame(), anova(const LM&, const LM&)
    [[nodiscard]] std::vector<AnovaRow> anova() const;

    /// @brief Tests Type II or Type III linear-model terms with the full fit's residual mean square.
    ///
    /// Type II compares submodels with and without a term after dropping any
    /// higher-order terms that contain all its factors. Type III tests the
    /// term's coefficients jointly after adjustment for every other term.
    /// With a full factorial model, sum coding gives equal-level marginal
    /// hypotheses, while treatment coding tests main effects at reference
    /// levels. Tests use the fitted rows and case weights.
    /// @param type Type II or Type III.
    /// @return One row per formula term, then the residual row. Models with no terms return an empty table.
    /// @throws std::invalid_argument Unknown ANOVA type.
    /// @throws std::runtime_error Numerical loss of estimability of the hypothesis.
    /// @throws std::overflow_error A term sum of squares exceeds finite double range.
    /// @par Example
    /// @code
    /// const dm::stats::LM fit({1, 2, 5, 6}, dm::Mat<>{{0}, {1}, {2}, {3}});
    /// const auto table = fit.marginal_anova(dm::stats::MarginalAnovaType::III);
    /// std::println("{:.1f}", table.front().sum_sq); // 16.2
    /// @endcode
    /// @par References
    /// - Fox and Weisberg, An R Companion to Applied Regression. https://cran.r-project.org/web/packages/car/refman/car.html
    /// @see anova(), LmOptions
    [[nodiscard]] std::vector<AnovaRow> marginal_anova(MarginalAnovaType type) const;

    /// @brief anova() as a DataFrame with columns `term`, `df`, `sum_sq`, `mean_sq`, `f_value`,
    ///        `p_value`; the Residuals row has null `f_value` and `p_value`, as R prints it.
    ///
    /// @return The table.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("len ~ dose + supp"), dm::datasets::tooth_growth());
    /// std::print("{}\n", model.anova_frame().nrows());  // 3
    /// @endcode
    ///
    /// @par References
    /// - R `anova.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/anova.lm.html
    ///
    /// @see anova()
    [[nodiscard]] dstruct::DataFrame anova_frame() const;

    /// @brief R's `print(summary(fit))` layout: call, residual quantiles, coefficient table with
    ///        significance stars, residual standard error, \f$R^2\f$ and the overall F test.
    ///
    /// @return The summary text.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("len ~ dose"), dm::datasets::tooth_growth());
    /// std::print("{}", model.summary());  // Call: lm(formula = len ~ dose) ...
    /// @endcode
    ///
    /// @par References
    /// - R `summary.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.lm.html
    ///
    /// @see print_summary(), coefficient_frame()
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to @p os.
    /// @param os  Output stream.
    void print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    void print_summary() const;

    /// @brief Point predictions \f$x_0^\top\hat\beta\f$ for each row of @p newdata, like R's
    ///        `predict(fit, newdata)`.
    ///
    /// @p newdata needs every predictor column (not the response); factor columns must use
    /// levels seen in fitting. Rows with a null predictor predict NaN.
    ///
    /// @param newdata  Rows to predict.
    /// @return One prediction per row of @p newdata.
    /// @throws std::invalid_argument  If a column is missing or a factor level is new.
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const DF grid{{"x", DF::numeric_column_type(dm::c(0.0, 6.0))}};
    /// std::print("{::.1f}\n", dm::stats::LM(data, "y ~ x").predict(grid));  // [0.6, 5.4]
    /// @endcode
    ///
    /// @par References
    /// - R `predict.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/predict.lm.html
    ///
    /// @see predict(const dstruct::DataFrame&, PredictionInterval, double), predict_frame()
    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    /// @brief Averages fitted predictions over an explicit reference grid and estimates linear contrasts.
    ///
    /// Grid rows use the fitted formula's factor levels and transformations.
    /// Means average the resulting model rows with weights normalized within
    /// each group. The reported covariance propagates coefficient uncertainty.
    /// Student t intervals use the fitted residual degrees of freedom, with
    /// Bonferroni adjustment applied to the means and contrasts as two families.
    /// @param grid Nonempty frame with all predictors required for prediction and no missing model rows.
    /// @param options Row labels and weights, optional contrast matrix, and interval settings.
    /// @return Weighted means in first-appearance group order and contrasts in matrix row order.
    /// @throws std::invalid_argument Invalid grid, options, weights or contrast matrix.
    /// @throws std::runtime_error The grid has a factor level absent from the fit.
    /// @throws std::overflow_error An estimate or interval exceeds finite double range.
    /// @par Example
    /// @code
    /// dm::dstruct::DataFrame data, grid;
    /// data.add_column("y", std::vector<double>{1, 2, 5, 6});
    /// data.add_column("x", std::vector<double>{0, 1, 2, 3});
    /// grid.add_column("x", std::vector<double>{0, 3});
    /// const dm::stats::MarginalMeansOptions options{.groups = {"low", "high"},
    ///     .contrasts = dm::linalg::DenseMatrix<double>(1, 2, {-1, 1})};
    /// const auto result = dm::stats::LM(data, "y ~ x").estimated_marginal_means(grid, options);
    /// std::println("{:.1f}", result.contrasts.estimates.front().estimate); // 5.4
    /// @endcode
    /// @par References
    /// - Lenth, emmeans reference manual: https://cran.r-project.org/web/packages/emmeans/refman/emmeans.html
    /// @see predict()
    [[nodiscard]] MarginalMeansResult estimated_marginal_means(
        const dstruct::DataFrame& grid, const MarginalMeansOptions& options = {}) const;

    /// @brief Predict a matrix-fitted model from dense predictor columns in training order.
    /// Adds the fitted intercept automatically, without expecting it in newdata.
    /// @param newdata Finite predictors with the training predictor-column count; may have zero rows.
    /// @return One fitted mean per input row.
    /// @throws std::invalid_argument For a formula-fitted model, column mismatch or nonfinite values.
    /// @par Example
    /// See the dense matrix constructor's prediction example.
    /// @see LM(const std::vector<double>&, const Mat<>&, bool)
    [[nodiscard]] std::vector<double> predict(const Mat<>& newdata) const;

    /// @brief Predict a matrix-fitted model directly from stored CSR entries.
    /// Implicit zeros contribute zero; no dense predictor grid is allocated.
    /// @param newdata Finite CSR predictors in training order, excluding any intercept; zero rows are valid.
    /// @return One fitted mean per input row, computed in O(rows + nnz) time.
    /// @throws std::invalid_argument For a formula-fitted model, column mismatch or nonfinite values.
    /// @par Example
    /// See the sparse matrix constructor's prediction example.
    /// @see LM(const std::vector<double>&, const SpMat<>&, bool, linalg::SparseLinearRegressionOptions)
    [[nodiscard]] std::vector<double> predict(const SpMat<>& newdata) const;

    /// @brief Predictions with a confidence or prediction interval, like R's
    ///        `predict(fit, newdata, interval = ..., level = ...)`.
    ///
    /// With \f$q=x_0^\top(X^\top WX)^{-1}x_0\f$ the confidence interval is
    /// \f$\hat y_0\pm t\,\hat\sigma\sqrt q\f$ and the prediction interval (for a new observation
    /// of unit weight) is \f$\hat y_0\pm t\,\hat\sigma\sqrt{1+q}\f$, with
    /// \f$t=t_{n-p,\,1-\alpha/2}\f$. `se_fit` holds the half-width divided by \f$t\f$ (note R's
    /// `se.fit` is always \f$\hat\sigma\sqrt q\f$).
    ///
    /// @param newdata  Rows to predict.
    /// @param interval  Which interval to add.
    /// @param level  Coverage level in (0, 1) (default 0.95).
    /// @return Fit and, unless @p interval is None, se_fit, lower and upper.
    /// @throws std::invalid_argument  As predict(const dstruct::DataFrame&).
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const DF at3{{"x", DF::numeric_column_type(dm::c(3.0))}};
    /// const auto pi = dm::stats::LM(data, "y ~ x").predict(at3, dm::stats::PredictionInterval::Prediction);
    /// std::print("{:.4f} {:.4f}\n", pi.lower[0], pi.upper[0]);  // -0.8189 6.8189
    /// @endcode
    ///
    /// @par References
    /// - R `predict.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/predict.lm.html
    ///
    /// @see predict_frame()
    [[nodiscard]] LmPrediction predict(const dstruct::DataFrame& newdata, PredictionInterval interval,
                                       double level = 0.95) const;

    /// @brief predict() as a DataFrame with a `fit` column, plus `se_fit`, `lwr` and `upr` when
    ///        an interval is requested; NaN predictions become nulls.
    ///
    /// @param newdata  Rows to predict.
    /// @param interval  Which interval to add (default None).
    /// @param level  Coverage level (default 0.95).
    /// @return One row per row of @p newdata.
    /// @throws std::invalid_argument  As predict(const dstruct::DataFrame&).
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const DF at3{{"x", DF::numeric_column_type(dm::c(3.0))}};
    /// const auto frame = dm::stats::LM(data, "y ~ x").predict_frame(at3, dm::stats::PredictionInterval::Confidence);
    /// std::print("{}\n", frame.columns());  // ["fit", "se_fit", "lwr", "upr"]
    /// @endcode
    ///
    /// @see predict(const dstruct::DataFrame&, PredictionInterval, double)
    [[nodiscard]] dstruct::DataFrame predict_frame(const dstruct::DataFrame& newdata,
                                                   PredictionInterval        interval = PredictionInterval::None,
                                                   double                    level    = 0.95) const;

    /// @brief R's `plot(fit, which = 1)`: residuals against fitted values, with a zero line and
    ///        a lowess smoother; the three largest residuals are labelled with their 1-based data row
    ///        number, as R does.
    ///
    /// A curved smoother suggests a missing nonlinear term; a funnel suggests unequal variance.
    /// @return The plot.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("len ~ dose"), dm::datasets::tooth_growth());
    /// model.plot_residuals_vs_fitted().save("resid_fitted.svg");
    /// @endcode
    ///
    /// @par References
    /// - R `plot.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/plot.lm.html
    ///
    /// @see plot_diagnostics(), lowess()
    [[nodiscard]] graphics::RPlot plot_residuals_vs_fitted() const;

    /// @brief R's `plot(fit, which = 2)`: normal Q-Q plot of the standardized residuals with
    ///        the line through their quartiles.
    ///
    /// Theoretical quantiles use R's `ppoints()` positions. Points bending away at both ends
    /// mean heavier (or lighter) tails than normal.
    /// @return The plot.
    ///
    /// @par Example
    /// See plot_residuals_vs_fitted().
    ///
    /// @par References
    /// - R `plot.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/plot.lm.html
    ///
    /// @see plot_diagnostics()
    [[nodiscard]] graphics::RPlot plot_normal_qq() const;

    /// @brief R's `plot(fit, which = 3)`: \f$\sqrt{|r_i|}\f$ of the standardized residuals
    ///        against fitted values, with a lowess smoother.
    ///
    /// A rising smoother means the spread grows with the mean (heteroscedasticity).
    /// @return The plot.
    ///
    /// @par Example
    /// See plot_residuals_vs_fitted().
    ///
    /// @par References
    /// - R `plot.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/plot.lm.html
    ///
    /// @see plot_diagnostics()
    [[nodiscard]] graphics::RPlot plot_scale_location() const;

    /// @brief R's `plot(fit, which = 5)`: standardized residuals against leverage, with dashed
    ///        Cook's distance contours at 0.5 and 1.
    ///
    /// Points beyond a contour have that much influence on \f$\hat\beta\f$. The contours
    /// \f$r=\pm\sqrt{Dp(1-h)/h}\f$ are drawn only inside the plot area, and the three points
    /// with the largest Cook's distance are labelled with their 1-based data row number.
    /// @return The plot.
    ///
    /// @par Example
    /// See plot_residuals_vs_fitted().
    ///
    /// @par References
    /// - R. D. Cook (1977). "Detection of influential observation in linear regression".
    ///   Technometrics 19(1), 15-18. doi:10.1080/00401706.1977.10489493
    /// - R `plot.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/plot.lm.html
    ///
    /// @see plot_diagnostics(), cooks_distance()
    [[nodiscard]] graphics::RPlot plot_residuals_vs_leverage() const;

    /// @brief R's default `par(mfrow = c(2, 2)); plot(fit)`: the four diagnostic plots in a 2x2
    ///        layout, 1100 x 860 pixels, panels fitted to their cells.
    ///
    /// @return The layout; save it with `.save("diagnostics.svg")`.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("len ~ dose + supp"), dm::datasets::tooth_growth());
    /// const auto fig = model.plot_diagnostics();
    /// std::print("{}\n", fig.panel_count());  // 4
    /// @endcode
    ///
    /// @par References
    /// - R `plot.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/plot.lm.html
    ///
    /// @see save_diagnostic_plots()
    [[nodiscard]] graphics::RLayout plot_diagnostics() const;

    /// @brief Saves the four diagnostic plots as separate SVG files
    ///        `<path_prefix>_residuals_vs_fitted.svg`, `_normal_qq.svg`, `_scale_location.svg`
    ///        and `_residuals_vs_leverage.svg`.
    /// @param path_prefix  Path and file-name stem.
    /// @throws std::runtime_error  If a file cannot be written.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("len ~ dose"), dm::datasets::tooth_growth());
    /// model.save_diagnostic_plots("tooth");  // tooth_normal_qq.svg, ...
    /// @endcode
    ///
    /// @see plot_diagnostics()
    void save_diagnostic_plots(const std::string& path_prefix) const;

 private:
    /// @brief Resolves the formula, builds the (weighted) design, fits and computes every summary.
    /// @param data  The data frame given to the constructor.
    /// @param options  The options given to the constructor.
    void fit(const dstruct::DataFrame& data, LmOptions options);
    /// @brief Validate a direct matrix model and resolve its numeric column schema.
    /// @param y Finite response with rows entries.
    /// @param rows Predictor rows.
    /// @param cols Predictor columns excluding the intercept.
    /// @param intercept Whether to add a constant coefficient.
    /// @throws std::invalid_argument For nonfinite response values or invalid dimensions.
    void initialise_matrix_design(const std::vector<double>& y, std::size_t rows,
                                  std::size_t cols, bool intercept);
    /// @brief Fill inference and diagnostics shared by formula, dense and sparse fits.
    /// @param result Canonical least-squares coefficients and covariance.
    /// @param y Unweighted response for fitted rows.
    /// @param fitted Unweighted fitted values, transferred into the model.
    /// @throws std::runtime_error If an exact fit's Gram matrix is singular.
    void finish_fit(linalg::LinearRegressionResult<double> result,
                    const std::vector<double>& y, std::vector<double> fitted);
    /// @brief Check that a prediction matrix belongs to a direct matrix model's schema.
    /// @param cols Predictor count, excluding any intercept.
    /// @throws std::invalid_argument For formula models or incompatible column counts.
    void validate_matrix_prediction(std::size_t cols) const;
    /// @brief Fills leverage_, standardized_residuals_, studentized_residuals_ and cooks_distance_.
    /// Visits stored CSR row entries for sparse models; dense models use their weighted design grid.
    void compute_diagnostics();
    /// @brief Builds each fitted row's coefficient-space score with the requested leverage scaling.
    /// @param estimator HC2 or HC3 for leverage scaling, or HC0 or HC1 for raw scores.
    /// @return Matrix of scores with rows in fitted-row order and columns in coefficient order.
    [[nodiscard]] std::vector<std::vector<long double>> sandwich_scores(HcCovariance estimator) const;
    /// @brief Returns weighted residuals scaled to maximum absolute magnitude one.
    /// @return One value per fitted row.
    /// @throws std::domain_error The model has numerically zero residual variation.
    [[nodiscard]] std::vector<long double> scaled_working_residuals() const;
    /// @brief Copies the weighted fit design and reserves zero columns for auxiliary terms.
    /// @param additional_columns Number of zero columns appended after the fitted design.
    /// @return Dense working-precision design in fitted row and column order.
    /// @throws std::length_error The augmented matrix size exceeds the index range.
    [[nodiscard]] linalg::DenseMatrix<long double> augmented_working_design(
        std::size_t additional_columns = 0) const;
    /// @brief Throws if @p newdata has a factor level not seen in fitting.
    /// @param newdata  Rows about to be predicted.
    void validate_categorical_levels(const dstruct::DataFrame& newdata) const;

    Formula    formula_; ///< The parsed formula.
    DesignInfo design_;  ///< Terms, coefficient names and factor levels resolved from the data.
    LmOptions  options_; ///< Options given to the constructor.

    linalg::DenseMatrix<double> design_matrix_; ///< Weighted design \f$W^{1/2}X\f$ used to fit.
    std::optional<linalg::SparseCSR<double>> sparse_design_matrix_; ///< Owned sparse fit design; dense storage stays empty.
    bool matrix_input_{false}; ///< Whether predictors came from a direct matrix constructor.
    linalg::SparseLinearRegressionOptions sparse_options_{}; ///< Solver options retained for sparse ANOVA refits.
    std::vector<double>         response_;      ///< Weighted response \f$W^{1/2}y\f$ used to fit.

    std::vector<double>         coefficients_;    ///< \f$\hat\beta\f$.
    std::vector<double>         fitted_;          ///< \f$X\hat\beta\f$ (unweighted).
    std::vector<double>         residuals_;       ///< \f$y-X\hat\beta\f$ (unweighted).
    std::vector<double>         standard_errors_; ///< Coefficient standard errors.
    std::vector<double>         t_values_;        ///< Coefficient t statistics.
    std::vector<double>         p_values_;        ///< Coefficient p-values.
    linalg::DenseMatrix<double> covariance_;      ///< \f$\hat\sigma^2(X^\top WX)^{-1}\f$.
    linalg::DenseMatrix<double> xtx_inverse_;     ///< \f$(X^\top WX)^{-1}\f$, for leverage and prediction.

    double      r_squared_{0.0};          ///< \f$R^2\f$.
    double      adjusted_r_squared_{0.0}; ///< Adjusted \f$R^2\f$.
    double      sigma_{0.0};              ///< \f$\hat\sigma\f$.
    double      sigma2_{0.0};             ///< \f$\hat\sigma^2=\mathrm{RSS}/(n-p)\f$.
    double      f_statistic_{0.0};        ///< Overall F statistic (NaN if undefined).
    double      f_p_value_{0.0};          ///< p-value of the overall F statistic.
    std::size_t observations_{0};         ///< Fitted rows \f$n\f$.
    std::size_t degrees_of_freedom_{0};   ///< \f$n-p\f$.

    std::vector<double> leverage_;               ///< Hat-matrix diagonal.
    std::vector<double> standardized_residuals_; ///< Internally studentized residuals.
    std::vector<double> studentized_residuals_;  ///< Externally studentized residuals.
    std::vector<double> cooks_distance_;         ///< Cook's distances.
    std::vector<double> weights_;                ///< Case weights (empty for OLS).
    std::vector<std::size_t> fitted_rows_;       ///< Data row of each fitted row.
};

/// @brief F-test of a smaller model nested in a larger one, like R's `anova(reduced, full)`.
///
/// Tests whether the columns the larger model adds are all zero:
/// \f$F=\frac{(\mathrm{RSS}_0-\mathrm{RSS}_1)/(d_0-d_1)}{\mathrm{RSS}_1/d_1}\f$ on
/// \f$(d_0-d_1,\,d_1)\f$ degrees of freedom, with \f$d\f$ the residual degrees of freedom.
/// Nesting is the caller's responsibility (R does not check it either); only the row counts
/// and the degrees of freedom are checked.
///
/// @param reduced  The smaller model.
/// @param full  The larger model, fit to the same rows.
/// @return Both models' residual df and RSS, and the F test.
/// @throws std::invalid_argument  If the models were fit to different numbers of rows, or
///         @p full does not have fewer residual degrees of freedom than @p reduced.
///
/// @par Example
/// @code
/// const auto data = dm::datasets::tooth_growth();
/// const auto small = dm::stats::LM(dm::formula("len ~ dose"), data);
/// const auto large = dm::stats::LM(dm::formula("len ~ dose + supp"), data);
/// const auto test = dm::stats::anova(small, large);
/// std::print("{} {:.3f} {:.5f}\n", test.df, test.f_value, test.p_value);  // 1 11.447 0.00130
/// @endcode
///
/// @par References
/// - R `anova.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/anova.lm.html
///
/// @see LM::anova()
[[nodiscard]] NestedAnova anova(const LM& reduced, const LM& full);

} // namespace datamunge::stats
