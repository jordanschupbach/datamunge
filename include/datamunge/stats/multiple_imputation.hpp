#pragma once

/// @file multiple_imputation.hpp
/// @brief Multiple imputation by chained equations (predictive mean matching and Bayesian normal regression) and Rubin's pooling rules.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/random/random.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Univariate imputation model used for one incomplete column.
/// @par Example
/// @code
/// const auto m = dm::stats::ImputationMethod::pmm;
/// std::println("{}", m == dm::stats::ImputationMethod::norm); // false
/// @endcode
enum class ImputationMethod {
    pmm,  ///< Predictive mean matching: impute an observed value from one of the closest donors (as `mice.impute.pmm`).
    norm  ///< Bayesian linear regression: draw parameters, then add normal noise (as `mice.impute.norm`).
};

/// @brief Options for mice_impute().
/// @par Example
/// @code
/// const dm::stats::MiceOptions o;
/// std::println("{} {} {}", o.imputations, o.iterations, o.donors); // 5 10 5
/// @endcode
struct MiceOptions {
    std::size_t imputations{5};             ///< Number of completed datasets m.
    std::size_t iterations{10};             ///< Gibbs iterations per chain.
    std::size_t donors{5};                  ///< Donor pool size for predictive mean matching.
    std::vector<ImputationMethod> methods{};///< Method per column (empty: pmm for all).
    std::vector<double> shift{};            ///< Delta added to every imputed value of each column after it is imputed (empty: none); a pattern-mixture departure from MAR, as mice's post-processing.
    std::vector<bool> shift_rows{};         ///< Rows whose imputed values receive the shift (empty: all rows), e.g. dropouts in one arm.
    double ridge{1e-5};                     ///< Relative ridge added to the diagonal of X'X if it is singular.
};

/// @brief Completed datasets and chain diagnostics from mice_impute().
/// @par Example
/// @code
/// const dm::stats::MiceResult r;
/// std::println("{}", r.completed.size()); // 0
/// @endcode
struct MiceResult {
    std::vector<linalg::DenseMatrix<double>> completed;   ///< m completed copies of the data.
    std::vector<linalg::DenseMatrix<double>> chain_means; ///< Per imputation, iterations x columns: mean of the imputed values (NaN for complete columns).
    std::vector<std::size_t> incomplete_columns;          ///< Columns that were imputed, in visit order.
};

/// @brief Multiple imputation by chained equations (fully conditional specification).
///
/// Each chain starts by filling every missing value with a random draw from the observed values of
/// its column. Each iteration then visits the incomplete columns from left to right. For each one,
/// it regresses the observed values on an intercept and all other columns (at their current
/// values) and draws the regression parameters from their posterior, as `mice`'s `.norm.draw`:
/// \f$\sigma^{*2}=\mathrm{RSS}/\chi^2_{n_{obs}-p}\f$ and \f$\beta^*=\hat\beta+\sigma^*L z\f$ with
/// \f$LL^\top=(X^\top X)^{-1}\f$. It then replaces the missing values. Predictive mean matching
/// (match type 1) compares \f$x_{mis}^\top\beta^*\f$ with \f$x_{obs}^\top\hat\beta\f$ and imputes the
/// observed value of a donor drawn at random from the closest `donors`, so imputations are always
/// observed values. The normal method imputes \f$x^\top\beta^*+\sigma^*\epsilon\f$. Different chains
/// give the m imputations, whose spread reflects the uncertainty about the missing values. A
/// nonzero `shift` for a column adds that delta to its imputed values each time they are drawn,
/// which imputes under a specified departure from MAR.
/// @param data Data with NaN for missing values (numeric columns; code categories as numbers).
/// @param rng Random generator.
/// @param options Number of imputations and iterations, methods, donors and shifts.
/// @return The completed datasets and chain means.
/// @throws std::invalid_argument For an empty matrix, infinite values, a column with no observed values,
/// mismatched method or shift lengths, or zero imputations or iterations.
/// @par Example
/// @code
/// const double na = std::nan("");
/// const dm::linalg::DenseMatrix<double> x(6, 2, {1, 2, 2, 4, 3, na, 4, 8, 5, 10, 6, na});
/// dm::random::SplitMix64 rng(1);
/// const auto r = dm::stats::mice_impute(x, rng, {.imputations = 3});
/// std::println("{} {}", r.completed.size(), std::isnan(r.completed[0](2, 1))); // 3 false
/// @endcode
/// @par References
/// - S. van Buuren and K. Groothuis-Oudshoorn (2011). "mice: Multivariate imputation by chained
///   equations in R." Journal of Statistical Software 45(3), 1-67. doi:10.18637/jss.v045.i03
/// - R. J. A. Little (1988). "Missing-data adjustments in large surveys." Journal of Business and
///   Economic Statistics 6(3), 287-296. doi:10.1080/07350015.1988.10509663
/// @see pool_rubin()
[[nodiscard]] MiceResult mice_impute(const linalg::DenseMatrix<double>& data, random::RandomGenerator& rng,
                                     const MiceOptions& options = {});

/// @brief Pooled inference for one parameter by Rubin's rules.
/// @par Example
/// @code
/// const dm::stats::PooledEstimate p;
/// std::println("{}", p.df); // 0
/// @endcode
struct PooledEstimate {
    double estimate{};       ///< \f$\bar Q\f$, the mean of the m estimates.
    double within{};         ///< \f$\bar U\f$, the mean complete-data variance.
    double between{};        ///< \f$B\f$, the variance of the m estimates.
    double total{};          ///< \f$T=\bar U+(1+1/m)B\f$.
    double standard_error{}; ///< \f$\sqrt T\f$.
    double df{};             ///< Barnard–Rubin degrees of freedom (Rubin's 1987 value when the complete-data df is infinite).
    double riv{};            ///< Relative increase in variance \f$(1+1/m)B/\bar U\f$.
    double lambda{};         ///< Proportion of variance due to missingness \f$(1+1/m)B/T\f$.
    double fmi{};            ///< Fraction of missing information \f$(r+2/(\nu+3))/(r+1)\f$.
    double t_value{};        ///< \f$\bar Q/\sqrt T\f$.
    double p_value{};        ///< Two-sided t p-value.
    double lower{};          ///< Lower confidence limit.
    double upper{};          ///< Upper confidence limit.
};

/// @brief Pooled estimates for a parameter vector, with the averaged covariance matrices.
/// @par Example
/// @code
/// const dm::stats::RubinPooling r;
/// std::println("{}", r.imputations); // 0
/// @endcode
struct RubinPooling {
    std::vector<PooledEstimate> parameters; ///< Pooled inference for each parameter.
    std::vector<double> estimate;           ///< \f$\bar Q\f$.
    linalg::DenseMatrix<double> within;     ///< \f$\bar U\f$.
    linalg::DenseMatrix<double> between;    ///< \f$B\f$.
    std::size_t imputations{};              ///< m.
    double complete_df{};                   ///< Complete-data degrees of freedom used (infinity for large samples).
};

/// @brief Combines estimates and covariance matrices from m completed-data analyses by Rubin's rules.
///
/// Matches `mice::pool` (rule "rubin1987"): \f$T=\bar U+(1+1/m)B\f$, and the Barnard–Rubin
/// degrees of freedom \f$\nu=\left(\frac1{\nu_{old}}+\frac1{\nu_{obs}}\right)^{-1}\f$ with
/// \f$\nu_{old}=(m-1)/\lambda^2\f$ and \f$\nu_{obs}=\frac{\nu_{com}+1}{\nu_{com}+3}\nu_{com}(1-\lambda)\f$,
/// which reduces to \f$\nu_{old}\f$ for an infinite complete-data df \f$\nu_{com}\f$.
/// @param estimates m parameter vectors of equal length.
/// @param covariances m covariance matrices of matching size.
/// @param complete_df Complete-data residual degrees of freedom (infinity: large sample).
/// @param confidence Coverage of the t intervals.
/// @return Pooled estimates, variances, df, fractions of missing information and intervals.
/// @throws std::invalid_argument For fewer than two imputations, mismatched sizes, non-finite values,
/// nonpositive complete-data df, or confidence outside (0, 1).
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> u(1, 1, {1.0});
/// const auto r = dm::stats::pool_rubin({{1}, {2}, {3}}, {u, u, u});
/// std::println("{} {:.4f} {:.2f}", r.parameters[0].estimate, r.parameters[0].total, r.parameters[0].df); // 2 2.3333 6.12
/// @endcode
/// @par References
/// - D. B. Rubin (1987). Multiple Imputation for Nonresponse in Surveys. Wiley.
/// - J. Barnard and D. B. Rubin (1999). "Small-sample degrees of freedom with multiple imputation."
///   Biometrika 86(4), 948-955. doi:10.1093/biomet/86.4.948
/// @see pooled_wald_test(), mice_impute()
[[nodiscard]] RubinPooling pool_rubin(const std::vector<std::vector<double>>& estimates,
                                      const std::vector<linalg::DenseMatrix<double>>& covariances,
                                      double complete_df = std::numeric_limits<double>::infinity(),
                                      double confidence = 0.95);

/// @brief Multiparameter test from multiply imputed data.
/// @par Example
/// @code
/// const dm::stats::PooledWaldTest t;
/// std::println("{}", t.df1); // 0
/// @endcode
struct PooledWaldTest {
    double statistic{}; ///< \f$D_1=\bar Q^\top\tilde T^{-1}\bar Q/k\f$ with \f$\tilde T=(1+r)\bar U\f$.
    double df1{};       ///< k, the number of tested parameters.
    double df2{};       ///< Denominator df (Li–Raghunathan–Rubin, or Reiter's small-sample form).
    double riv{};       ///< Average relative increase in variance \f$r=(1+1/m)\mathrm{tr}(B\bar U^{-1})/k\f$.
    double p_value{};   ///< Upper-tail F p-value.
};

/// @brief Tests that several pooled parameters are jointly zero (the D1 statistic, as `mice::D1` and `mitml`).
/// @param pooling Result of pool_rubin().
/// @param indices Parameters to test.
/// @return D1 statistic with its F reference distribution.
/// @throws std::invalid_argument For an empty, repeated or out-of-range index set.
/// @throws std::runtime_error If the within-imputation covariance block is singular.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> u(1, 1, {1.0});
/// const auto r = dm::stats::pool_rubin({{1}, {2}, {3}}, {u, u, u});
/// std::println("{:.4f}", dm::stats::pooled_wald_test(r, {0}).statistic); // 1.7143
/// @endcode
/// @par References
/// - K. H. Li, T. E. Raghunathan and D. B. Rubin (1991). "Large-sample significance levels from
///   multiply imputed data using moment-based statistics and an F reference distribution." JASA
///   86(416), 1065-1073. doi:10.1080/01621459.1991.10475152
/// - J. P. Reiter (2007). "Small-sample degrees of freedom for multi-component significance tests
///   with multiple imputation for missing data." Biometrika 94(2), 502-508. doi:10.1093/biomet/asm028
/// @see pool_rubin()
[[nodiscard]] PooledWaldTest pooled_wald_test(const RubinPooling& pooling, const std::vector<std::size_t>& indices);

/// @brief Analysis applied to each completed dataset: returns the parameter estimates and their covariance.
using ImputedAnalysis =
    std::function<std::pair<std::vector<double>, linalg::DenseMatrix<double>>(const linalg::DenseMatrix<double>&)>;

/// @brief Options for tipping_point_analysis().
/// @par Example
/// @code
/// const dm::stats::TippingPointOptions o{.column = 1, .deltas = {0, -1, -2}};
/// std::println("{} {}", o.deltas.size(), o.alpha); // 3 0.05
/// @endcode
struct TippingPointOptions {
    std::size_t column{};                                      ///< Incomplete column whose imputations are shifted.
    std::vector<bool> shifted_rows{};                          ///< Rows (pattern or arm) whose imputations are shifted (empty: all).
    std::vector<double> deltas{};                              ///< Departures from MAR to evaluate, in the order to scan.
    std::size_t parameter{};                                   ///< Index of the parameter of interest in the analysis output.
    double alpha{0.05};                                        ///< Significance level defining the tipping point.
    double complete_df{std::numeric_limits<double>::infinity()}; ///< Complete-data df for pooling.
    MiceOptions mice{};                                        ///< Imputation settings (its shift and shift_rows are overridden).
    std::uint64_t seed{1};                                     ///< Seed reused for every delta (common random numbers).
};

/// @brief Pooled result at one departure from MAR.
/// @par Example
/// @code
/// const dm::stats::TippingPointRow r{.delta = -1.5};
/// std::println("{}", r.delta); // -1.5
/// @endcode
struct TippingPointRow {
    double delta{};          ///< Shift added to the imputed values.
    PooledEstimate result;   ///< Rubin-pooled inference for the parameter of interest.
};

/// @brief Sensitivity of a multiply imputed conclusion to departures from MAR.
/// @par Example
/// @code
/// const dm::stats::TippingPointResult r;
/// std::println("{}", r.tipping_delta.has_value()); // false
/// @endcode
struct TippingPointResult {
    std::vector<TippingPointRow> rows;   ///< One pooled result per delta, in scan order.
    std::optional<double> tipping_delta; ///< Delta at which the p-value first crosses alpha relative to the first delta's conclusion (linear interpolation), if it does.
    std::optional<double> null_delta;    ///< Delta at which the pooled estimate first changes sign (linear interpolation), if it does.
};

/// @brief Pattern-mixture delta-adjustment sensitivity analysis with tipping-point search.
///
/// For each delta, imputes the data with mice_impute() under MAR and adds delta to the imputed
/// values of `column` in `shifted_rows` (for example, dropouts in the active arm), so that missing
/// outcomes are assumed systematically better or worse than MAR predicts. It then analyzes each
/// completed dataset with `analysis` and pools the results by Rubin's rules. The same seed is used
/// for every delta, so differences between rows reflect the shift rather than Monte Carlo noise.
/// The tipping point is the departure at which the conclusion (significant or not at alpha)
/// first changes. Judging whether such a departure is plausible is a substantive question that
/// the analysis cannot answer.
/// @param data Data with NaN for missing values.
/// @param analysis Completed-data analysis returning estimates and their covariance matrix.
/// @param options Column, rows, delta grid, parameter, level and imputation settings.
/// @return Pooled results by delta and the interpolated tipping and null deltas.
/// @throws std::invalid_argument For an empty delta grid, an out-of-range column or parameter, a
/// shifted-rows mask of the wrong length, or invalid alpha; and as for mice_impute() and pool_rubin().
/// @par Example
/// @code
/// const double na = std::nan("");
/// const dm::linalg::DenseMatrix<double> x(6, 2, {0, 1.1, 1, 2.2, 2, na, 3, 3.9, 4, na, 5, 6.1});
/// const auto mean = [](const dm::linalg::DenseMatrix<double>& c) {
///     double s = 0;
///     for (std::size_t i = 0; i < c.rows(); ++i) s += c(i, 1) / c.rows();
///     return std::pair{std::vector<double>{s}, dm::linalg::DenseMatrix<double>(1, 1, 0.1)};
/// };
/// const auto r = dm::stats::tipping_point_analysis(x, mean, {.column = 1, .deltas = {0, -3}});
/// std::println("{:.3f}", r.rows[0].result.estimate - r.rows[1].result.estimate); // 1.000
/// @endcode
/// @par References
/// - J. R. Carpenter and M. G. Kenward (2013). Multiple Imputation and its Application. Wiley, chapter 10.
/// - National Research Council (2010). The Prevention and Treatment of Missing Data in Clinical
///   Trials. National Academies Press. doi:10.17226/12955
/// @see mice_impute(), pool_rubin()
[[nodiscard]] TippingPointResult tipping_point_analysis(const linalg::DenseMatrix<double>& data,
                                                        const ImputedAnalysis& analysis,
                                                        const TippingPointOptions& options);

} // namespace datamunge::stats
