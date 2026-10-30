#pragma once

/// @file glmm.hpp
/// @brief Single-grouping-factor PQL models, working-step diagnostics and prediction targets.

#include <cstddef>
#include <cstdint>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/formula.hpp>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Conditional response distribution with dispersion fixed at one.
enum class GLMMFamily {
  Binomial, ///< Exactly zero/one observations with the logit link.
  Poisson   ///< Nonnegative integer counts with the log link; no exposure offset is fitted.
};

/// @brief PQL relinearization and bounded covariance-search controls.
/// @par Example
/// @code
/// dm::stats::GLMMOptions opt;
/// opt.family=dm::stats::GLMMFamily::Poisson;
/// opt.fixed_random_intercept_sd=1.0; // condition on a known link-scale SD
/// @endcode
/// @see GLMM
struct GLMMOptions {
  GLMMFamily family{
      GLMMFamily::Binomial};      ///< Conditional family; alternative links are not implemented.
  std::size_t max_iterations{20}; ///< Positive maximum number of outer working-model updates.
  double tol{1e-6}; ///< Positive finite relative infinity-norm tolerance for beta, group effects
                    ///< and covariance factor.
  std::size_t de_population_size{30}; ///< At least four DE candidates when covariance is estimated.
  std::size_t de_max_generations{150}; ///< Positive DE generation limit per outer update.
  double      theta_bound{
      5.0}; ///< Positive finite bound on covariance-factor entries; diagonals are nonnegative.
  std::uint64_t seed{42}; ///< DE seed, reset at each outer update for reproducibility.
  std::optional<double>
      fixed_random_intercept_sd; ///< Optional finite nonnegative known SD; requires (1|group),
                                 ///< bypasses covariance search.
};

/// @brief Fits binary or count mixed models by a maximum-working-likelihood PQL approximation.
///
/// A single lme4-style random term supports an intercept and plain numeric slopes,
/// e.g. `y ~ x + (1 + x | group)`. Fixed terms use Formula. Formula-null, group-null
/// and random-slope-null rows are omitted jointly; retained numeric values must be finite.
/// Requires n>p>0, full fixed rank and at least two retained groups. Binomial proportions
/// without trial counts and fractional Poisson counts are rejected. No weights/offsets,
/// nested/crossed groups or observation-level dispersion parameter are implemented.
/// Working and stored training means are bounded at 1e-8 (and 1-1e-8 for binomial)
/// for finite relinearization; predictions use the direct stable inverse link.
///
/// Each outer iteration freezes a working response/weights and estimates covariance
/// by profiled Gaussian ML (not REML). DE plus bounded local polishing is approximate.
/// This is not exact GLMM marginal maximum likelihood: working_criterion(), historical
/// deviance(), aic() and bic() are working-Gaussian quantities, unsuitable for original
/// response likelihood comparisons. Wald standard errors condition on estimated covariance.
///
/// The Woodbury solve factors q-by-q matrices within each group and a p-by-p fixed
/// information matrix. One objective evaluation costs O(n*(p*p+p*q+q*q)+G*q*q*q+p*p*p),
/// using O(n*(p+q)+G*q*q+p*p) workspace. Recording K states additionally stores
/// O(K*(p+G*q+q*q)). Const methods have no mutable caches and support concurrent reads
/// of immutable input frames. Borrowed references live as long as the model.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;
/// d.add_column("y",std::vector<double>{1,3,5,7});
/// d.add_column("g",std::vector<std::string>{"A","A","B","B"});
/// dm::stats::GLMMOptions opt; opt.family=dm::stats::GLMMFamily::Poisson;
/// opt.fixed_random_intercept_sd=1.0;
/// const dm::stats::GLMM m(d,"y ~ 1 + (1 | g)",opt);
/// std::print("{} {}\n",m.observations(),m.num_groups()); // 4 2
/// @endcode
/// @par References
/// - N. E. Breslow and D. G. Clayton (1993), Approximate Inference in Generalized
///   Linear Mixed Models, JASA 88(421), 9-25. https://doi.org/10.1080/01621459.1993.10594284
/// - R MASS glmmPQL: https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/glmmPQL.html
///   The present working covariance fit uses ML rather than that routine's default REML.
/// @see GLM, LMM, GLMMOptions
class GLMM {
 public:
  /// @brief Fits the model immediately from a formula string.
  /// @param data Training frame; rows incomplete for either design or group are dropped.
  /// @param formula Fixed formula plus exactly one random-effects term.
  /// @param options Family, iteration/search controls and optional known intercept SD.
  /// @throws std::invalid_argument Invalid options, formula, values, response domain or dimensions.
  /// @throws std::runtime_error No complete rows, rank deficiency or a nonfinite/numerically
  /// invalid working fit.
  /// @par Example
  /// See the runnable class example.
  GLMM(const dstruct::DataFrame& data, const std::string& formula, GLMMOptions options = {});

  /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
  ///
  /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
  /// converts to Formula implicitly, so `GLMM("y ~ x + (1|g)", data)` also works.
  /// @param formula  Model formula, e.g. `dm::formula("y ~ x + (1|g)")`.
  /// @param data  Data frame holding every column the formula names.
  /// @param options  Same as in the `(data, formula, ...)` constructor.
  /// @throws std::invalid_argument As the data-first constructor.
  /// @throws std::runtime_error As the data-first constructor.
  /// @par Example
  /// See GLMM; `GLMM(dm::formula("y ~ 1 + (1|g)"),d,opt)` is equivalent.
  GLMM(const Formula& formula, const dstruct::DataFrame& data, GLMMOptions options = {})
      : GLMM(data, formula.text(), std::move(options)) {}

  /// @brief Original mixed-model formula text.
  /// @return Original mixed-model formula text.
  [[nodiscard]] const std::string& formula_text() const { return formula_text_; }
  /// @brief Conditional response family name.
  /// @return Owned string binomial or poisson.
  [[nodiscard]] std::string family() const;
  /// @brief Grouping column name.
  /// @return Grouping column name.
  [[nodiscard]] const std::string& group_variable() const { return group_variable_; }
  /// @brief Whether the random design contains an intercept.
  /// @return Whether the random design contains an intercept.
  [[nodiscard]] bool has_random_intercept() const { return random_intercept_; }
  /// @brief Names of the q random-effect coordinates, in covariance order.
  /// @return Names of the q random-effect coordinates, in covariance order.
  [[nodiscard]] const std::vector<std::string>& random_effect_names() const {
    return random_effect_names_;
  }
  /// @brief Number of retained training observations n.
  /// @return Number of retained training observations n.
  [[nodiscard]] std::size_t observations() const { return observations_; }
  /// @brief Number of retained distinct groups G.
  /// @return Number of retained distinct groups G.
  [[nodiscard]] std::size_t num_groups() const { return group_labels_.size(); }
  /// @brief Full fixed-effects column count p.
  /// @return Full fixed-effects column count p.
  [[nodiscard]] std::size_t rank() const { return coefficients_.size(); }
  /// @brief Completed PQL outer updates.
  /// @return Completed PQL outer updates.
  [[nodiscard]] std::size_t iterations() const { return iterations_used_; }

  /// @brief Borrowed final fixed coefficients in design order.
  /// @return Borrowed final fixed coefficients in design order.
  [[nodiscard]] const std::vector<double>& coefficients() const { return coefficients_; }
  /// @brief Borrowed fixed-effect names in coefficient order.
  /// @return Borrowed fixed-effect names in coefficient order.
  [[nodiscard]] const std::vector<std::string>& coefficient_names() const {
    return design_.coefficient_names;
  }
  /// @brief Conditional working-Gaussian standard errors of the fixed effects.
  /// @return Conditional working-Gaussian standard errors of the fixed effects.
  [[nodiscard]] const std::vector<double>& standard_errors() const { return standard_errors_; }
  /// @brief Fixed coefficient divided by its working-model standard error.
  /// @return Fixed coefficient divided by its working-model standard error.
  [[nodiscard]] const std::vector<double>& z_values() const { return z_values_; }
  /// @brief Two-sided normal Wald reference probabilities; approximate under PQL.
  /// @return Two-sided normal Wald reference probabilities; approximate under PQL.
  [[nodiscard]] const std::vector<double>& p_values() const { return p_values_; }
  /// @brief Borrowed conditional response means at retained training rows.
  /// @return Borrowed conditional response means at retained training rows.
  [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }

  /// @brief Borrowed link-scale covariance D=Lambda*Lambda-transpose.
  /// @return Borrowed link-scale covariance D=Lambda*Lambda-transpose.
  [[nodiscard]] const linalg::DenseMatrix<double>& random_effect_covariance() const {
    return random_effect_covariance_;
  }
  /// @brief Link-scale random-effect standard deviations.
  /// @return Owned square roots of the covariance diagonal, length q.
  /// @par Example
  /// See random_effects_frame().
  [[nodiscard]] std::vector<double> random_effect_std_devs() const;
  /// @brief Correlation between two random-effect coordinates.
  /// @param i Zero-based coordinate, less than q.
  /// @param j Zero-based coordinate, less than q.
  /// @return Covariance standardized by both SDs; zero if either variance is zero.
  /// @throws std::out_of_range Coordinate outside [0,q).
  /// @par Example
  /// See random_effects_frame(); `m.random_effect_correlation(0,0)` is one for positive variance.
  [[nodiscard]] double random_effect_correlation(std::size_t i, std::size_t j) const;

  /// @brief Borrowed group labels in first-retained-appearance order.
  /// @return Borrowed group labels in first-retained-appearance order.
  [[nodiscard]] const std::vector<std::string>& group_labels() const { return group_labels_; }
  /// @brief Borrowed working-Gaussian BLUP vectors on the link scale, aligned to group_labels().
  /// @return Borrowed working-Gaussian BLUP vectors on the link scale, aligned to group_labels().
  [[nodiscard]] const std::vector<std::vector<double>>& random_effects() const {
    return random_effects_;
  }

  /// @brief Historical alias for working_criterion(); not a response-family or marginal GLMM
  /// deviance.
  /// @return Historical alias for working_criterion(); not a response-family or marginal GLMM
  /// deviance.
  [[nodiscard]] double deviance() const { return deviance_; }
  /// @brief Historical working criterion plus twice the fitted parameter count.
  /// @return working_criterion()+2*k; k excludes covariance when SD is fixed.
  /// This is not marginal GLMM AIC and should not select original-response models.
  /// @par Example
  /// See model_summary(), which deliberately labels the underlying criterion.
  [[nodiscard]] double aic() const;
  /// @brief Historical working criterion plus log(n) times the fitted parameter count.
  /// @return working_criterion()+log(n)*k; not a marginal GLMM BIC.
  /// @par Example
  /// See model_summary().
  [[nodiscard]] double bic() const;

  /// @brief Formats the working-model fit and numerical status.
  /// @return Owned text with explicit PQL/working-criterion labels.
  /// @par Example
  /// See print_summary().
  [[nodiscard]] std::string summary() const;
  /// @brief Writes summary text to a caller-owned stream.
  /// @param os Destination stream, mutated by insertion.
  /// @par Example
  /// See the standard-output overload.
  void print_summary(std::ostream& os) const;
  /// @brief Writes summary text to standard output.
  /// @par Example
  /// @code
  /// const auto
  /// d=dm::stats::simulate_glmm_random_intercepts(dm::stats::GLMMFamily::Poisson,4,5,1,.3,.4,42);
  /// dm::stats::GLMMOptions opt; opt.family=dm::stats::GLMMFamily::Poisson;
  /// opt.fixed_random_intercept_sd=.4; const dm::stats::GLMM m(d,"y ~ x + (1|group)",opt);
  /// std::ostringstream os; m.print_summary(os);
  /// std::print("{}\n",os.str()==m.summary()); // true
  /// @endcode
  void print_summary() const;

  /// @brief Conditional means for known groups, or zero-random-effect means otherwise.
  /// @param newdata Query fixed predictors; group may be missing, null or unseen.
  /// @return Query-sized owned vector with NaN at null fixed predictors or, for a known
  /// group, null required random slopes. A missing required slope column is an error.
  /// Unknown/null/missing groups use b=0, which is not integration over a new group.
  /// @throws std::invalid_argument Nonfinite values, unknown fixed categorical levels,
  /// missing needed random slopes, mismatched grouping-column type or response-mean overflow.
  /// @throws std::out_of_range Missing fixed columns.
  /// @throws std::runtime_error No complete fixed-design rows.
  /// @par Example
  /// See prediction_frame().
  [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

  /// @brief Conditional mean with all random effects set to zero.
  /// @param newdata Fixed predictors only; groups and random slopes are ignored.
  /// @return Query-sized owned vector, NaN at null fixed predictors; not a marginal mean.
  /// @throws std::invalid_argument Nonfinite predictors, unknown factor levels or overflow.
  /// @throws std::out_of_range Missing fixed columns.
  /// @throws std::runtime_error No complete fixed-design rows.
  /// @par Example
  /// See prediction_frame().
  [[nodiscard]] std::vector<double> predict_fixed(const dstruct::DataFrame& newdata) const;

  /// @brief Returns conditional and zero-random-effect means with group-use status.
  /// @param newdata Query frame as for predict().
  /// @return Owned fit/fixed_fit/group_status frame; status is known, unseen, missing,
  /// or incomplete when a needed predictor is null. No uncertainty intervals are implied.
  /// @throws std::invalid_argument As for predict().
  /// @throws std::out_of_range As for predict().
  /// @throws std::runtime_error As for predict().
  /// @par Example
  /// @code
  /// const auto
  /// d=dm::stats::simulate_glmm_random_intercepts(dm::stats::GLMMFamily::Poisson,4,5,1,.3,.4,42);
  /// dm::stats::GLMMOptions opt; opt.family=dm::stats::GLMMFamily::Poisson;
  /// opt.fixed_random_intercept_sd=.4; const dm::stats::GLMM m(d,"y ~ x + (1|group)",opt);
  /// std::print("{}\n",m.prediction_frame(d).string_at("group_status",0)); // known
  /// const auto conditional=m.predict(d),fixed=m.predict_fixed(d);
  /// @endcode
  [[nodiscard]] dstruct::DataFrame prediction_frame(const dstruct::DataFrame& newdata) const;

  /// @brief Whether beta, group effects and covariance factor met the relative change tolerance.
  /// @return Numerical fixed-point flag; not a marginal-likelihood accuracy guarantee.
  [[nodiscard]] bool converged() const { return converged_; }
  /// @brief Flags a fitted covariance-factor diagonal near zero or a searched entry at its bound.
  /// @return Heuristic boundary flag using 1e-6 relative tolerance; fixed SDs only flag zero.
  [[nodiscard]] bool covariance_at_boundary() const { return covariance_at_boundary_; }
  /// @brief Final Gaussian working criterion n*log(2*pi)+log|V|+profiled residual quadratic.
  /// @return A pseudo-data quantity, not a response deviance or a marginal GLMM likelihood.
  [[nodiscard]] double working_criterion() const { return deviance_; }
  /// @brief Final fixed-effect covariance, conditional on the PQL covariance estimate.
  /// @return Borrowed p-by-p inverse working information matrix.
  [[nodiscard]] const linalg::DenseMatrix<double>& covariance() const {
    return fixed_effect_covariance_;
  }
  /// @brief Conditional linear predictors X*beta+Z*b for retained training rows.
  /// @return Borrowed vector, length observations().
  [[nodiscard]] const std::vector<double>& linear_predictors() const { return linear_predictors_; }
  /// @brief Original zero-based retained training row indices.
  /// @return Borrowed indices aligned to all training vectors.
  [[nodiscard]] const std::vector<std::size_t>& kept_rows() const { return fitted_rows_; }

  /// @brief Fixed-effect estimates, conditional working SEs and Wald z tests.
  /// @return Owned term/estimate/std_error/z_value/p_value frame.
  /// @par Example
  /// @code
  /// const auto
  /// d=dm::stats::simulate_glmm_random_intercepts(dm::stats::GLMMFamily::Poisson,4,5,1,.3,.4,42);
  /// dm::stats::GLMMOptions opt; opt.family=dm::stats::GLMMFamily::Poisson;
  /// opt.fixed_random_intercept_sd=.4; const dm::stats::GLMM m(d,"y ~ x + (1|group)",opt);
  /// std::print("{}\n",m.coefficient_frame().nrows()); // 2
  /// @endcode
  [[nodiscard]] dstruct::DataFrame coefficient_frame() const;

  /// @brief Group BLUPs in wide format, with their retained observation counts.
  /// @return Owned group/n and one column per random-effect name, in group_labels() order.
  /// @par Example
  /// @code
  /// const auto
  /// d=dm::stats::simulate_glmm_random_intercepts(dm::stats::GLMMFamily::Poisson,4,5,1,.3,.4,42);
  /// dm::stats::GLMMOptions opt; opt.family=dm::stats::GLMMFamily::Poisson;
  /// opt.fixed_random_intercept_sd=.4; const dm::stats::GLMM m(d,"y ~ x + (1|group)",opt);
  /// std::print("{}\n",m.random_effects_frame().nrows()); // 4
  /// const auto sd=m.random_effect_std_devs();
  /// std::print("{:.1f}\n",m.random_effect_correlation(0,0)); // 1.0
  /// @endcode
  [[nodiscard]] dstruct::DataFrame random_effects_frame() const;

  /// @brief Reconstructs the frozen inputs to one PQL working-model fit.
  /// @param iteration Zero-based outer update index, less than iterations().
  /// @return Owned row/group/response/eta/mu/weight/z columns in retained-row order.
  /// @throws std::out_of_range Invalid iteration.
  /// @par Example
  /// @code
  /// const auto
  /// d=dm::stats::simulate_glmm_random_intercepts(dm::stats::GLMMFamily::Poisson,4,5,1,.3,.4,42);
  /// dm::stats::GLMMOptions opt; opt.family=dm::stats::GLMMFamily::Poisson;
  /// opt.fixed_random_intercept_sd=.4; const dm::stats::GLMM m(d,"y ~ x + (1|group)",opt);
  /// std::print("{}\n",m.working_data().nrows()); // 20
  /// @endcode
  [[nodiscard]] dstruct::DataFrame working_data(std::size_t iteration = 0) const;

  /// @brief Accepted PQL update trace, including fixed coefficients and random-effect SDs.
  /// @return Owned iteration/max_relative_change/working_criterion and beta:/sd: columns.
  /// Criteria use different pseudo-data each update; monotonicity is not implied.
  /// @par Example
  /// @code
  /// const auto
  /// d=dm::stats::simulate_glmm_random_intercepts(dm::stats::GLMMFamily::Poisson,4,5,1,.3,.4,42);
  /// dm::stats::GLMMOptions opt; opt.family=dm::stats::GLMMFamily::Poisson;
  /// opt.fixed_random_intercept_sd=.4; const dm::stats::GLMM m(d,"y ~ x + (1|group)",opt);
  /// std::print("{}\n",m.iteration_history().nrows()==m.iterations()); // true
  /// @endcode
  [[nodiscard]] dstruct::DataFrame iteration_history() const;

  /// @brief Profiles one frozen working dataset at a supplied covariance factor.
  /// @param theta Packed lower-triangle entries, row-major, length q*(q+1)/2; finite.
  /// @param iteration Zero-based frozen outer update, less than iterations().
  /// @return Gaussian working criterion after profiling fixed effects; search bounds are not
  /// imposed. This supports diagnostics of a single working fit, not a GLMM likelihood profile.
  /// @throws std::invalid_argument Wrong factor size or nonfinite entries.
  /// @throws std::out_of_range Invalid iteration.
  /// @throws std::runtime_error Numerically singular working system.
  /// @par Example
  /// @code
  /// const auto
  /// d=dm::stats::simulate_glmm_random_intercepts(dm::stats::GLMMFamily::Poisson,4,5,1,.3,.4,42);
  /// dm::stats::GLMMOptions opt; opt.family=dm::stats::GLMMFamily::Poisson;
  /// opt.fixed_random_intercept_sd=.4; const dm::stats::GLMM m(d,"y ~ x + (1|group)",opt);
  /// std::print("{}\n",std::isfinite(m.working_criterion_at({.4},0))); // true
  /// @endcode
  [[nodiscard]] double working_criterion_at(
      const std::vector<double>& theta,
      std::size_t                iteration = 0) const;

  /// @brief Evaluates conditional binary predictions on response-bearing query rows.
  /// @param data Numeric binary response, fixed predictors and optional grouping/random columns.
  /// @param threshold Finite threshold strictly between zero and one; p>=threshold predicts one.
  /// @return Owned n/accuracy/brier/log_loss row after joint response/predictor omission.
  /// Known groups use their fitted effects; others use b=0. Log probabilities are epsilon-clipped.
  /// @throws std::invalid_argument Non-binomial model, invalid threshold/response or predict()
  /// errors.
  /// @throws std::runtime_error No complete scoring rows.
  /// @throws std::out_of_range Missing required columns.
  /// @par Example
  /// @code
  /// const auto
  /// d=dm::stats::simulate_glmm_random_intercepts(dm::stats::GLMMFamily::Binomial,4,5,1,.3,.4,42);
  /// dm::stats::GLMMOptions opt; opt.family=dm::stats::GLMMFamily::Binomial;
  /// opt.fixed_random_intercept_sd=.4; const dm::stats::GLMM m(d,"y ~ x + (1|group)",opt);
  /// std::print("{:.0f}\n",m.binomial_score(d).double_at("n",0)); // 20
  /// @endcode
  [[nodiscard]] dstruct::DataFrame binomial_score(
      const dstruct::DataFrame& data,
      double                    threshold = .5) const;

  /// @brief Training means and Pearson residuals, without a normality assumption.
  /// @return Owned row/group/response/fitted/fixed_fit/eta/pearson_residual columns.
  /// @par Example
  /// @code
  /// const auto
  /// d=dm::stats::simulate_glmm_random_intercepts(dm::stats::GLMMFamily::Poisson,4,5,1,.3,.4,42);
  /// dm::stats::GLMMOptions opt; opt.family=dm::stats::GLMMFamily::Poisson;
  /// opt.fixed_random_intercept_sd=.4; const dm::stats::GLMM m(d,"y ~ x + (1|group)",opt);
  /// std::print("{}\n",m.training_details().nrows()); // 20
  /// @endcode
  [[nodiscard]] dstruct::DataFrame training_details() const;

  /// @brief One-row dimensions, convergence and working-criterion summary.
  /// @return Owned family/n/groups/p/q/iterations/converged/covariance_boundary/working_criterion
  /// frame.
  /// @par Example
  /// @code
  /// const auto
  /// d=dm::stats::simulate_glmm_random_intercepts(dm::stats::GLMMFamily::Poisson,4,5,1,.3,.4,42);
  /// dm::stats::GLMMOptions opt; opt.family=dm::stats::GLMMFamily::Poisson;
  /// opt.fixed_random_intercept_sd=.4; const dm::stats::GLMM m(d,"y ~ x + (1|group)",opt); const
  /// auto table=m.model_summary(); std::print("{} {}\n",table.nrows(),m.aic()>m.working_criterion()
  /// && m.bic()>m.working_criterion()); // 1 true
  /// @endcode
  [[nodiscard]] dstruct::DataFrame model_summary() const;

 private:
  /// @brief Validates/expands designs and runs bounded working-ML PQL in place.
  /// @param data Training frame.
  void fit(const dstruct::DataFrame& data);

  std::string              formula_text_;           ///< Original formula.
  std::string              group_variable_;         ///< Single grouping column.
  bool                     random_intercept_{true}; ///< Presence of a random intercept.
  std::vector<std::string> random_slope_vars_;      ///< Plain numeric random-slope columns.
  std::vector<std::string> random_effect_names_;    ///< Names in random covariance order.

  Formula     fixed_formula_; ///< Formula with random term removed.
  DesignInfo  design_;        ///< Resolved fixed design and factor levels.
  GLMMOptions options_;       ///< Validated fitting controls.

  linalg::DenseMatrix<double> design_matrix_; ///< Retained fixed design X.
  std::vector<double>         response_;      ///< Retained raw response y.

  std::vector<std::string>              group_labels_; ///< Group labels in appearance order.
  std::vector<std::size_t>              group_index_;  ///< Group number for each retained row.
  std::vector<std::vector<std::size_t>> group_rows_;   ///< Retained-row indices within each group.
  linalg::DenseMatrix<double>           random_effect_design_; ///< Within-group random design Z.

  linalg::DenseMatrix<double> lambda_; ///< Lower-triangular link-scale covariance factor.

  std::vector<double> coefficients_;      ///< Returned fixed coefficients.
  std::vector<double> linear_predictors_; ///< Returned conditional link predictors.
  std::vector<double> fitted_;            ///< Returned conditional mean responses.
  std::vector<double> standard_errors_;   ///< Conditional working-model coefficient SEs.
  std::vector<double> z_values_;          ///< Fixed-effect Wald z statistics.
  std::vector<double> p_values_;          ///< Two-sided normal Wald probabilities.
  linalg::DenseMatrix<double>
      fixed_effect_covariance_; ///< Inverse final working fixed information.

  linalg::DenseMatrix<double> random_effect_covariance_; ///< Link-scale D=Lambda*Lambda-transpose.
  std::vector<std::vector<double>> random_effects_;      ///< Returned group working-model BLUPs.

  std::vector<std::size_t> fitted_rows_; ///< Original indices of retained rows.
  bool                     group_is_numeric_{
      false}; ///< Training grouping type, used to reject incompatible query keys.
  bool converged_{false};         ///< Full PQL state change met tolerance.
  bool covariance_at_boundary_{false}; ///< Singular-factor or search-box boundary heuristic.
  std::vector<std::vector<double>> beta_path_; ///< Initialized beta plus each accepted beta.
  std::vector<std::vector<std::vector<double>>>
      effects_path_; ///< Initialized group effects plus accepted states.
  std::vector<std::vector<double>>
                      theta_path_;  ///< Initialized covariance factor plus accepted packed factors.
  std::vector<double> change_path_; ///< Relative infinity-norm state changes.
  std::vector<double> criterion_path_; ///< Working criterion for each frozen pseudo-dataset.

  double      deviance_{0.0};      ///< Historical storage for final working criterion.
  std::size_t observations_{0};    ///< Retained row count.
  std::size_t iterations_used_{0}; ///< Actual completed outer updates.
};

/// @brief Balanced random-intercept binary/count fixture with known generating effects.
/// @param family Binomial or Poisson.
/// @param groups Number of independent groups, at least two.
/// @param observations_per_group At least two observations per group.
/// @param intercept Conditional fixed intercept on the link scale, finite.
/// @param slope Finite fixed slope; x spans [-1,1] evenly within every group.
/// @param random_intercept_sd Finite nonnegative generating link-scale SD.
/// @param seed Standard-library mt19937_64 seed; draws all group effects before responses.
/// @return Owned group/x/y/true_effect/true_mean frame; groups are named g1, g2, ... .
/// Standard distribution implementations may differ across C++ standard libraries.
/// @throws std::invalid_argument Invalid dimensions/parameters or a nonfinite Poisson rate above
/// 1e6.
/// @par Example
/// @code
/// const auto
/// d=dm::stats::simulate_glmm_random_intercepts(dm::stats::GLMMFamily::Poisson,4,5,1,.3,.4,42);
/// std::print("{} {}\n",d.nrows(),d.ncols()); // 20 5
/// @endcode
/// @see GLMM
dstruct::DataFrame simulate_glmm_random_intercepts(
    GLMMFamily    family,
    std::size_t   groups,
    std::size_t   observations_per_group,
    double        intercept,
    double        slope,
    double        random_intercept_sd,
    std::uint64_t seed = 42);

} // namespace datamunge::stats
