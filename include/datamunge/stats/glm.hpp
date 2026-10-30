#pragma once

/// @file glm.hpp
/// @brief Formula-based canonical-link GLMs, IRLS traces and mean-response inference.

#include <cstddef>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Response family and fixed default link; alternative links are not implemented.
/// @see GLM
enum class GLMFamily {
  Gaussian, ///< Continuous response, identity link and constant variance.
  Binomial, ///< Exactly zero/one response, logit link and variance mu*(1-mu).
  Poisson,  ///< Nonnegative response, log link and variance mu; fractional values fit a quasi-count
            ///< model.
  Gamma     ///< Strictly positive response, inverse link and variance mu*mu.
};

/// @brief Controls canonical-link IRLS; defaults follow R's glm.control().
/// @par Example
/// @code
/// dm::stats::GLMOptions opt;
/// opt.family = dm::stats::GLMFamily::Binomial;
/// opt.max_iter = 100;
/// @endcode
/// @see GLM
struct GLMOptions {
  GLMFamily family{GLMFamily::Gaussian}; ///< Response family and its fixed link.
  std::optional<std::string>
      weights_column; ///< Positive finite prior weights; null weights in retained rows are errors.
  std::size_t max_iter{25}; ///< Positive upper bound on accepted IRLS updates.
  double      tol{1e-8};    ///< Positive finite relative-deviance tolerance.
};

/// @brief Prediction output: means alone or pointwise confidence limits for the mean.
/// @see GLM::predict()
enum class GLMPredictionInterval {
  None,      ///< Only the response-scale mean vector is populated.
  Confidence ///< Also return delta-method standard errors and back-transformed link-scale limits.
};

/// @brief Owned prediction vectors aligned to query rows; omitted rows contain NaN.
/// Confidence limits describe the conditional mean, never a future observation.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d; d.add_column("y", std::vector<double>{1,2,3,4});
/// const dm::stats::GLM m(d,"y ~ 1");
/// const auto p = m.predict(d,dm::stats::GLMPredictionInterval::Confidence);
/// std::print("{:.1f}\n",p.fit[0]); // 2.5
/// @endcode
/// @see GLM::predict()
struct GLMPrediction {
  std::vector<double> fit;    ///< Response-scale means, one per query row.
  std::vector<double> se_fit; ///< Delta-method response-scale standard errors, empty for None.
  std::vector<double> lower;  ///< Ordered response-scale lower limits, empty for None.
  std::vector<double> upper;  ///< Ordered response-scale upper limits, empty for None.
};

/// @brief Maximum-likelihood canonical-link regression by iteratively reweighted least squares.
///
/// Formula-null rows are omitted jointly; all retained design/response values and
/// prior weights must be finite. Requires n > p > 0 and full column rank. Categorical
/// responses are not auto-encoded. Binomial responses must be exactly zero or one;
/// fractional nonnegative Poisson responses permit fitting but have no count-likelihood AIC.
/// Fractional binomial prior weights use a power likelihood, not R's rounded-trial AIC.
/// Read converged() and boundary() before using inference: deviance convergence alone
/// does not establish existence of a finite MLE, especially under logistic separation.
/// A finite last iterate is retained at max_iter; invalid/rank-deficient steps throw.
///
/// This implementation explicitly accumulates the QR factor Q: fitting K updates
/// costs O(K*n*n*p + K*p*p*p) work and O(n*n + n*p + K*p) storage.
/// Const operations share no mutable caches; concurrent const use is supported when
/// input frames are not mutated. Returned references live as long as the model.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;
/// d.add_column("y",std::vector<double>{0,0,1,1});
/// const dm::stats::GLM m(d,"y ~ 1",{.family=dm::stats::GLMFamily::Binomial});
/// std::print("{:.1f} {}\n",m.fitted_values()[0],m.converged()); // 0.5 true
/// @endcode
/// @par References
/// - J. A. Nelder and R. W. M. Wedderburn (1972), Generalized Linear Models,
///   JRSS A 135(3), 370-384. https://doi.org/10.2307/2344614
/// - R Core Team, glm: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html
/// - A. Albert and J. A. Anderson (1984), On the existence of maximum likelihood
///   estimates in logistic regression models. https://doi.org/10.1093/biomet/71.1.1
/// @see GLMOptions, LM
class GLM {
 public:
  /// @brief Fits immediately using a formula string and the supplied data.
  /// @param data Training frame; formula-null rows are omitted.
  /// @param formula Numeric response and model terms, with an intercept by default.
  /// @param options Family, prior weights and IRLS stopping controls.
  /// @throws std::invalid_argument Invalid controls, family, values, weights or dimensions.
  /// @throws std::runtime_error No complete rows, rank deficiency or an invalid IRLS step.
  /// @par Example
  /// See the runnable example on GLM.
  GLM(const dstruct::DataFrame& data, const std::string& formula, GLMOptions options = {});

  /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
  ///
  /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
  /// converts to Formula implicitly, so `GLM("y ~ x", data)` also works.
  /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
  /// @param data  Data frame holding every column the formula names.
  /// @param options  Same as in the `(data, formula, ...)` constructor.
  /// @throws std::invalid_argument As the data-first constructor.
  /// @throws std::runtime_error As the data-first constructor.
  /// @par Example
  /// See GLM's class example; `GLM(dm::formula("y ~ 1"),d)` is equivalent.
  GLM(const Formula& formula, const dstruct::DataFrame& data, GLMOptions options = {})
      : GLM(data, formula.text(), std::move(options)) {}

  /// @brief Original formula text.
  /// @return Original formula text.
  [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
  /// @brief Selected family and fixed link.
  /// @return Selected family and fixed link.
  [[nodiscard]] GLMFamily family() const { return options_.family; }
  /// @brief Whether the expanded design contains an intercept.
  /// @return Whether the expanded design contains an intercept.
  [[nodiscard]] bool has_intercept() const { return design_.has_intercept; }
  /// @brief Retained training-row count n.
  /// @return Retained training-row count n.
  [[nodiscard]] std::size_t observations() const { return observations_; }
  /// @brief Number of full-rank design columns p.
  /// @return Number of full-rank design columns p.
  [[nodiscard]] std::size_t rank() const { return coefficients_.size(); }
  /// @brief Residual degrees of freedom n-p.
  /// @return Residual degrees of freedom n-p.
  [[nodiscard]] std::size_t degrees_of_freedom() const { return degrees_of_freedom_; }

  /// @brief Coefficient vector in expanded-design order.
  /// @return Coefficient vector in expanded-design order.
  [[nodiscard]] const std::vector<double>& coefficients() const { return coefficients_; }
  /// @brief Names in the same order as coefficients().
  /// @return Names in the same order as coefficients().
  [[nodiscard]] const std::vector<std::string>& coefficient_names() const {
    return design_.coefficient_names;
  }
  /// @brief Fitted response-scale means for retained rows.
  /// @return Fitted response-scale means for retained rows.
  [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
  /// @brief Fitted link-scale values X*beta for retained rows.
  /// @return Fitted link-scale values X*beta for retained rows.
  [[nodiscard]] const std::vector<double>& linear_predictors() const { return linear_predictor_; }
  /// @brief Signed square roots of prior-weighted unit deviances for retained rows.
  /// @return Signed square roots of prior-weighted unit deviances for retained rows.
  [[nodiscard]] const std::vector<double>& residuals() const { return deviance_residuals_; }
  /// @brief sqrt(prior_weight)*(y-mu)/sqrt(V(mu)), without dispersion scaling.
  /// @return sqrt(prior_weight)*(y-mu)/sqrt(V(mu)), without dispersion scaling.
  [[nodiscard]] const std::vector<double>& pearson_residuals() const { return pearson_residuals_; }
  /// @brief Weighted deviance residuals divided by sqrt(dispersion*max(1e-12,1-h)).
  /// @return Weighted deviance residuals divided by sqrt(dispersion*max(1e-12,1-h)).
  [[nodiscard]] const std::vector<double>& standardized_residuals() const {
    return standardized_residuals_;
  }
  /// @brief Final-weight hat diagonals h; their sum is p up to rounding.
  /// @return Final-weight hat diagonals h; their sum is p up to rounding.
  [[nodiscard]] const std::vector<double>& leverage() const { return leverage_; }
  /// @brief Square roots of the coefficient-covariance diagonal.
  /// @return Square roots of the coefficient-covariance diagonal.
  [[nodiscard]] const std::vector<double>& standard_errors() const { return standard_errors_; }
  /// @brief Coefficient/SE; z for binomial/Poisson, t for Gaussian/Gamma.
  /// @return Coefficient/SE; z for binomial/Poisson, t for Gaussian/Gamma.
  [[nodiscard]] const std::vector<double>& test_statistics() const { return test_statistics_; }
  /// @brief Two-sided normal or residual-df Student reference p-values.
  /// @return Two-sided normal or residual-df Student reference p-values.
  [[nodiscard]] const std::vector<double>& p_values() const { return p_values_; }
  /// @brief Coefficient covariance: dispersion*(X-transpose*W*X)^-1 at the fitted means.
  /// @return Coefficient covariance: dispersion*(X-transpose*W*X)^-1 at the fitted means.
  [[nodiscard]] const linalg::DenseMatrix<double>& covariance() const { return covariance_; }

  /// @brief Sum of prior-weighted unit deviances, without dispersion scaling.
  /// @return Sum of prior-weighted unit deviances, without dispersion scaling.
  [[nodiscard]] double deviance() const { return deviance_; }
  /// @brief Null-model deviance: weighted mean if intercepted, inverse_link(0) otherwise.
  /// @return Null-model deviance: weighted mean if intercepted, inverse_link(0) otherwise.
  [[nodiscard]] double null_deviance() const { return null_deviance_; }
  /// @brief One for binomial/Poisson; Pearson chi-square/(n-p) for Gaussian/Gamma.
  /// @return One for binomial/Poisson; Pearson chi-square/(n-p) for Gaussian/Gamma.
  [[nodiscard]] double dispersion() const { return dispersion_; }
  /// @brief R-family AIC including the dispersion parameter when estimated; NaN for fractional
  /// Poisson responses.
  /// @return R-family AIC including the dispersion parameter when estimated; NaN for fractional
  /// Poisson responses. Fractional binomial weights use a Bernoulli power likelihood rather than
  /// rounded R trials. Gamma follows R's family convention using deviance/sum(weights)
  /// inside its density, an approximation to a fully maximized shape likelihood; the
  /// Pearson dispersion used for coefficient inference is a separate estimate.
  [[nodiscard]] double aic() const { return aic_; }

  /// @brief Owned two-sided Wald limits for one coefficient.
  /// @see confidence_intervals()
  struct Interval {
    double lower; ///< Lower coefficient limit on the link scale.
    double upper; ///< Upper coefficient limit on the link scale.
  };
  /// @brief Two-sided Wald coefficient limits, using z or residual-df t critical values.
  /// @param level Finite confidence probability strictly between zero and one.
  /// @return One ordered interval per coefficient; these are not profile-likelihood limits.
  /// @throws std::invalid_argument Invalid level.
  /// @par Example
  /// See coefficient_table(), which includes these same limits.
  [[nodiscard]] std::vector<Interval> confidence_intervals(double level = 0.95) const;

  /// @brief R-style coefficient/deviance summary with convergence and boundary flags.
  /// @return Owned formatted text; no I/O occurs.
  /// @par Example
  /// See print_summary().
  [[nodiscard]] std::string summary() const;
  /// @brief Writes summary() to a caller-owned stream.
  /// @param os Destination, mutated by insertion.
  /// @par Example
  /// See the standard-output overload.
  void print_summary(std::ostream& os) const;
  /// @brief Writes summary() to standard output.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d; d.add_column("y",std::vector<double>{1,2,3,4});
  /// const dm::stats::GLM m(d,"y ~ 1");
  /// std::ostringstream out; m.print_summary(out);
  /// std::print("{}\n",out.str()==m.summary()); // true
  /// @endcode
  void print_summary() const;

  /// @brief Response-scale conditional means aligned to every query row.
  /// @param newdata Predictors; response and weights are not required.
  /// @return Query-sized vector, NaN at formula-null predictor rows.
  /// @throws std::invalid_argument Nonfinite expanded values, unknown levels, missing columns or
  /// invalid Gamma queries.
  /// @throws std::runtime_error No complete query rows (formula-engine convention).
  /// @par Example
  /// See the detailed predict() overload.
  [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;
  /// @brief Predicts conditional means and optional pointwise mean-response confidence limits.
  /// @param newdata Query frame; null predictors yield NaN, response is ignored.
  /// @param interval None or Confidence; no observation/predictive intervals are implemented.
  /// @param level Finite confidence probability strictly between zero and one, even for None.
  /// @return Owned vectors. SE is delta-method; limits transform link-scale z/t endpoints.
  /// For Gamma inverse links, a confidence interval reaching eta<=0 has upper limit infinity;
  /// a query with eta<=0 is rejected. Limits are pointwise and condition on the fitted model.
  /// @throws std::invalid_argument Invalid level/enum, nonfinite query design, unknown levels or
  /// Gamma eta<=0.
  /// @throws std::runtime_error No complete query rows.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d; d.add_column("y",std::vector<double>{0,0,1,1});
  /// const dm::stats::GLM m(d,"y ~ 1",{.family=dm::stats::GLMFamily::Binomial});
  /// const auto p=m.predict(d,dm::stats::GLMPredictionInterval::Confidence);
  /// std::print("{:.1f}\n",p.fit[0]); // 0.5
  /// @endcode
  [[nodiscard]] GLMPrediction predict(
      const dstruct::DataFrame& newdata,
      GLMPredictionInterval     interval,
      double                    level = 0.95) const;
  /// @brief Tabular predictions with columns fit and, for Confidence, se_fit/lwr/upr.
  /// @param newdata Query frame as for predict().
  /// @param interval Output detail as for predict().
  /// @param level Confidence probability as for predict().
  /// @return Owned query-aligned frame; missing predictions are numeric NaNs.
  /// @throws std::invalid_argument As for predict().
  /// @throws std::runtime_error As for predict().
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d; d.add_column("y",std::vector<double>{1,2,3,4});
  /// const dm::stats::GLM m(d,"y ~ 1");
  /// std::print("{:.1f}\n",m.predict_frame(d).double_at("fit",0)); // 2.5
  /// @endcode
  [[nodiscard]] dstruct::DataFrame predict_frame(
      const dstruct::DataFrame& newdata,
      GLMPredictionInterval     interval = GLMPredictionInterval::None,
      double                    level    = 0.95) const;

  /// @brief Returns whether the finite relative-deviance criterion was met.
  /// @return A numerical stopping flag, not a separation or MLE-existence test.
  [[nodiscard]] bool converged() const { return converged_; }
  /// @brief Count of completed IRLS updates.
  /// @return Actual iterations, between one and max_iter.
  [[nodiscard]] std::size_t iterations() const { return iteration_deviance_.size(); }
  /// @brief Flags a binomial fitted probability within 1e-8 of zero or one.
  /// @return False for other families; a heuristic warning, not a formal separation test.
  [[nodiscard]] bool boundary() const { return boundary_; }
  /// @brief Original zero-based row indices of retained training rows.
  /// @return Borrowed indices aligned to fitted_values() and all residuals.
  [[nodiscard]] const std::vector<std::size_t>& kept_rows() const { return fitted_rows_; }

  /// @brief Returns one row per accepted IRLS update and its coefficient vector.
  /// @return Owned columns iteration (one-based), deviance, relative_change, step_fraction
  /// and beta:<coefficient name>. First change compares against the initialized means.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d; d.add_column("y",std::vector<double>{0,0,1,1});
  /// const dm::stats::GLM m(d,"y ~ 1",{.family=dm::stats::GLMFamily::Binomial});
  /// const auto trace=m.iteration_history();
  /// std::print("{}\n",trace.nrows()==m.iterations()); // true
  /// @endcode
  [[nodiscard]] dstruct::DataFrame iteration_history() const;

  /// @brief Reconstructs inputs to a particular IRLS update without refitting.
  /// @param iteration Zero-based accepted update index; zero means initialized per-row means.
  /// @return Owned row/response/mu/eta/weight/z columns; weight includes prior weights.
  /// @throws std::out_of_range If iteration >= iterations().
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d; d.add_column("y",std::vector<double>{0,0,1,1});
  /// const dm::stats::GLM m(d,"y ~ 1",{.family=dm::stats::GLMFamily::Binomial});
  /// const auto w=m.working_data();
  /// std::print("{:.4f}\n",w.double_at("weight",0)); // 0.1875
  /// @endcode
  [[nodiscard]] dstruct::DataFrame working_data(std::size_t iteration = 0) const;

  /// @brief Coefficients, uncertainty and Wald tests in an owned frame.
  /// @param level Finite confidence probability strictly between zero and one.
  /// @return term/estimate/std_error/statistic/p_value/lower/upper columns.
  /// @throws std::invalid_argument Invalid level.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d; d.add_column("y",std::vector<double>{1,2,3,4});
  /// const dm::stats::GLM m(d,"y ~ 1");
  /// std::print("{:.1f}\n",m.coefficient_table().double_at("estimate",0)); // 2.5
  /// @endcode
  [[nodiscard]] dstruct::DataFrame coefficient_table(double level = 0.95) const;

  /// @brief Training rows, fitted means and weighted residual/leverage diagnostics.
  /// @return Owned row/response/fitted/eta/prior_weight/working_weight/deviance_residual/
  /// pearson_residual/standardized_residual/leverage columns, in retained-row order.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d; d.add_column("y",std::vector<double>{1,2,3,4});
  /// const dm::stats::GLM m(d,"y ~ 1");
  /// std::print("{:.1f}\n",m.training_details().double_at("fitted",0)); // 2.5
  /// @endcode
  [[nodiscard]] dstruct::DataFrame training_details() const;

  /// @brief One-row model and numerical-status summary.
  /// @return Owned
  /// family/link/n/p/df/deviance/null_deviance/dispersion/aic/iterations/converged/boundary
  /// columns.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d; d.add_column("y",std::vector<double>{1,2,3,4});
  /// const dm::stats::GLM m(d,"y ~ 1");
  /// std::print("{:.0f}\n",m.model_summary().double_at("n",0)); // 4
  /// @endcode
  [[nodiscard]] dstruct::DataFrame model_summary() const;

  /// @brief Scores binary predictions on an external response-bearing frame.
  /// @param data Binomial response and predictors; joint formula-null rows are omitted.
  /// @param threshold Finite classification threshold strictly between zero and one; p>=threshold
  /// predicts one.
  /// @return One owned row: n/accuracy/brier/log_loss; unweighted even for a weighted fit.
  /// Probabilities are clipped only at machine epsilon when computing logs.
  /// @throws std::invalid_argument Non-binomial model, invalid threshold/response or nonfinite
  /// design.
  /// @throws std::runtime_error No complete scoring rows.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d; d.add_column("y",std::vector<double>{0,0,1,1});
  /// const dm::stats::GLM m(d,"y ~ 1",{.family=dm::stats::GLMFamily::Binomial});
  /// std::print("{:.2f}\n",m.binomial_score(d).double_at("brier",0)); // 0.25
  /// @endcode
  [[nodiscard]] dstruct::DataFrame binomial_score(
      const dstruct::DataFrame& data,
      double                    threshold = 0.5) const;

  /// @brief Raw weighted deviance residuals against response-scale means.
  /// @return Owned editable plot; no rendering or I/O occurs.
  /// @par Example
  /// See save_diagnostic_plots().
  [[nodiscard]] graphics::RPlot plot_residuals_vs_fitted() const;
  /// @brief Normal-quantile plot of standardized deviance residuals; not a binomial normality test.
  /// @return Owned editable plot; no rendering or I/O occurs.
  /// @par Example
  /// See save_diagnostic_plots().
  [[nodiscard]] graphics::RPlot plot_normal_qq() const;
  /// @brief Square-root absolute standardized deviance residuals against fitted means.
  /// @return Owned editable plot; no rendering or I/O occurs.
  /// @par Example
  /// See save_diagnostic_plots().
  [[nodiscard]] graphics::RPlot plot_scale_location() const;
  /// @brief Standardized deviance residuals against final-weight leverage.
  /// @return Owned editable plot; no rendering or I/O occurs.
  /// @par Example
  /// See save_diagnostic_plots().
  [[nodiscard]] graphics::RPlot plot_residuals_vs_leverage() const;
  /// @brief Saves the four diagnostic plots as SVG files.
  /// @param path_prefix Path prepended to the four method-name suffixes; parent directory must
  /// exist.
  /// @throws std::runtime_error Plot rendering or file I/O fails.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d; d.add_column("y",std::vector<double>{1,2,3,4});
  /// const dm::stats::GLM m(d,"y ~ 1");
  /// const auto plot=m.plot_residuals_vs_fitted();
  /// m.save_diagnostic_plots("glm_demo");
  /// @endcode
  void save_diagnostic_plots(const std::string& path_prefix) const;

 private:
  /// @brief Validates inputs, fits IRLS and computes inference in place.
  /// @param data Training frame.
  /// @param options Fit configuration.
  void fit(const dstruct::DataFrame& data, GLMOptions options);
  /// @brief Rejects unseen query factor levels using the stored design.
  /// @param newdata Prediction frame.
  void validate_categorical_levels(const dstruct::DataFrame& newdata) const;

  Formula    formula_; ///< Original parsed formula.
  DesignInfo design_;  ///< Resolved design and categorical levels.
  GLMOptions options_; ///< Validated fit options.

  linalg::DenseMatrix<double> design_matrix_; ///< Retained unweighted expanded predictors.
  std::vector<double>         response_;      ///< Retained original-scale numeric response.
  std::vector<double>         prior_weights_; ///< Retained prior weights, empty for all ones.

  std::vector<double> coefficients_;       ///< Last accepted coefficient vector.
  std::vector<double> fitted_;             ///< Last response-scale means.
  std::vector<double> linear_predictor_;   ///< Last link-scale predictors.
  std::vector<double> deviance_residuals_; ///< Signed square roots of weighted deviances.
  std::vector<double> pearson_residuals_;  ///< Weighted Pearson residuals.
  std::vector<double>
      standardized_residuals_;           ///< Dispersion/leverage-standardized deviance residuals.
  std::vector<double>         leverage_; ///< Final-weight hat diagonals.
  std::vector<double>         standard_errors_; ///< Coefficient standard errors.
  std::vector<double>         test_statistics_; ///< Coefficient/SE statistics.
  std::vector<double>         p_values_;        ///< Two-sided Wald reference probabilities.
  linalg::DenseMatrix<double> covariance_;      ///< Dispersion-scaled coefficient covariance.
  linalg::DenseMatrix<double> xtx_inverse_; ///< Unscaled final-weight information inverse from QR.
  std::vector<double>         final_weights_; ///< IRLS weights recomputed at fitted means.

  std::vector<std::size_t> fitted_rows_; ///< Original zero-based retained rows.
  std::vector<std::vector<double>>
                      iteration_coefficients_; ///< Accepted coefficient vectors in time order.
  std::vector<double> iteration_deviance_;     ///< Deviance after each accepted update.
  std::vector<double> iteration_change_; ///< Relative deviance changes against the previous means.
  std::vector<double> iteration_steps_;  ///< Accepted fraction of the proposed coefficient step.
  bool                converged_{false}; ///< Relative-deviance stopping criterion met.
  bool                boundary_{false};  ///< Binomial near-zero/one heuristic.

  double      deviance_{0.0};         ///< Final weighted deviance.
  double      null_deviance_{0.0};    ///< Weighted null-model deviance.
  double      dispersion_{1.0};       ///< Fixed one or estimated Pearson dispersion.
  double      aic_{0.0};              ///< Family AIC with parameter penalty.
  std::size_t observations_{0};       ///< Retained row count.
  std::size_t degrees_of_freedom_{0}; ///< Retained n minus full-rank p.
};

} // namespace datamunge::stats
