#pragma once
/// @file inla_mixed_model.hpp
/// @brief Formula-based dense Gaussian-strategy INLA with proper bounded factor-scale priors.

#include <cstddef>
#include <cstdint>
#include <datamunge/bayes/inla.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/formula.hpp>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Supported conditional observation families, with fixed unit dispersion outside Gaussian.
/// @par Example
/// @code
/// dm::stats::INLAMixedModelOptions o;o.family=dm::stats::INLAMixedModelFamily::Binomial;
/// @endcode
/// @see INLAMixedModelOptions
enum class INLAMixedModelFamily {
  Gaussian, ///< Normal observations, identity link, integrated residual log SD.
  Binomial, ///< Individual Bernoulli zero/one outcomes, logit link (not grouped counts).
  Poisson   ///< Nonnegative integer counts, log link without an exposure offset.
};

/// @brief Prior and numerical controls for a single-grouping-factor mixed model.
/// Factor entries have independent proper uniform priors inside the declared box:
/// diagonal Lambda in [epsilon,bound], off-diagonal Lambda in [-bound,bound].
/// Gaussian residual log sigma is uniform on a finite, data-centered interval; this
/// empirical prior construction limits interpretations of evidence across datasets.
/// These are priors on factor entries/log sigma, not uniform priors on variances.
/// @par Example
/// @code
/// dm::stats::INLAMixedModelOptions o;
/// o.fixed_effect_prior_sd=2.5;o.lambda_bound=3;o.grid_points_per_dim=31;
/// @endcode
/// @see INLAMixedModel, bayes::INLAOptions
struct INLAMixedModelOptions {
  INLAMixedModelFamily           family{INLAMixedModelFamily::Gaussian}; ///< Conditional family.
  bayes::INLAIntegrationStrategy strategy{
      bayes::INLAIntegrationStrategy::Grid}; ///< Grid or mode-only EB.
  double fixed_effect_prior_sd{
      1000}; ///< Independent zero-mean normal fixed-effect prior SD, positive finite.
  double lambda_diag_epsilon{1e-3}; ///< Strictly positive factor diagonal lower support bound.
  double lambda_bound{5};           ///< Finite factor bound, greater than epsilon.
  double log_sigma_search_radius{
      3}; ///< Positive Gaussian log-sigma support half-width around log response SD.
  std::size_t grid_points_per_dim{
      7}; ///< Odd >=3 per theta axis for Grid; dimension <=4, at most one million points.
  double        grid_span{4}; ///< Positive whitened-grid half-span in curvature SD units.
  std::size_t   mode_population_size{40};  ///< DE population, at least four.
  std::size_t   mode_max_generations{200}; ///< Positive DE generation budget.
  std::uint64_t seed{42};                  ///< DE seed.
};
/// @brief Approximate Bayesian mixed model using dense Gaussian-strategy INLA.
/// Supports one lme4-style random term, e.g. y~x+(1+x|group). Latent coordinates
/// are fixed effects followed by one random-effect block per group. At each theta
/// the conditional Gaussian is centered at the mode; hyperparameter integration
/// combines those Gaussians. No marginal skew correction, sparse solver or exact
/// posterior sampler is supplied. See INLAMixedModelOptions for the proper priors.
/// Complete cases across response/fixed predictors/group/random slopes are fitted;
/// nonfinite retained numeric inputs are rejected. All fitted storage is owned.
/// Units/centering change the meaning of the independent fixed-effect priors.
/// @par Example
/// @code
/// auto o=dm::stats::INLAMixedModelOptions{};
/// o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
/// o.mode_population_size=8;o.mode_max_generations=4;
/// const dm::stats::INLAMixedModel m(dm::datasets::tooth_growth(),"len ~ dose + (1 | supp)",o);
/// std::print("{}\n",m.observations()); // 60
/// @endcode
/// @par References
/// - Rue, Martino and Chopin (2009), JRSS B 71, 319-392.
///   https://doi.org/10.1111/j.1467-9868.2008.00700.x
/// @see bayes::INLA, INLAMixedModelOptions
class INLAMixedModel {
 public:
  /// @brief Fit immediately from a DataFrame and a fixed/random-effects formula.
  /// @param data Training frame, borrowed only during construction.
  /// @param formula Single random-effects term and supported fixed formula.
  /// @param options Prior/numerical controls, defaults from INLAMixedModelOptions.
  /// @throws std::invalid_argument Invalid formula/schema/options/outcomes or too few complete
  /// rows/groups.
  /// @throws std::runtime_error No usable converged INLA fit at the selected hyperparameter mode.
  /// @par Example
  /// @code
  /// auto o=dm::stats::INLAMixedModelOptions{};
  /// o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
  /// o.mode_population_size=8;o.mode_max_generations=4;
  /// const dm::stats::INLAMixedModel m(dm::datasets::tooth_growth(),"len ~ dose + (1 | supp)",o);
  /// std::print("{}\n",m.observations()); // 60
  /// @endcode
  /// @see INLAMixedModel, used_rows()
  INLAMixedModel(
      const dstruct::DataFrame& data,
      const std::string&        formula,
      INLAMixedModelOptions     options = {});

  /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
  ///
  /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
  /// converts to Formula implicitly, so `INLAMixedModel("y ~ x", data)` also works.
  /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
  /// @param data  Data frame holding every column the formula names.
  /// @param options  Same as in the `(data, formula, ...)` constructor.
  /// @throws std::invalid_argument As the data-first constructor.
  /// @throws std::runtime_error As the data-first constructor.
  /// @par Example
  /// See the class example, with Formula(formula) supplied first.
  /// @see INLAMixedModel
  INLAMixedModel(
      const Formula&            formula,
      const dstruct::DataFrame& data,
      INLAMixedModelOptions     options = {})
      : INLAMixedModel(data, formula.text(), std::move(options)) {}

  /// @brief Fitted formula text, borrowed for the model lifetime.
  /// @return Fitted formula text, borrowed for the model lifetime.
  [[nodiscard]] const std::string& formula_text() const { return formula_text_; }
  /// @brief Observation family name: gaussian, binomial or poisson; owned string.
  /// @return Observation family name: gaussian, binomial or poisson; owned string.
  [[nodiscard]] std::string family() const;
  /// @brief Fitted grouping column name, borrowed.
  /// @return Fitted grouping column name, borrowed.
  [[nodiscard]] const std::string& group_variable() const { return group_variable_; }
  /// @brief Ordered random-intercept/slope names, borrowed.
  /// @return Ordered random-intercept/slope names, borrowed.
  [[nodiscard]] const std::vector<std::string>& random_effect_names() const {
    return random_effect_names_;
  }
  /// @brief Number of retained complete fitting rows.
  /// @return Number of retained complete fitting rows.
  [[nodiscard]] std::size_t observations() const { return observations_; }
  /// @brief Number of fitted grouping levels.
  /// @return Number of fitted grouping levels.
  [[nodiscard]] std::size_t num_groups() const { return group_labels_.size(); }
  /// @brief Group keys in first retained occurrence order, borrowed.
  /// @return Group keys in first retained occurrence order, borrowed.
  [[nodiscard]] const std::vector<std::string>& group_labels() const { return group_labels_; }

  /// @brief Approximate mixture means of fixed effects, borrowed.
  /// @return Approximate mixture means of fixed effects, borrowed.
  [[nodiscard]] const std::vector<double>& fixed_effects_mean() const {
    return fixed_effects_mean_;
  }
  /// @brief Total approximate mixture marginal SDs of fixed effects, borrowed.
  /// @return Total approximate mixture marginal SDs of fixed effects, borrowed.
  [[nodiscard]] const std::vector<double>& fixed_effects_sd() const { return fixed_effects_sd_; }
  /// @brief Fixed-effect column names in fitted design order, borrowed.
  /// @return Fixed-effect column names in fitted design order, borrowed.
  [[nodiscard]] const std::vector<std::string>& coefficient_names() const {
    return design_.coefficient_names;
  }

  /// @brief Approximate random-effect means by group then term, borrowed.
  /// @return Approximate random-effect means by group then term, borrowed.
  [[nodiscard]] const std::vector<std::vector<double>>& random_effects_mean() const {
    return random_effects_mean_;
  }
  /// @brief Total random-effect marginal SDs by group then term, borrowed.
  /// @return Total random-effect marginal SDs by group then term, borrowed.
  [[nodiscard]] const std::vector<std::vector<double>>& random_effects_sd() const {
    return random_effects_sd_;
  }

  /// @brief Random-effect SDs evaluated at the posterior mean packed factor.
  /// This is a plug-in transform; it generally differs from the posterior mean SD.
  /// @return Owned SD vector in random_effect_names() order.
  /// @par Example
  /// @code
  /// auto o=dm::stats::INLAMixedModelOptions{};
  /// o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
  /// o.mode_population_size=8;o.mode_max_generations=4;
  /// const dm::stats::INLAMixedModel m(dm::datasets::tooth_growth(),"len ~ dose + (1 | supp)",o);
  /// std::print("{}\n",m.random_effect_std_devs().size()); // 1
  /// @endcode
  /// @see variance_components_frame()
  [[nodiscard]] std::vector<double> random_effect_std_devs() const;
  /// @brief Gaussian residual exp(E[log sigma]), a plug-in SD rather than E[sigma].
  /// @return Positive transformed mean log SD.
  /// @throws std::logic_error Family is not Gaussian.
  /// @par Example
  /// @code
  /// auto o=dm::stats::INLAMixedModelOptions{};
  /// o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
  /// o.mode_population_size=8;o.mode_max_generations=4;
  /// const dm::stats::INLAMixedModel m(dm::datasets::tooth_growth(),"len ~ dose + (1 | supp)",o);
  /// std::print("{}\n",m.residual_std_dev()>0); // true
  /// @endcode
  /// @see variance_components_frame()
  [[nodiscard]] double residual_std_dev() const;

  /// @brief Approximate integrated log evidence; NaN under EB.
  /// @return Approximate integrated log evidence; NaN under EB.
  [[nodiscard]] double log_marginal_likelihood() const { return log_marginal_likelihood_; }

  /// @brief Return a human-readable approximate-posterior summary without mutation.
  /// @return Owned formula/family/mean/SD text; EB evidence prints NaN.
  /// @par Example
  /// @code
  /// auto o=dm::stats::INLAMixedModelOptions{};
  /// o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
  /// o.mode_population_size=8;o.mode_max_generations=4;
  /// const dm::stats::INLAMixedModel m(dm::datasets::tooth_growth(),"len ~ dose + (1 | supp)",o);
  /// std::print("{}\n",!m.summary().empty()); // true
  /// @endcode
  /// @see print_summary()
  [[nodiscard]] std::string summary() const;
  /// @brief Write the same text as summary() to a stream (modifies stream formatting).
  /// @param os Borrowed output stream.
  /// @par Example
  /// @code
  /// auto o=dm::stats::INLAMixedModelOptions{};
  /// o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
  /// o.mode_population_size=8;o.mode_max_generations=4;
  /// const dm::stats::INLAMixedModel m(dm::datasets::tooth_growth(),"len ~ dose + (1 | supp)",o);
  /// std::ostringstream out;m.print_summary(out);
  /// std::print("{}\n",!out.str().empty()); // true
  /// @endcode
  /// @see summary()
  void print_summary(std::ostream& os) const;
  /// @brief Print summary() to standard output.
  /// @par Example
  /// See print_summary(std::ostream&); the no-argument overload uses std::cout.
  /// @see summary()
  void print_summary() const;

  /// @brief Inverse link of the approximate posterior mean latent predictor.
  /// This is a plug-in response prediction, not an integrated posterior predictive
  /// mean for logit/log links. Seen groups use their fitted random effects;
  /// missing/null/unseen groups use fixed effects with random effects set to zero.
  /// In particular that fallback does not integrate over a new group's uncertainty.
  /// Rows with missing/nonfinite numeric predictors/slopes return NaN in input order.
  /// An empty query frame returns an empty vector.
  /// @param newdata Matching fitted fixed-predictor schema and any random slope columns.
  /// @return Owned response-scale vector with newdata.nrows() entries.
  /// @throws std::invalid_argument Wrong predictor/group types, missing slope columns,
  /// or nonfinite numeric group keys.
  /// @throws std::runtime_error Unseen fixed categorical predictor level.
  /// @throws std::out_of_range Missing required fixed-predictor column.
  /// @par Example
  /// @code
  /// auto o=dm::stats::INLAMixedModelOptions{};
  /// o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
  /// o.mode_population_size=8;o.mode_max_generations=4;
  /// const dm::stats::INLAMixedModel m(dm::datasets::tooth_growth(),"len ~ dose + (1 | supp)",o);
  /// std::print("{}\n",m.predict(dm::datasets::tooth_growth()).size()); // 60
  /// @endcode
  /// @see posterior(), fixed_effects_frame()
  [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

  /// @brief Borrow the full fitted Gaussian-mixture result and grid diagnostics.
  /// @return Immutable posterior result, valid for the model lifetime.
  const bayes::INLAResult& posterior() const { return posterior_; }
  /// @brief Borrow original training indices of complete retained rows.
  /// @return Immutable source row indices, in fitting order.
  const std::vector<std::size_t>& used_rows() const { return used_rows_; }
  /// @brief Borrow the retained latent design [X|group-placed Z] for replay checks.
  /// @return Immutable observations-by-latent-coordinates matrix.
  const linalg::DenseMatrix<double>& latent_design() const { return latent_design_; }
  /// @brief Borrow complete retained responses in latent_design() row order.
  /// @return Immutable outcomes.
  const std::vector<double>& response() const { return response_; }
  /// @brief Named fixed-effect approximate means/SDs and within/between variance terms.
  /// @return Owned term/mean/sd/within_variance/between_variance table.
  /// @par Example
  /// @code
  /// auto o=dm::stats::INLAMixedModelOptions{};
  /// o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
  /// o.mode_population_size=8;o.mode_max_generations=4;
  /// const dm::stats::INLAMixedModel m(dm::datasets::tooth_growth(),"len ~ dose + (1 | supp)",o);
  /// std::print("{}\n",m.fixed_effects_frame().nrows()); // 2
  /// @endcode
  /// @see random_effects_frame()
  dstruct::DataFrame fixed_effects_frame() const;
  /// @brief Named random effects with total SD and within/between variance terms.
  /// @return Owned group/term/mean/sd/within_variance/between_variance/q025/q975 table.
  /// Endpoints are equal-tail quantiles of the approximate Gaussian mixture.
  /// @par Example
  /// @code
  /// auto o=dm::stats::INLAMixedModelOptions{};
  /// o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
  /// o.mode_population_size=8;o.mode_max_generations=4;
  /// const dm::stats::INLAMixedModel m(dm::datasets::tooth_growth(),"len ~ dose + (1 | supp)",o);
  /// std::print("{}\n",m.random_effects_frame().nrows()); // 2
  /// @endcode
  /// @see fixed_effects_frame()
  dstruct::DataFrame random_effects_frame() const;
  /// @brief Transform every grid point before averaging physical random/residual SDs.
  /// O(K*q^2) work and O(K*q) storage. No input mutation/shared state. These moments
  /// are approximate; their values depend on the finite factor-scale hyperpriors.
  /// @return component/mean_sd/sd_of_sd/sd_at_mean_theta in random-term then residual order.
  /// @par Example
  /// @code
  /// auto o=dm::stats::INLAMixedModelOptions{};
  /// o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
  /// o.mode_population_size=8;o.mode_max_generations=4;
  /// const dm::stats::INLAMixedModel m(dm::datasets::tooth_growth(),"len ~ dose + (1 | supp)",o);
  /// std::print("{}\n",m.variance_components_frame().nrows()); // 2
  /// @endcode
  /// @see random_effect_std_devs(), residual_std_dev()
  dstruct::DataFrame variance_components_frame() const;

 private:
  /// @brief Resolve complete cases, build latent precision/design and store the INLA mixture.
  /// @param data Borrowed training frame.
  /// @throws std::invalid_argument Invalid training schema, responses or prior controls.
  /// @throws std::runtime_error Failed INLA mode/grid fitting.
  void fit(const dstruct::DataFrame& data);

  std::string              formula_text_;           ///< Owned original formula.
  std::string              group_variable_;         ///< Grouping column name.
  bool                     random_intercept_{true}; ///< Whether group blocks contain an intercept.
  std::vector<std::string> random_slope_vars_;      ///< Numeric random-slope columns.
  std::vector<std::string> random_effect_names_;    ///< Group-block coefficient names.

  Formula               fixed_formula_; ///< Resolved fixed-effects formula.
  DesignInfo            design_;        ///< Frozen fitted fixed-effects schema.
  INLAMixedModelOptions options_;       ///< Owned prior and numerical controls.

  std::vector<std::string>              group_labels_; ///< First-occurrence group keys.
  std::vector<std::size_t>              group_index_;  ///< Retained row to group mapping.
  std::vector<std::vector<std::size_t>> group_rows_;   ///< Retained row indices by group.

  std::vector<double>              fixed_effects_mean_;  ///< Approximate fixed-effect means.
  std::vector<double>              fixed_effects_sd_;    ///< Approximate fixed-effect SDs.
  std::vector<std::vector<double>> random_effects_mean_; ///< Group-by-term approximate means.
  std::vector<std::vector<double>> random_effects_sd_;   ///< Group-by-term approximate SDs.
  std::vector<double>              theta_mean_; ///< Approximate theta means on fitted scale.
  std::size_t lambda_free_count_{0};            ///< Number of packed covariance-factor parameters.

  double                   log_marginal_likelihood_{0.0}; ///< Integrated log evidence, NaN for EB.
  std::size_t              observations_{0};              ///< Number of complete fitting rows.
  bayes::INLAResult        posterior_;        ///< Full retained Gaussian-mixture result.
  std::vector<std::size_t> used_rows_;        ///< Retained original training row indices.
  linalg::DenseMatrix<double> latent_design_; ///< Retained [X|Z] latent observation matrix.
  std::vector<double>         response_;      ///< Retained responses.
  bool                        group_is_numeric_{false}; ///< Fitted grouping storage type.
};

} // namespace datamunge::stats
