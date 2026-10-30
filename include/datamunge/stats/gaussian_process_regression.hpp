#pragma once

/// @file gaussian_process_regression.hpp
/// @brief Exact RBF Gaussian-process regression, profiled likelihood and conditional uncertainty diagnostics.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Preprocessing and bounded-grid empirical-Bayes settings for GaussianProcessRegression.
///
/// Signal variance is profiled at every candidate, with floor 1e-12 in standardized-response
/// units. The response is always centred/scaled using population moments, with scale one
/// when its standard deviation is below 1e-12. Predictor standardization is optional.
/// @par Example
/// @code
/// dm::stats::GaussianProcessRegressionOptions options;
/// options.length_scale=.5; options.noise_ratio=.05; // Fix these; still profile the signal variance.
/// @endcode
/// @par References
/// - C. E. Rasmussen and C. K. I. Williams (2006). Gaussian Processes for Machine Learning,
///   chapters 2 and 5. https://gaussianprocess.org/gpml/
/// @see GaussianProcessRegression
struct GaussianProcessRegressionOptions {
    double length_scale{-1.0}; ///< Positive RBF length scale; any finite negative value selects a geometric grid.
    double noise_ratio{-1.0}; ///< Noise/signal variance ratio, nonnegative; any finite negative value selects a grid.
    std::size_t n_length_scale_grid{20}; ///< Positive grid size over [0.1 sqrt(p),3 sqrt(p)]; size one uses its lower bound.
    std::size_t n_noise_grid{15}; ///< Positive grid size over [1e-4,1]; size one uses 1e-4.
    bool standardize{true}; ///< Centre/scale expanded predictors with population moments; tiny scales become one.
    double jitter_ratio{1e-8}; ///< Finite nonnegative numerical ridge relative to signal variance; excluded from new-observation noise.
};

/// @brief Owned pointwise Gaussian prediction, in the original response units.
///
/// The target is the latent function for predict_detail(), or a new noisy observation
/// for predict_observation_detail(). These plug-in intervals condition on the fitted
/// mean, scaling and hyperparameters; they are not simultaneous bands or SEs of fitted parameters.
/// Rows preserve the query frame's order; rows with a null predictor have NaN entries.
/// @par Example
/// See GaussianProcessRegression::predict_detail().
/// @see GaussianProcessRegression
struct GaussianProcessRegressionPrediction {
    std::vector<double> fit; ///< Posterior means, one per query row.
    std::vector<double> se_fit; ///< Posterior target standard deviations, not standard errors of parameter estimates.
    std::vector<double> lower; ///< Lower central pointwise Gaussian bound at the requested level.
    std::vector<double> upper; ///< Upper central pointwise Gaussian bound at the requested level.
};

/// @brief Formula-driven exact RBF Gaussian-process regression with profiled covariance scale.
///
/// Uses z=(y-mean(y))/population_sd(y), R_ij=exp(-||x_i-x_j||^2/(2 ell^2)),
/// C=R+(noise_ratio+jitter_ratio) I and K=s C. The signal variance s is max(1e-12,z'C^-1 z/n).
/// Both selection and prediction use this same K. Negative length/noise settings select
/// a finite geometric grid by marginal likelihood; this is empirical Bayes, without
/// integration over hyperparameters, centring or scaling uncertainty. Default predictor
/// standardization acts on the formula's expanded design columns; no intercept column is fitted.
/// Null training rows are dropped jointly; non-finite retained values are rejected.
/// The object owns its design, data and factorizations; const reads may run concurrently.
/// A grid of G candidates costs O(G*(n^3+n^2*p)) work and O(n^2+np+G) storage.
///
/// fitted_values(), r_squared() and rmse() retain their historical fixed-fit conditional
/// leave-one-out meaning: preprocessing, mean, scales and hyperparameters from all data
/// are held fixed. They do not refit the full pipeline in each deletion fold.
///
/// @par Example
/// @code
/// dm::dstruct::DataFrame data; data.add_column("x",std::vector<double>{0,1});
/// data.add_column("y",std::vector<double>{-1,1});
/// dm::stats::GaussianProcessRegressionOptions options;
/// options.length_scale=1/std::sqrt(2*std::log(2.0)); options.noise_ratio=.5;
/// options.standardize=false; options.jitter_ratio=0;
/// const dm::stats::GaussianProcessRegression gp(data,"y ~ x",options);
/// std::print("{:.3f}\n",gp.predict(data)[0]); // -0.500
/// @endcode
/// @par References
/// - C. E. Rasmussen and C. K. I. Williams (2006). Gaussian Processes for Machine Learning,
///   equations 2.23-2.30 and 5.12. https://gaussianprocess.org/gpml/
/// @see predict_detail(), predict_observation_detail(), training_details()
class GaussianProcessRegression {
 public:
    /// @brief Fits an exact RBF GP, with a fixed or grid-selected length scale and noise ratio.
    /// @param data Training frame; nulls in selected response/predictors are dropped jointly.
    /// @param formula Numeric-response formula, e.g. "Petal.Length ~ Petal.Width"; expanded features define RBF distance.
    /// @param options Preprocessing, finite kernel settings and positive grid sizes; defaults select both grids.
    /// @throws std::invalid_argument For invalid formula/options, non-finite retained values or overflowing standardization.
    /// @throws std::runtime_error If no complete rows remain, no candidate is finite/SPD, or final factorization fails.
    /// @par Example
    /// See the class example for a hand-checkable two-point fit.
    /// @see GaussianProcessRegression, hyperparameter_search()
    GaussianProcessRegression(const dstruct::DataFrame& data,const std::string& formula,
                              GaussianProcessRegressionOptions options={});

    /// @brief Fits from a Formula object in R's (formula,data) order.
    /// @param formula Formula or implicitly converted string, with the primary constructor's semantics.
    /// @param data Training frame, subject to the same joint null-dropping.
    /// @param options Same options as the primary constructor.
    /// @throws std::invalid_argument As in the primary constructor.
    /// @throws std::runtime_error As in the primary constructor.
    /// @par Example
    /// The class example also works with GaussianProcessRegression("y ~ x",data,options).
    /// @see GaussianProcessRegression
    GaussianProcessRegression(const Formula& formula,const dstruct::DataFrame& data,GaussianProcessRegressionOptions options={})
        : GaussianProcessRegression(data,formula.text(),std::move(options)) {}

    /// @brief Original formula text.
    /// @return Borrowed string valid until destruction or assignment.
    [[nodiscard]] const std::string& formula_text() const {return formula_.text();}
    /// @brief Names of expanded design columns used for kernel distance.
    /// @return Borrowed vector, valid until destruction or assignment.
    [[nodiscard]] const std::vector<std::string>& predictor_names() const {return predictor_names_;}
    /// @brief Number of retained paired training rows.
    /// @return Positive retained row count n.
    [[nodiscard]] std::size_t observations() const {return observations_;}
    /// @brief Input row indices retained after joint null dropping.
    /// @return Borrowed zero-based indices in training order.
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const {return kept_rows_;}
    /// @brief Length scale in standardized expanded-feature units, or original expanded units when disabled.
    /// @return Finite positive ell.
    [[nodiscard]] double length_scale() const {return length_scale_used_;}
    /// @brief Profiled signal variance in standardized-response units, preserving the original API convention.
    /// @return Positive s, at least 1e-12; multiply by response_scale() squared for original squared units.
    [[nodiscard]] double signal_variance() const {return signal_variance_used_;}
    /// @brief Observation-noise variance in standardized-response units, without numerical jitter.
    /// @return noise_ratio times signal_variance().
    [[nodiscard]] double noise_variance() const {return noise_ratio_used_*signal_variance_used_;}
    /// @brief Noise/signal variance ratio used in the fit.
    /// @return Finite nonnegative eta.
    [[nodiscard]] double noise_ratio() const {return noise_ratio_used_;}
    /// @brief Numerical ridge ratio used consistently in fitting and prediction.
    /// @return Finite nonnegative j; new-observation variance excludes this ridge.
    [[nodiscard]] double jitter_ratio() const {return options_.jitter_ratio;}
    /// @brief Fixed plug-in prior mean, the retained response mean.
    /// @return Mean in original response units.
    [[nodiscard]] double response_mean() const {return y_mean_;}
    /// @brief Population response standard deviation used for scaling, or one for a tiny/constant response.
    /// @return Positive scale in response units.
    [[nodiscard]] double response_scale() const {return y_scale_;}
    /// @brief Exact log density of standardized responses under the selected, ridge-augmented covariance.
    /// @return Normalized-scale log marginal likelihood, including the variance-floor correction.
    [[nodiscard]] double log_marginal_likelihood() const {return log_marginal_likelihood_used_;}
    /// @brief Log density on the original response scale, treating estimated preprocessing as fixed.
    /// @return log_marginal_likelihood() minus n log(response_scale()); no extra parameter penalty or integration.
    [[nodiscard]] double log_marginal_likelihood_original() const;
    /// @brief Whether length scale was requested automatically, including a one-point grid.
    /// @return True when options.length_scale was negative.
    [[nodiscard]] bool length_scale_was_selected() const {return options_.length_scale<0;}
    /// @brief Whether noise ratio was requested automatically, including a one-point grid.
    /// @return True when options.noise_ratio was negative.
    [[nodiscard]] bool noise_ratio_was_selected() const {return options_.noise_ratio<0;}
    /// @brief Actual length-scale grid, a singleton for fixed ell.
    /// @return Borrowed finite positive candidate vector in increasing order.
    [[nodiscard]] const std::vector<double>& length_scale_grid() const {return length_scale_grid_;}
    /// @brief Best standardized-response log likelihood over noise ratios at each length scale.
    /// @return Borrowed vector aligned with length_scale_grid(); failed profiles are negative infinity.
    [[nodiscard]] const std::vector<double>& length_scale_profile_log_likelihood() const {return length_scale_profile_ll_;}
    /// @brief Historical fixed-fit conditional leave-one-out means, in original response units.
    /// @return Borrowed n-vector in retained training order; see class caveat about full-data preprocessing.
    [[nodiscard]] const std::vector<double>& fitted_values() const {return fitted_;}
    /// @brief Full-data posterior means at training inputs, distinct from fitted_values().
    /// @return Borrowed n-vector in retained training order.
    [[nodiscard]] const std::vector<double>& posterior_fitted_values() const {return posterior_fitted_;}
    /// @brief Historical conditional leave-one-out 1-SSE/TSS.
    /// @return R squared; one for constant training responses.
    [[nodiscard]] double r_squared() const;
    /// @brief Historical conditional leave-one-out root mean squared residual.
    /// @return RMSE in original response units; see the class preprocessing caveat.
    [[nodiscard]] double rmse() const;

    /// @brief Human-readable settings, scale conventions and conditional deletion metrics.
    /// @return Owned summary string; no printing or mutation.
    /// @par Example
    /// @code
    /// const dm::stats::GaussianProcessRegression gp(dm::datasets::iris(),"Petal.Length ~ Petal.Width");
    /// std::print("{}",gp.summary());
    /// @endcode
    /// @see model_summary(), print_summary()
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to a caller-owned stream.
    /// @param os Output stream; its usual I/O failure policy applies.
    /// @par Example
    /// See summary(); print_summary(std::cout) writes the same text.
    /// @see summary()
    void print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    /// @par Example
    /// See summary(); replace std::print with gp.print_summary().
    /// @see summary()
    void print_summary() const;

    /// @brief Predicts posterior means in query-row order, preserving null rows as NaN.
    /// @param newdata Query predictors, with training categorical levels; empty returns empty.
    /// @return Owned means in response units; response column is not needed.
    /// @throws std::invalid_argument For missing columns, unknown levels or non-finite retained predictors.
    /// @throws std::runtime_error If a nonempty frame has no complete predictor rows or posterior numerics fail.
    /// @par Example
    /// See the class example; predict_detail() adds uncertainty to the same means.
    /// @see predict_detail()
    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    /// @brief Pointwise posterior means and standard deviations for the latent function.
    /// @param newdata Query predictors; null rows return NaNs, empty returns empty vectors.
    /// @param level Central Gaussian interval probability, finite and strictly between zero and one; default .95.
    /// @return Owned means, latent standard deviations and bounds in original response units.
    /// @throws std::invalid_argument For invalid level, schema/levels or non-finite predictors.
    /// @throws std::runtime_error For no complete query rows or numerically invalid posterior variance.
    /// @par Example
    /// @code
    /// dm::stats::GaussianProcessRegressionOptions options; options.length_scale=.5; options.noise_ratio=.05;
    /// const auto data=dm::datasets::iris();
    /// const dm::stats::GaussianProcessRegression gp(data,"Petal.Length ~ Petal.Width",options);
    /// std::print("{}\n",gp.predict_detail(data).fit.size()); // 150
    /// @endcode
    /// @par References
    /// - Rasmussen and Williams (2006), equations 2.23-2.24. https://gaussianprocess.org/gpml/chapters/RW2.pdf
    /// @see predict_observation_detail(), predict_covariance()
    [[nodiscard]] GaussianProcessRegressionPrediction predict_detail(const dstruct::DataFrame& newdata,double level=.95) const;

    /// @brief Pointwise predictive distribution of a new observation, including independent observation noise.
    /// @param newdata Query predictors, with the same null/schema semantics as predict_detail().
    /// @param level Central Gaussian interval probability in (0,1), default .95.
    /// @return Same means as predict_detail(), with noise_variance()*response_scale()^2 added to each variance.
    /// Numerical jitter is excluded from new-observation noise. Hyperparameters remain fixed.
    /// @throws std::invalid_argument For invalid level/schema/values.
    /// @throws std::runtime_error For no complete query rows or invalid posterior numerics.
    /// @par Example
    /// See prediction_table(), which displays both targets side by side.
    /// @see predict_detail(), prediction_table()
    [[nodiscard]] GaussianProcessRegressionPrediction predict_observation_detail(const dstruct::DataFrame& newdata,double level=.95) const;

    /// @brief Joint covariance between query targets, including off-diagonal posterior dependence.
    /// @param newdata Query predictors; null rows yield NaN rows/columns, empty yields a 0 by 0 matrix.
    /// @param observation_noise If true, add independent observation noise only on the diagonal; default false.
    /// @return Owned M by M covariance in squared original response units. Repeated queries have
    /// identical latent values; their independent new-observation noises remain distinct.
    /// Cost O(M*(n^2+np)+M^2*(n+p)).
    /// @throws std::invalid_argument For schema/category errors or non-finite retained predictors.
    /// @throws std::runtime_error For no complete rows or numerically invalid covariance.
    /// @par Example
    /// @code
    /// dm::stats::GaussianProcessRegressionOptions options; options.length_scale=.5; options.noise_ratio=.05;
    /// const auto data=dm::datasets::iris();
    /// const dm::stats::GaussianProcessRegression gp(data,"Petal.Length ~ Petal.Width",options);
    /// std::print("{}\n",gp.predict_covariance(data.head(2)).rows()); // 2
    /// @endcode
    /// @par References
    /// - Rasmussen and Williams (2006), equation 2.24. https://gaussianprocess.org/gpml/chapters/RW2.pdf
    /// @see predict_detail(), predict_observation_detail()
    [[nodiscard]] linalg::DenseMatrix<double> predict_covariance(const dstruct::DataFrame& newdata,bool observation_noise=false) const;

    /// @brief Tabulates latent-function and new-observation uncertainty together.
    /// @param newdata Query predictors; no response is needed, null rows remain as NaNs.
    /// @param level Central pointwise interval probability in (0,1), default .95.
    /// @return Owned numeric columns row (one-based), mean, latent_sd, observation_sd,
    /// latent_lower, latent_upper, observation_lower, observation_upper; bounds use original units.
    /// @throws std::invalid_argument For invalid level/schema/values.
    /// @throws std::runtime_error As in predict_detail().
    /// @par Example
    /// @code
    /// dm::stats::GaussianProcessRegressionOptions options; options.length_scale=.5; options.noise_ratio=.05;
    /// const auto data=dm::datasets::iris();
    /// const dm::stats::GaussianProcessRegression gp(data,"Petal.Length ~ Petal.Width",options);
    /// std::print("{}\n",gp.prediction_table(data.head(2)).nrows()); // 2
    /// @endcode
    /// @see predict_detail(), predict_observation_detail()
    [[nodiscard]] dstruct::DataFrame prediction_table(const dstruct::DataFrame& newdata,double level=.95) const;

    /// @brief Training posterior and conditional deletion diagnostics, aligned to retained source rows.
    /// @return Owned columns row (one-based source index), observed, posterior_mean, loo_mean,
    /// loo_residual, loo_observation_sd, loo_z. The deletion mean holds all fitted preprocessing
    /// and hyperparameters fixed; its observation variance excludes numerical jitter on the held-out
    /// target. A zero SD gives NaN z. Costs O(n) after the fit's inverse-diagonal calculation.
    /// @par Example
    /// See model_summary(); training_details() returns one row per retained observation.
    /// @par References
    /// - Rasmussen and Williams (2006), equation 5.12. https://gaussianprocess.org/gpml/chapters/RW5.pdf
    /// @see fitted_values(), posterior_fitted_values(), score()
    [[nodiscard]] dstruct::DataFrame training_details() const;

    /// @brief All length/noise grid evaluations, including failed candidates.
    /// @return Owned columns length_scale, noise_ratio, signal_variance (standardized y),
    /// log_likelihood (standardized y), log_likelihood_original and selected (0/1).
    /// Failed candidates have NaN amplitude and negative-infinite likelihood. The first maximum wins ties.
    /// @par Example
    /// See model_summary(); hyperparameter_search() provides the underlying candidate table.
    /// @see model_summary(), length_scale_profile_log_likelihood()
    [[nodiscard]] dstruct::DataFrame hyperparameter_search() const {return search_;}

    /// @brief One-row model summary with explicitly named original-unit variances.
    /// @return Owned columns n, length_scale, noise_ratio, jitter_ratio, signal_variance_y,
    /// noise_variance_y, log_likelihood_y, conditional_loo_rmse, conditional_loo_r_squared,
    /// posterior_rmse. Likelihood treats empirical preprocessing as fixed.
    /// @par Example
    /// @code
    /// dm::stats::GaussianProcessRegressionOptions options; options.length_scale=.5; options.noise_ratio=.05;
    /// const dm::stats::GaussianProcessRegression gp(dm::datasets::iris(),"Petal.Length ~ Petal.Width",options);
    /// std::print("{}\n",gp.model_summary().nrows()); // 1
    /// @endcode
    /// @see training_details(), hyperparameter_search(), score()
    [[nodiscard]] dstruct::DataFrame model_summary() const;

    /// @brief Scores supplied responses using the existing fit, without refitting or tuning.
    ///
    /// The caller must exclude these rows from training to interpret this as held-out performance.
    /// @param data Frame with query predictors and the original response column. Null response/query rows are omitted.
    /// @param level Central interval probability in (0,1), default .95.
    /// @return Owned one-row columns n, rmse, mae, r_squared, latent_coverage, observation_coverage,
    /// mean_observation_width and mean_nlpd (Gaussian negative log predictive density).
    /// Coverage is empirical, not a guarantee; NLPD is NaN if any predictive variance is zero.
    /// Constant-response R squared is one for zero SSE, otherwise zero.
    /// @throws std::invalid_argument For invalid level/schema/non-finite responses or no scoreable rows.
    /// @throws std::runtime_error As in prediction_table().
    /// @par Example
    /// See model_summary(); gp.score(heldout) assesses external predictions, whereas training_details() diagnoses deletions.
    /// @see prediction_table(), training_details()
    [[nodiscard]] dstruct::DataFrame score(const dstruct::DataFrame& data,double level=.95) const;

    /// @brief Scatter with posterior mean and a pointwise latent-function interval for one raw numeric predictor.
    /// @param data Frame containing the named predictor and response, used only for the scatter overlay.
    /// @param grid_resolution Number of grid points across the training range plus 5 percent padding; at least two, default 200.
    /// @param level Central latent interval probability in (0,1), default .95.
    /// @return Editable RPlot in original units; transformed/multiple design columns are unsupported.
    /// @throws std::invalid_argument For unsupported predictor, invalid level/grid or invalid overlay schema/values.
    /// @throws std::runtime_error For prediction failures.
    /// @par Example
    /// @code
    /// dm::stats::GaussianProcessRegressionOptions options; options.length_scale=.5; options.noise_ratio=.05;
    /// const auto data=dm::datasets::iris();
    /// const dm::stats::GaussianProcessRegression gp(data,"Petal.Length ~ Petal.Width",options);
    /// gp.plot_fit(data).size(800,500).save_svg("gpr_fit.svg");
    /// @endcode
    /// @see predict_detail(), prediction_table()
    [[nodiscard]] graphics::RPlot plot_fit(const dstruct::DataFrame& data,std::size_t grid_resolution=200,double level=.95) const;
    /// @brief Plots conditional deletion means against retained training responses.
    /// @return Editable RPlot in original units; no pipeline refitting is performed.
    /// @par Example
    /// See plot_fit(); replace its plot call with gp.plot_predicted_vs_actual().
    /// @see training_details()
    [[nodiscard]] graphics::RPlot plot_predicted_vs_actual() const;
    /// @brief Plots conditional deletion residuals against conditional deletion means.
    /// @return Editable RPlot with a zero reference, in original response units.
    /// @par Example
    /// See plot_fit(); replace its plot call with gp.plot_residuals_vs_fitted().
    /// @see training_details()
    [[nodiscard]] graphics::RPlot plot_residuals_vs_fitted() const;
    /// @brief Plots the standardized-response profile likelihood on the searched length grid.
    /// @return Editable RPlot; each ordinate is maximized over the noise-ratio grid, including a singleton grid.
    /// @throws std::invalid_argument If length scale was fixed rather than requested automatically.
    /// @par Example
    /// See summary(); gp.plot_length_scale_profile() plots the default automatic fit's profile.
    /// @see hyperparameter_search(), length_scale_profile_log_likelihood()
    [[nodiscard]] graphics::RPlot plot_length_scale_profile() const;

 private:
    /// @brief Validates inputs, searches the profiled likelihood and caches conditional deletion diagnostics.
    /// @param data Training frame; no caller-owned data are mutated.
    /// @throws std::invalid_argument For invalid data/options/standardization.
    /// @throws std::runtime_error For missing complete rows or numerical failure.
    void fit(const dstruct::DataFrame& data);
    /// @brief Cross-covariance between one standardized query and every retained training row.
    /// @param query_std Finite expanded query of the training design width.
    /// @return n-vector in standardized-response covariance units, with signal amplitude included.
    std::vector<double> kernel_vector(const std::vector<double>& query_std) const;
    /// @brief Shared pointwise prediction engine for latent or noisy targets.
    /// @param newdata Query predictor frame.
    /// @param level Valid central Gaussian probability.
    /// @param observation_noise Whether to add physical observation noise to the variance.
    /// @return Owned row-aligned predictions, with NaNs for null predictor rows.
    /// @throws std::invalid_argument For invalid level, schema or values.
    /// @throws std::runtime_error For empty complete-case design or invalid numerics.
    GaussianProcessRegressionPrediction predict_impl(const dstruct::DataFrame& newdata,double level,bool observation_noise) const;

    Formula formula_; ///< Owned original formula text.
    DesignInfo design_; ///< Owned training-resolved formula expansion and categorical levels.
    GaussianProcessRegressionOptions options_; ///< Original validated kernel/preprocessing choices.
    std::vector<std::string> predictor_names_; ///< Expanded distance-coordinate labels, without an intercept.
    std::vector<double> feature_mean_; ///< Per-feature centring constants, zero when disabled.
    std::vector<double> feature_scale_; ///< Positive per-feature population SDs, one when disabled/tiny.
    linalg::DenseMatrix<double> training_X_std_; ///< Retained expanded coordinates after optional standardization.
    std::vector<double> training_y_; ///< Retained original-scale responses.
    double y_mean_{0.0}; ///< Empirical prior mean in original response units.
    double y_scale_{1.0}; ///< Population response SD or one for a tiny/constant response.
    std::vector<double> training_y_std_; ///< Centred and scaled responses z.
    std::vector<std::size_t> kept_rows_; ///< Zero-based original indices of retained training rows.
    std::vector<double> length_scale_grid_; ///< Fixed singleton or bounded geometric candidate grid.
    std::vector<double> length_scale_profile_ll_; ///< Best standardized log likelihood at each length scale.
    std::vector<double> noise_ratio_grid_; ///< Fixed singleton or bounded geometric noise-ratio grid.
    dstruct::DataFrame search_; ///< Owned records of every evaluated grid pair.
    double length_scale_used_{1.0}; ///< Selected or specified positive length scale.
    double noise_ratio_used_{0.0}; ///< Selected or specified nonnegative physical noise ratio.
    double signal_variance_used_{1.0}; ///< Profiled amplitude s in standardized-response variance units.
    double log_marginal_likelihood_used_{0.0}; ///< Selected standardized-response log density.
    std::optional<linalg::CholeskyDecomposition<double>> chol_; ///< Factor of K=s*(R+(eta+j)I), in standardized-response units.
    std::vector<double> alpha_; ///< K inverse times standardized response z.
    std::vector<double> fitted_; ///< Conditional deletion means on the original response scale.
    std::vector<double> posterior_fitted_; ///< Full-data posterior latent means at training inputs.
    std::vector<double> loo_sd_; ///< Conditional held-out observation SDs, excluding query jitter.
    std::size_t observations_{0}; ///< Number of complete retained training rows.
};

} // namespace datamunge::stats
