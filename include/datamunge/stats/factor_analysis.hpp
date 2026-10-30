#pragma once
/// @file factor_analysis.hpp
/// @brief Principal-axis factor analysis with explicit boundary, residual and iteration evidence.
#include <cstddef>
#include <cstdint>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <iosfwd>
#include <string>
#include <vector>
namespace datamunge::stats {
/// @brief Controls iterative principal-axis factoring (PAF), not maximum likelihood.
/// Both extraction and optional pairwise varimax use max_iter/tol. Extraction's
/// absolute tolerance is in fitted variance units; rotation uses its normalized objective.
/// @par Example
/// @code
/// const dm::stats::FactorAnalysisOptions o{1,true,"none",1000,1e-8};
/// std::print("{}\n",o.n_factors); // 1
/// @endcode
///
/// @par References
/// - W. Revelle, psych::fa (principal-axis iteration):
/// https://personality-project.org/r/psych/help/fa.html
/// @see FactorAnalysis
struct FactorAnalysisOptions {
  std::size_t n_factors{
      2}; ///< Extract k in 1..p; negative parameter-count df is allowed and reported.
  bool standardize{true}; ///< Centre and divide by sample SD (n-1); false fits covariance units.
  std::string rotation{"varimax"}; ///< "none" or Kaiser-normalized orthogonal "varimax".
  std::size_t max_iter{1000}; ///< Positive extraction iterations and, separately, rotation sweeps.
  double      tol{1e-6}; ///< Finite positive absolute maximum clipped-communality update tolerance.
};
/// @brief Exploratory common-factor covariance model fitted by iterative principal axes.
/// For centred rows Z, models C approximately as Lambda*Lambda^T+diag(Psi), with
/// unit-variance uncorrelated latent factors and diagonal variable-specific variance.
/// This is PAF, not R factanal's maximum-likelihood estimator or a minres optimizer.
/// Each reduced matrix replaces C's diagonal with clipped communalities; negative
/// retained eigenvalues become zero. SMC initialization and regression scores use an
/// eigenvalue-floored inverse, floor 1e-8*max(diag(C)); this is not a pseudoinverse.
/// Orthogonal varimax preserves the common covariance, not the identities of factors.
/// Factors are ordered by squared loading sum, signs by largest-magnitude loading.
/// Rows with any null selected feature are dropped; other retained values must be
/// finite, with positive sample variance in every selected column. Repeated names,
/// invalid options and unrepresentable arithmetic are rejected. Negative parameter-
/// count df is allowed for exploration, not certified identification. Boundary fits
/// can converge: check diagnostics(), raw_uniqueness and covariance residuals.
/// Existing uniquenesses() remain clipped nonnegative; reported communalities remain
/// actual squared loading norms and may exceed observed variance (Heywood violation).
/// Owns fitted state; inputs unchanged; const operations share no mutable caches.
/// For t extraction steps, bounded Jacobi sweeps and small k, main work O(np^2+t*p^3),
/// plus bounded rotation work; storage O(np+p^2+t*p) including the iteration history.
/// @par Example
/// @code
/// const auto d=dm::datasets::iris();
/// const dm::stats::FactorAnalysis m(d,
///     {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"}, {1,true,"none"});
/// std::print("{} {}\n",m.observations(),m.num_factors()); // 150 1
/// @endcode
///
/// @par References
/// - W. Revelle, psych::fa (principal-axis iteration):
/// https://personality-project.org/r/psych/help/fa.html
/// - H. F. Kaiser (1958), The varimax criterion for analytic rotation in factor analysis.
///   https://doi.org/10.1007/BF02289233
/// - R factanal (model/scores; different fitting method):
///   https://stat.ethz.ch/R-manual/R-devel/library/stats/html/factanal.html
/// @see loading_table(), variance_table(), diagnostics()
class FactorAnalysis {
 public:
  /// @brief Fits selected numeric columns immediately, retaining complete-row order.
  /// @param data Borrowed frame, unchanged; null selected features cause listwise deletion.
  /// @param feature_columns Nonempty distinct numeric names, in fitted matrix order.
  /// @param options PAF/rotation controls; defaults two factors, scaling and varimax.
  /// @throws std::invalid_argument Invalid schema/options, fewer than two complete rows,
  /// nonfinite retained data, zero variance, or unrepresentable preprocessing/fit/scores.
  /// @par Example
  /// See the class example.
  /// @par References
  /// - W. Revelle, psych::fa (principal-axis iteration):
  /// https://personality-project.org/r/psych/help/fa.html
  FactorAnalysis(
      const dstruct::DataFrame&       data,
      const std::vector<std::string>& feature_columns,
      FactorAnalysisOptions           options = {});
  /// @brief Selected feature schema in fitted matrix order.
  /// @return Borrowed names, valid for model lifetime.
  [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
  /// @brief Number of complete rows used in fitting.
  /// @return n>=2.
  [[nodiscard]] std::size_t observations() const { return observations_; }
  /// @brief Number of extracted columns, including any zero-loading directions.
  /// @return Requested k, not a numerical rank or identification certificate.
  [[nodiscard]] std::size_t num_factors() const { return loadings_.cols(); }
  /// @brief Maps score rows back to the original input.
  /// @return Borrowed zero-based indices in increasing fitted order.
  [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const {
    return kept_row_indices_;
  }
  /// @brief Fitted feature offsets.
  /// @return Borrowed p means, computed on retained rows.
  [[nodiscard]] const std::vector<double>& means() const { return mean_; }
  /// @brief Fitted feature divisors.
  /// @return Borrowed p sample SDs if standardized; otherwise ones.
  [[nodiscard]] const std::vector<double>& scales() const { return scale_; }
  /// @brief Observed sample covariance in fitted feature units.
  /// @return Borrowed p-by-p centred Gram matrix divided by n-1; correlation if scaled.
  [[nodiscard]] const linalg::DenseMatrix<double>& covariance_matrix() const { return covariance_; }
  /// @brief Post-rotation p-by-k loading matrix.
  /// @return Borrowed matrix in fitted units; signs and rotations do not identify causal factors.
  [[nodiscard]] const linalg::DenseMatrix<double>& loadings() const { return loadings_; }
  /// @brief Actual common variances h_j^2=sum_c Lambda_jc^2.
  /// @return Borrowed p-vector in fitted variance units, possibly exceeding observed variance.
  [[nodiscard]] const std::vector<double>& communalities() const { return communalities_; }
  /// @brief Compatibility nonnegative uniqueness, max(C_jj-h_j^2,0).
  /// @return Borrowed p-vector; clipping can prevent common+unique from matching C_jj.
  /// Use variance_table() for signed raw_uniqueness and Heywood flags.
  [[nodiscard]] const std::vector<double>& uniquenesses() const { return uniquenesses_; }
  /// @brief Squared loading sum for each ordered factor.
  /// @return Borrowed k-vector in fitted variance units; rotated sums are not eigenvalues.
  [[nodiscard]] const std::vector<double>& variance_explained() const {
    return variance_explained_;
  }
  /// @brief Regression scores Z*C_floor_inverse*Lambda, not observed latent variables.
  /// @return Borrowed n-by-k matrix; generally neither orthogonal nor unit sample variance.
  /// Uses observed C (with eigenvalue floor), not the reproduced covariance.
  [[nodiscard]] const linalg::DenseMatrix<double>& scores() const { return scores_; }
  /// @brief Number of extraction iterations completed.
  /// @return Count in 1..max_iter, excluding rotation sweeps.
  [[nodiscard]] std::size_t iterations() const { return iterations_; }
  /// @brief Whether the maximum clipped-communality change fell below tol.
  /// @return Extraction status only; does not establish admissibility, global optimality or
  /// rotation convergence.
  [[nodiscard]] bool converged() const { return converged_; }
  /// @brief Extraction evidence before rotation, one row per completed step.
  /// @return Borrowed frame: iteration, max_change, offdiagonal_rms, heywood_variables,
  /// communality_1 through communality_p (clipped update values in feature order).
  /// Loadings can exceed those clipped values; diagnostics() describes the final loadings.
  [[nodiscard]] const dstruct::DataFrame& iteration_history() const { return iteration_history_; }
  /// @brief Copies one factor's feature loadings.
  /// @param factor_index Zero-based index less than num_factors().
  /// @return Owned p-vector, preserving fitted feature order.
  /// @throws std::out_of_range Invalid factor index.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::FactorAnalysis m(d,
  ///     {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"}, {1,true,"none"});
  /// std::print("{}\n",m.factor_loadings(0).size()); // 4
  /// @endcode
  /// @see loadings(), scores()
  [[nodiscard]] std::vector<double> factor_loadings(std::size_t factor_index) const;
  /// @brief Copies one factor's retained-row regression scores.
  /// @param factor_index Zero-based index less than num_factors().
  /// @return Owned n-vector, preserving fitted row order.
  /// @throws std::out_of_range Invalid factor index.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::FactorAnalysis m(d,
  ///     {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"}, {1,true,"none"});
  /// std::print("{}\n",m.factor_scores(0).size()); // 150
  /// @endcode
  /// @see loadings(), scores()
  [[nodiscard]] std::vector<double> factor_scores(std::size_t factor_index) const;
  /// @brief Squared loading sums divided by the observed trace(C).
  /// @return Owned k shares; sum need not be one and can exceed one for an improper fit.
  /// Covariance mode uses the actual observed trace, not the sum of clipped uniquenesses.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::FactorAnalysis m(d,
  ///     {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"}, {1,true,"none"});
  /// std::print("{}\n",m.proportion_variance().size()); // 1
  /// @endcode
  ///
  /// @par References
  /// - W. Revelle, psych::fa (principal-axis iteration):
  /// https://personality-project.org/r/psych/help/fa.html
  /// @see diagnostics(), covariance_matrix()
  [[nodiscard]] std::vector<double> proportion_variance() const;
  /// @brief Forms the common covariance Lambda*Lambda^T.
  /// @return Owned p-by-p matrix in fitted units, invariant under orthogonal rotation.
  /// @throws std::invalid_argument Unrepresentable covariance.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::FactorAnalysis m(d,
  ///     {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"}, {1,true,"none"});
  /// std::print("{}\n",m.common_covariance().rows()); // 4
  /// @endcode
  ///
  /// @par References
  /// - W. Revelle, psych::fa (principal-axis iteration):
  /// https://personality-project.org/r/psych/help/fa.html
  /// @see diagnostics(), covariance_matrix()
  [[nodiscard]] linalg::DenseMatrix<double> common_covariance() const;
  /// @brief Adds the compatibility clipped uniqueness diagonal to common covariance.
  /// @return Owned fitted p-by-p covariance. A Heywood variable has an excessive diagonal;
  /// this method does not conceal that mismatch by substituting negative uniqueness.
  /// @throws std::invalid_argument Unrepresentable covariance.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::FactorAnalysis m(d,
  ///     {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"}, {1,true,"none"});
  /// std::print("{}\n",m.reproduced_covariance().rows()); // 4
  /// @endcode
  ///
  /// @par References
  /// - W. Revelle, psych::fa (principal-axis iteration):
  /// https://personality-project.org/r/psych/help/fa.html
  /// @see diagnostics(), covariance_matrix()
  [[nodiscard]] linalg::DenseMatrix<double> reproduced_covariance() const;
  /// @brief Presents post-rotation loadings by named feature.
  /// @return Owned p-row frame: feature, Factor1 through Factork.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::FactorAnalysis m(d,
  ///     {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"}, {1,true,"none"});
  /// std::print("{}\n",m.loading_table().nrows()); // 4
  /// @endcode
  ///
  /// @par References
  /// - W. Revelle, psych::fa (principal-axis iteration):
  /// https://personality-project.org/r/psych/help/fa.html
  /// @see diagnostics(), covariance_matrix()
  [[nodiscard]] dstruct::DataFrame loading_table() const;
  /// @brief Exposes the observed/common/unique split and unclipped boundary evidence.
  /// @return Owned p-row frame: feature, observed_variance, communality, raw_uniqueness
  /// (C_jj-h_j^2), uniqueness (clipped), heywood (0/1), common_fraction.
  /// Heywood means excess >64*epsilon*max(C_jj,h_j^2); fractions can exceed one.
  /// @throws std::invalid_argument Unrepresentable common fraction.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::FactorAnalysis m(d,
  ///     {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"}, {1,true,"none"});
  /// std::print("{}\n",m.variance_table().nrows()); // 4
  /// @endcode
  ///
  /// @par References
  /// - W. Revelle, psych::fa (principal-axis iteration):
  /// https://personality-project.org/r/psych/help/fa.html
  /// @see diagnostics(), covariance_matrix()
  [[nodiscard]] dstruct::DataFrame variance_table() const;
  /// @brief Lists every upper-triangle observed, common and reproduced covariance.
  /// @return Owned p*(p+1)/2 rows: feature_i, feature_j, observed, common, reproduced,
  /// residual (observed-reproduced). Diagonal rows expose clamped-uniqueness mismatch.
  /// @throws std::invalid_argument Unrepresentable covariance.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::FactorAnalysis m(d,
  ///     {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"}, {1,true,"none"});
  /// std::print("{}\n",m.covariance_pairs().nrows()); // 10
  /// @endcode
  ///
  /// @par References
  /// - W. Revelle, psych::fa (principal-axis iteration):
  /// https://personality-project.org/r/psych/help/fa.html
  /// @see diagnostics(), covariance_matrix()
  [[nodiscard]] dstruct::DataFrame covariance_pairs() const;
  /// @brief Separates numerical stopping, covariance fit and parameter-count evidence.
  /// @return One owned row: observations, features, factors, iterations, converged (0/1),
  /// degrees_freedom=((p-k)^2-p-k)/2, heywood_variables, offdiagonal_rms,
  /// max_diagonal_excess, total_variance, common_variance, kaiser_varimax.
  /// Nonnegative df is necessary by parameter counting, not sufficient for identification;
  /// no likelihood statistic or p-value is computed. Kaiser varimax is the normalized
  /// sum over factors of sum_j(q_jc^4)-(sum_j(q_jc^2))^2/p, q normalized by row norm.
  /// @throws std::invalid_argument Unrepresentable diagnostic arithmetic.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::FactorAnalysis m(d,
  ///     {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"}, {1,true,"none"});
  /// std::print("{:.0f}\n",m.diagnostics().double_at("degrees_freedom",0)); // 2
  /// @endcode
  ///
  /// @par References
  /// - W. Revelle, psych::fa (principal-axis iteration):
  /// https://personality-project.org/r/psych/help/fa.html
  /// @see diagnostics(), covariance_matrix()
  [[nodiscard]] dstruct::DataFrame diagnostics() const;
  /// @brief Builds a plain-text loading/variance and extraction-status summary.
  /// @return Owned string, including the material Heywood-variable count.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::FactorAnalysis m(d,
  ///     {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"}, {1,true,"none"});
  /// std::print("{}\n",!m.summary().empty()); // true
  /// @endcode
  ///
  /// @par References
  /// - W. Revelle, psych::fa (principal-axis iteration):
  /// https://personality-project.org/r/psych/help/fa.html
  /// @see diagnostics(), covariance_matrix()
  [[nodiscard]] std::string summary() const;
  /// @brief Writes summary() to a supplied stream without changing the fit.
  /// @param os Destination stream; its formatting flags/precision are modified.
  /// @par Example
  /// See summary().
  /// @see summary()
  void print_summary(std::ostream& os) const;
  /// @brief Writes the fitted summary to standard output.
  /// @par Example
  /// See summary().
  /// @see print_summary(std::ostream&)
  void print_summary() const;
  /// @brief Plots fitted regression scores with external categorical colours.
  /// Labels describe fitted observations; they do not participate in fitting.
  /// @param group_labels One string per score row, aligned through kept_row_indices().
  /// @param factor_x Zero-based horizontal factor, default zero.
  /// @param factor_y Zero-based vertical factor, default one; may equal factor_x.
  /// @return Owned plot with lexical group order and recycled colours.
  /// @throws std::out_of_range Either factor index is invalid.
  /// @throws std::invalid_argument Labels do not match observations().
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::FactorAnalysis m(d,
  ///     {"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"}, {1,true,"none"});
  /// const auto p=m.plot_scores(std::vector<std::string>(150,"all"),0,0);
  /// std::print("{}\n",p.series().size()); // 1
  /// @endcode
  /// @see factor_scores()
  [[nodiscard]] graphics::RPlot plot_scores(
      const std::vector<std::string>& group_labels,
      std::size_t                     factor_x = 0,
      std::size_t                     factor_y = 1) const;
  /// @brief Plots feature loading coordinates with labels and zero reference axes.
  /// A unit-circle reference is drawn only for standardized fits. Coordinates represent
  /// only the displayed factor pair; a point need not be on the circle even when total
  /// communality across more than two factors is one. Set equal physical axis scales if
  /// interpreting angles; covariance loadings are in heterogeneous feature units.
  /// @param factor_x Zero-based horizontal factor, default zero.
  /// @param factor_y Distinct zero-based vertical factor, default one.
  /// @return Owned annotated plot; caller may adjust limits/title/size.
  /// @throws std::out_of_range Either factor index is invalid.
  /// @throws std::invalid_argument Factor indices coincide.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::FactorAnalysis
  /// m(d,{"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"},{2});
  /// std::print("{}\n",!m.plot_loadings().series().empty()); // true
  /// @endcode
  /// @see loading_table()
  [[nodiscard]] graphics::RPlot plot_loadings(std::size_t factor_x = 0, std::size_t factor_y = 1)
      const;

 private:
  /// @brief Validates, preprocesses, extracts, rotates and scores the constructor data.
  /// @param data Borrowed frame, unchanged.
  /// @throws std::invalid_argument Invalid data/options or unrepresentable arithmetic.
  void                     fit(const dstruct::DataFrame& data);
  std::vector<std::string> feature_columns_;  ///< Fitted ordered numeric schema.
  FactorAnalysisOptions    options_;          ///< Validated extraction and rotation controls.
  std::vector<std::size_t> kept_row_indices_; ///< Original complete-row indices.
  std::vector<double>      mean_;             ///< Fitted centring offsets.
  std::vector<double>      scale_;            ///< Positive fitted feature divisors.
  std::vector<double> communalities_; ///< Actual row squared loading norms, not clipped updates.
  std::vector<double> uniquenesses_; ///< Compatibility nonnegative observed-minus-common variances.
  std::vector<double> variance_explained_; ///< Descending post-rotation squared column norms.
  linalg::DenseMatrix<double> covariance_; ///< Observed covariance in fitted feature units.
  linalg::DenseMatrix<double> loadings_;   ///< Final sign/order-normalized p-by-k loadings.
  linalg::DenseMatrix<double> scores_;     ///< Retained-row regression scores.
  dstruct::DataFrame
              iteration_history_; ///< Per-step clipped communalities and raw-loading evidence.
  double      total_variance_{0}; ///< Actual observed trace, including when uniqueness clips.
  std::size_t observations_{0};   ///< Retained training row count.
  std::size_t iterations_{0};     ///< Completed PAF updates, excluding rotation.
  bool        converged_{false};  ///< Clipped update criterion satisfied.
};
/// @brief Samples a zero-mean Gaussian common-factor model with independent standard factors.
/// Rows are x=Lambda*f+sqrt(Psi)*epsilon; all factor/noise draws are independent.
/// Input loadings/unique variances are in original feature units; no sample standardization.
/// mt19937_64 and std::normal_distribution give reproducibility within a toolchain, not
/// bit-identical normal draws across standard libraries. Work O(npk), storage O(np+k).
/// Inputs borrowed unchanged; no shared generator state.
/// @param loadings Finite nonempty p-by-k loadings, p,k>=1.
/// @param uniqueness p finite nonnegative variances, not standard deviations.
/// @param observations Number of rows, at least two, default 500.
/// @param seed Local generator seed, default 123.
/// @return Owned numeric DataFrame columns v1 through vp, in loading-row order.
/// @throws std::invalid_argument Bad dimensions/values/count or unrepresentable sample.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> L(3,1,{.7,.7,.7});
/// const auto d=dm::stats::factor_model_sample(L,{.51,.51,.51},20,123);
/// std::print("{} {}\n",d.nrows(),d.ncols()); // 20 3
/// @endcode
///
/// @par References
/// - W. Revelle, psych::fa (principal-axis iteration):
/// https://personality-project.org/r/psych/help/fa.html
/// @see FactorAnalysis
[[nodiscard]] dstruct::DataFrame factor_model_sample(
    const linalg::DenseMatrix<double>& loadings,
    const std::vector<double>&         uniqueness,
    std::size_t                        observations = 500,
    std::uint64_t                      seed         = 123);
/// @brief Root mean squared covariance residual over ordered off-diagonal entries.
/// @param observed Finite square p-by-p observed matrix, including p=0 or p=1.
/// @param reproduced Same-shaped finite matrix in the same units; no symmetry assumption.
/// @return sqrt(sum_(i!=j)(observed_ij-reproduced_ij)^2/(p*(p-1))), or zero for p<2.
/// Owned scalar; inputs unchanged; work O(p^2), auxiliary storage O(1), no shared state.
/// @throws std::invalid_argument Bad shape/nonfinite entry or unrepresentable result.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> a(2,2,{1,.5,.5,1}),b(2,2,{1,.25,.25,1});
/// std::print("{:.2f}\n",dm::stats::covariance_residual_rms(a,b)); // 0.25
/// @endcode
/// @see FactorAnalysis::covariance_pairs()
[[nodiscard]] double covariance_residual_rms(
    const linalg::DenseMatrix<double>& observed,
    const linalg::DenseMatrix<double>& reproduced);
} // namespace datamunge::stats
