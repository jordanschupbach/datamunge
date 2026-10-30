#pragma once
/// @file gbm_classifier.hpp
/// @brief Plain-gradient multinomial tree boosting, prefix predictions and validation diagnostics.
#include <cstddef>
#include <cstdint>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/decision_tree_regressor.hpp>
#include <datamunge/stats/formula.hpp>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::stats {
/// @brief Controls the number, shrinkage, shape and row sampling of residual trees.
/// max_depth must be positive; unsplittable nodes remain constant. Rates above one
/// are allowed, but no monotone
/// training-loss guarantee is asserted for arbitrary rates or stochastic samples.
/// @par Example
/// @code
/// const dm::stats::GBMClassifierOptions o{20,.1,2};
/// std::print("{}\n",o.n_trees); // 20
/// @endcode
///
/// @par References
/// - J. H. Friedman (2001), Greedy function approximation: A gradient boosting machine.
///   https://doi.org/10.1214/aos/1013203451
/// @see GBMClassifier
struct GBMClassifierOptions {
  std::size_t n_trees{100}; ///< Positive boosting-round count; each round stores K class trees.
  double      learning_rate{.1};    ///< Finite positive multiplier on every class-tree output.
  std::size_t max_depth{3};         ///< Root depth zero; positive maximum tree depth.
  std::size_t min_samples_split{2}; ///< Minimum node size eligible to split; at least two.
  std::size_t min_samples_leaf{1};  ///< Positive minimum leaf size accepted by the CART splitter.
  double      subsample{
      1}; ///< Finite fraction in (0,1]; round(fraction*n), at least one, without replacement.
  std::uint64_t seed{
      42}; ///< Local SplitMix64 row-sampling seed; shared sampled rows across K class trees.
};
/// @brief Owned labels and softmax probabilities in original query-row order.
/// Missing predictors produce an empty string and empty probability row. A complete
/// row has K probabilities in GBMClassifier::classes() order; ties use the first sorted
/// class. Probabilities can underflow to zero and are not post-hoc calibrated.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;
/// d.add_column("x",std::vector<double>{0,1,2,3});
/// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
/// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
/// std::print("{}\n",m.predict_detail(d).probability[0].size()); // 2
/// @endcode
/// @see GBMClassifier::predict_detail()
struct GBMClassifierPrediction {
  std::vector<std::string>
      class_label; ///< Predicted labels, or empty strings for omitted query rows.
  std::vector<std::vector<double>>
      probability; ///< Row-by-class probability vectors, empty for omitted rows.
};
/// @brief Multiclass additive trees fitted to one-hot-minus-softmax residuals.
/// Starts from log empirical class priors. At each round, all K residual targets use the
/// same pre-round probabilities, then one least-squares CART tree per class is fitted.
/// A scalar learning_rate multiplies its leaf means. There is no Newton leaf correction,
/// (K-1)/K multiplier, centring constraint, line search or built-in early stopping; this
/// is not numerically identical to sklearn GradientBoostingClassifier/LogitBoost.
/// Formula supplies finite encoded predictors (including categorical/interaction terms)
/// and a string response. Complete cases define sorted classes and priors; null formula
/// rows are dropped. Needs at least one expanded predictor and two retained classes.
/// All prior trees remain fixed. Subsampling selects rows without replacement once per
/// round, shared across its K trees. The seed does not affect full-sample deterministic
/// CART fits; greedy split ties follow predictor/threshold order and absolute tolerances.
/// training_deviance is mean multinomial negative log likelihood, without factor two,
/// computed with log-sum-exp rather than flooring probabilities. Prefix zero is the prior
/// model. Evaluation is explicit, never used to fit or alter trees. Training accuracy or
/// loss cannot diagnose generalization or select a stopping round without held-out data.
/// Model owns the Formula schema, trees, encoded training data and row maps. Inputs are
/// unchanged; const operations share no mutable caches. This CART implementation scans
/// all candidate thresholds: conservative fit work O(M*K*p*n^2*(D+1)), with storage
/// O(np+M*K*(n+nodes_per_tree)) including each tree's stored training predictions.
/// Numeric inference traverses fitted trees; no query preprocessing is learned.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;
/// d.add_column("x",std::vector<double>{0,1,2,3});
/// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
/// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
/// std::print("{} {:.0f}\n",m.n_trees(),m.training_accuracy()); // 2 1
/// @endcode
///
/// @par References
/// - J. H. Friedman (2001), Greedy function approximation: A gradient boosting machine.
///   https://doi.org/10.1214/aos/1013203451
/// @see training_step(), staged_evaluation(), gbm_validation_search()
class GBMClassifier {
 public:
  /// @brief Fits the requested positive number of boosting rounds immediately.
  /// @param data Borrowed formula variables, unchanged; null formula rows are dropped.
  /// @param formula Categorical response formula, e.g. "Species ~ Petal.Length + Petal.Width".
  /// @param options Shrinkage, CART and sampling controls; see GBMClassifierOptions.
  /// @throws std::invalid_argument Invalid controls/schema, fewer than two retained classes,
  /// no expanded predictors, retained nonfinite entries or unrepresentable scores/loss.
  /// @throws std::runtime_error No complete formula observations.
  /// @throws std::domain_error Invalid Formula transform domain.
  /// @par Example
  /// See the class example.
  /// @see GBMClassifierOptions
  GBMClassifier(
      const dstruct::DataFrame& data,
      const std::string&        formula,
      GBMClassifierOptions      options = {});
  /// @brief Delegates fitting with R-style formula-first argument order.
  /// @param formula Formula object, accepting implicit string conversion.
  /// @param data Borrowed frame supplying model columns.
  /// @param options Same controls as the primary constructor.
  /// @throws std::invalid_argument Same as the primary constructor.
  /// @throws std::runtime_error No complete formula rows.
  /// @throws std::domain_error Invalid Formula transform domain.
  /// @par Example
  /// See the class example with arguments (dm::formula("label ~ x"),d,{2,.5,1}).
  /// @see GBMClassifier
  GBMClassifier(
      const Formula&            formula,
      const dstruct::DataFrame& data,
      GBMClassifierOptions      options = {})
      : GBMClassifier(data, formula.text(), std::move(options)) {}
  /// @brief Original parsed formula text.
  /// @return Borrowed text, valid for model lifetime.
  [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
  /// @brief Frozen lexical class order used by every score/probability column.
  /// @return Borrowed K labels; ties prefer the first.
  [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
  /// @brief Expanded encoded predictor order, excluding any intercept.
  /// @return Borrowed p design names.
  [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
  /// @brief Complete training-row count.
  /// @return Positive n after joint formula null dropping.
  [[nodiscard]] std::size_t observations() const { return observations_; }
  /// @brief Number of fitted boosting rounds, not total individual trees.
  /// @return M; total trees are M times classes().size().
  [[nodiscard]] std::size_t n_trees() const { return trees_.size(); }
  /// @brief Maps fitted rows back to original constructor rows.
  /// @return Borrowed zero-based original indices in retained order.
  [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const {
    return kept_row_indices_;
  }
  /// @brief Initial class scores before any trees.
  /// @return Borrowed K log empirical class priors in classes() order.
  [[nodiscard]] const std::vector<double>& initial_scores() const { return initial_scores_; }
  /// @brief Full-model training predictions in complete-case order.
  /// @return Borrowed n labels, aligned through kept_row_indices().
  [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
  /// @brief Mean unfloored negative log-softmax after each completed round.
  /// @return Borrowed M training losses, excluding the prior round; evaluate it with prefix zero.
  [[nodiscard]] const std::vector<double>& training_deviance() const { return training_deviance_; }
  /// @brief Averages each class tree's normalized SSE split-gain vector equally.
  /// @return Owned p values: mean of M*K per-tree normalized importances. A tree without
  /// splits contributes zeros, so the sum lies in [0,1], not necessarily one. This is not
  /// pooled raw gain, permutation importance, causal influence or a significance test.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{}\n",m.feature_importance().size()); // 1
  /// @endcode
  ///
  /// @par References
  /// - J. H. Friedman (2001), Greedy function approximation: A gradient boosting machine.
  ///   https://doi.org/10.1214/aos/1013203451
  /// @see staged_evaluation()
  [[nodiscard]] std::vector<double> feature_importance() const;
  /// @brief Names the legacy mean importance and its display-normalized share.
  /// @return Owned p rows: predictor, mean_normalized_gain, relative_share. The last column
  /// sums to one when any split contributes, or remains zero for an all-constant ensemble.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{}\n",m.importance_table().nrows()); // 1
  /// @endcode
  ///
  /// @par References
  /// - J. H. Friedman (2001), Greedy function approximation: A gradient boosting machine.
  ///   https://doi.org/10.1214/aos/1013203451
  /// @see staged_evaluation()
  [[nodiscard]] dstruct::DataFrame importance_table() const;
  /// @brief Fraction of retained training labels predicted correctly by the full ensemble.
  /// @return In-sample accuracy in [0,1], not held-out performance.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{:.0f}\n",m.training_accuracy()); // 1
  /// @endcode
  ///
  /// @par References
  /// - J. H. Friedman (2001), Greedy function approximation: A gradient boosting machine.
  ///   https://doi.org/10.1214/aos/1013203451
  /// @see staged_evaluation()
  [[nodiscard]] double training_accuracy() const;
  /// @brief Full-model training actual-versus-predicted label counts.
  /// @return Owned K-by-K counts; rows actual, columns predicted, both in classes() order.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{}\n",m.confusion_matrix().rows()); // 2
  /// @endcode
  ///
  /// @par References
  /// - J. H. Friedman (2001), Greedy function approximation: A gradient boosting machine.
  ///   https://doi.org/10.1214/aos/1013203451
  /// @see staged_evaluation()
  [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;
  /// @brief Formats controls, per-tree-normalized importance and training metrics.
  /// @return Owned plain text; the displayed deviance is mean negative log likelihood.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{}\n",!m.summary().empty()); // true
  /// @endcode
  ///
  /// @par References
  /// - J. H. Friedman (2001), Greedy function approximation: A gradient boosting machine.
  ///   https://doi.org/10.1214/aos/1013203451
  /// @see staged_evaluation()
  [[nodiscard]] std::string summary() const;
  /// @brief Writes summary() to a stream.
  /// @param os Destination; model is unchanged.
  /// @par Example
  /// See summary().
  /// @see summary()
  void print_summary(std::ostream& os) const;
  /// @brief Writes summary() to standard output.
  /// @par Example
  /// See summary().
  /// @see print_summary(std::ostream&)
  void print_summary() const;
  /// @brief Predicts final labels with the full fitted ensemble.
  /// @param newdata Borrowed predictors; response not needed and extra columns ignored.
  /// @return Owned labels in original row order; missing predictors give empty labels.
  /// Empty input returns empty before schema checks; a nonempty all-incomplete query throws.
  /// @throws std::invalid_argument Invalid schema/response class, retained nonfinite
  /// predictors, or unrepresentable score/metric arithmetic.
  /// @throws std::runtime_error No complete nonempty rows or an unseen categorical predictor level.
  /// @throws std::out_of_range Missing predictor columns or an invalid round prefix.
  /// @throws std::domain_error A Formula transform is outside its domain.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{}\n",m.predict(d)[0]); // a
  /// @endcode
  /// @see predict_detail()
  [[nodiscard]] std::vector<std::string> predict(const dstruct::DataFrame& newdata) const;
  /// @brief Predicts full-model class probabilities and labels using the frozen schema.
  /// @param newdata Borrowed predictors, unchanged.
  /// @return Owned query-row labels and probabilities; see GBMClassifierPrediction.
  /// Empty input returns empty; a nonempty all-incomplete query throws.
  /// @throws std::invalid_argument Invalid schema/response class, retained nonfinite
  /// predictors, or unrepresentable score/metric arithmetic.
  /// @throws std::runtime_error No complete nonempty rows or an unseen categorical predictor level.
  /// @throws std::out_of_range Missing predictor columns or an invalid round prefix.
  /// @throws std::domain_error A Formula transform is outside its domain.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{}\n",m.predict_detail(d).class_label.size()); // 4
  /// @endcode
  /// @see predict_detail(const dstruct::DataFrame&,std::size_t)
  [[nodiscard]] GBMClassifierPrediction predict_detail(const dstruct::DataFrame& newdata) const;
  /// @brief Predicts with a fixed fitted prefix, including round-zero priors.
  /// No fitting or model mutation occurs; ties use lexical class order. A query row gives
  /// the same result alone or batched. Work O(n*rounds*K*(D+1)), auxiliary storage O(n*K).
  /// @param newdata Borrowed predictor schema; null predictors leave empty output rows.
  /// @param rounds Completed prefix length in 0..n_trees(); zero means prior probabilities.
  /// @return Owned row-aligned probabilities and labels, as for the full overload.
  /// @throws std::invalid_argument Invalid schema/response class, retained nonfinite
  /// predictors, or unrepresentable score/metric arithmetic.
  /// @throws std::runtime_error No complete nonempty rows or an unseen categorical predictor level.
  /// @throws std::out_of_range Missing predictor columns or an invalid round prefix.
  /// @throws std::domain_error A Formula transform is outside its domain.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{:.1f}\n",m.predict_detail(d,0).probability[0][0]); // 0.5
  /// @endcode
  /// @see raw_scores()
  [[nodiscard]] GBMClassifierPrediction predict_detail(
      const dstruct::DataFrame& newdata,
      std::size_t               rounds) const;
  /// @brief Returns full-model raw class scores before softmax.
  /// @param newdata Borrowed predictors.
  /// @return Owned row-by-K scores; see the prefix overload for missing-row semantics.
  /// @throws std::invalid_argument Invalid schema/response class, retained nonfinite
  /// predictors, or unrepresentable score/metric arithmetic.
  /// @throws std::runtime_error No complete nonempty rows or an unseen categorical predictor level.
  /// @throws std::out_of_range Missing predictor columns or an invalid round prefix.
  /// @throws std::domain_error A Formula transform is outside its domain.
  /// @par Example
  /// See the prefix overload.
  /// @see raw_scores(const dstruct::DataFrame&,std::size_t)
  [[nodiscard]] linalg::DenseMatrix<double> raw_scores(const dstruct::DataFrame& newdata) const {
    return raw_scores(newdata, n_trees());
  }
  /// @brief Returns log-prior plus the requested weighted class-tree prefix.
  /// Absolute scores are not probabilities; adding a common row offset leaves softmax unchanged.
  /// @param newdata Borrowed predictors; schema is frozen from fitting.
  /// @param rounds Prefix length in 0..n_trees(), including zero for log priors alone.
  /// @return Owned original-query-row by K matrix; all-NaN rows for null predictors.
  /// Empty input returns 0-by-K before schema validation; all-incomplete nonempty input throws.
  /// @throws std::invalid_argument Invalid schema/response class, retained nonfinite
  /// predictors, or unrepresentable score/metric arithmetic.
  /// @throws std::runtime_error No complete nonempty rows or an unseen categorical predictor level.
  /// @throws std::out_of_range Missing predictor columns or an invalid round prefix.
  /// @throws std::domain_error A Formula transform is outside its domain.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{:.6f}\n",m.raw_scores(d,0)(0,0)); // -0.693147
  /// @endcode
  /// @see predict_detail()
  [[nodiscard]] linalg::DenseMatrix<double> raw_scores(
      const dstruct::DataFrame& newdata,
      std::size_t               rounds) const;
  /// @brief Exposes one training round's simultaneous residual targets and fitted updates.
  /// All class residuals use the same pre-round probabilities, including rows omitted by
  /// stochastic sampling. in_bag identifies which original rows actually fitted those trees.
  /// Work O(round*n*K*(D+1)); output O(n*K), with no refitting or shared mutable caches.
  /// @param round One-based completed round in 1..n_trees().
  /// @return Owned n*K row-major/class-minor rows: round, original_row, class, one_hot,
  /// before_score, before_probability, pseudo_residual, tree_value, after_score,
  /// after_probability, in_bag (0/1). Scores include the learning-rate multiplier;
  /// tree_value itself is unshrunk. Original row indices refer to the constructor frame.
  /// @throws std::out_of_range Invalid training round.
  /// @throws std::invalid_argument Unrepresentable accumulated score.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{}\n",m.training_step(1).nrows()); // 8
  /// @endcode
  ///
  /// @par References
  /// - J. H. Friedman (2001), Greedy function approximation: A gradient boosting machine.
  ///   https://doi.org/10.1214/aos/1013203451
  /// @see round_sample_rows(), raw_scores()
  [[nodiscard]] dstruct::DataFrame training_step(std::size_t round) const;
  /// @brief Aggregates one step's full-training projection and loss-change evidence.
  /// @param round One-based completed round in 1..n_trees().
  /// @return Owned row: round, sample_size, full_sample (0/1), loss_before, loss_after,
  /// gradient_tree_inner, tree_squared_norm, quadratic_bound. Inner/norm are sums over
  /// every retained row and class. The bound is loss_before-eta*(1-eta/4)*norm/n;
  /// its descent interpretation requires a full-sample leaf-mean projection and 0<eta<4.
  /// Stochastic fits need not satisfy that projection identity or bound. No refitting;
  /// work/storage follow training_step().
  /// @throws std::out_of_range Invalid round.
  /// @throws std::invalid_argument Unrepresentable diagnostics.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{:.0f}\n",m.step_diagnostics(1).double_at("sample_size",0)); // 4
  /// @endcode
  /// @see training_step(), training_deviance()
  [[nodiscard]] dstruct::DataFrame step_diagnostics(std::size_t round) const;
  /// @brief Returns the actual original training rows sampled for a completed round.
  /// @param round One-based round in 1..n_trees().
  /// @return Owned sorted original row indices; complete training rows if subsample=1.
  /// @throws std::out_of_range Invalid round.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{}\n",m.round_sample_rows(1).size()); // 4
  /// @endcode
  ///
  /// @par References
  /// - J. H. Friedman (2001), Greedy function approximation: A gradient boosting machine.
  ///   https://doi.org/10.1214/aos/1013203451
  /// @see kept_row_indices(), training_step()
  [[nodiscard]] std::vector<std::size_t> round_sample_rows(std::size_t round) const;
  /// @brief Evaluates every fixed-model prefix on identical complete labelled rows.
  /// Includes the prior at round zero; each fitted tree is traversed once per evaluation
  /// row. No refitting, early stopping, calibration or model mutation occurs. Mean log loss
  /// uses stable log-sum-exp, without probability clipping. Brier is mean(sum_k(p_k-y_k)^2),
  /// without dividing by K. Accuracy is a fraction; training and evaluation rows are distinct
  /// only if the caller supplies distinct data. Work O(n*M*K*(D+1)), storage O(n*K+M).
  /// @param data Borrowed labelled data with the frozen response/predictor schema.
  /// @return Owned M+1 rows: round, input_rows, evaluated_rows, skipped_rows, accuracy,
  /// mean_log_loss, brier. Null formula rows are omitted jointly, at every stage.
  /// @throws std::invalid_argument Invalid schema/response class, retained nonfinite
  /// predictors, or unrepresentable score/metric arithmetic.
  /// @throws std::runtime_error No complete nonempty rows or an unseen categorical predictor level.
  /// @throws std::out_of_range Missing predictor columns or an invalid round prefix.
  /// @throws std::domain_error A Formula transform is outside its domain.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{}\n",m.staged_evaluation(d).nrows()); // 3
  /// @endcode
  ///
  /// @par References
  /// - J. H. Friedman (2001), Greedy function approximation: A gradient boosting machine.
  ///   https://doi.org/10.1214/aos/1013203451
  /// @see evaluation(), gbm_validation_search()
  [[nodiscard]] dstruct::DataFrame staged_evaluation(const dstruct::DataFrame& data) const;
  /// @brief Returns full-model held-out classification metrics.
  /// @param data Borrowed labelled evaluation frame, unchanged.
  /// @return One owned row with round/counts, accuracy, stable mean_log_loss and Brier.
  /// @throws std::invalid_argument Invalid schema/response class, retained nonfinite
  /// predictors, or unrepresentable score/metric arithmetic.
  /// @throws std::runtime_error No complete nonempty rows or an unseen categorical predictor level.
  /// @throws std::out_of_range Missing predictor columns or an invalid round prefix.
  /// @throws std::domain_error A Formula transform is outside its domain.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{:.0f}\n",m.evaluation(d).double_at("evaluated_rows",0)); // 4
  /// @endcode
  /// @see staged_evaluation()
  [[nodiscard]] dstruct::DataFrame evaluation(const dstruct::DataFrame& data) const;
  /// @brief Returns evaluation for a fixed fitted prefix, including zero.
  /// @param data Borrowed labelled data with staged_evaluation() validity rules.
  /// @param rounds Prefix length in 0..n_trees().
  /// @return One owned row with round/counts, accuracy, stable mean_log_loss and Brier.
  /// @throws std::invalid_argument Invalid schema/response class, retained nonfinite
  /// predictors, or unrepresentable score/metric arithmetic.
  /// @throws std::runtime_error No complete nonempty rows or an unseen categorical predictor level.
  /// @throws std::out_of_range Missing predictor columns or an invalid round prefix.
  /// @throws std::domain_error A Formula transform is outside its domain.
  /// @par Example
  /// See the full-model overload, passing an explicit prefix.
  /// @see staged_evaluation()
  [[nodiscard]] dstruct::DataFrame evaluation(const dstruct::DataFrame& data, std::size_t rounds)
      const;
  /// @brief Returns full-model actual-versus-predicted label counts.
  /// @param data Borrowed labelled evaluation frame, unchanged.
  /// @return Owned K rows: actual class and pred_<class> count columns, in classes() order.
  /// @throws std::invalid_argument Invalid schema/response class, retained nonfinite
  /// predictors, or unrepresentable score/metric arithmetic.
  /// @throws std::runtime_error No complete nonempty rows or an unseen categorical predictor level.
  /// @throws std::out_of_range Missing predictor columns or an invalid round prefix.
  /// @throws std::domain_error A Formula transform is outside its domain.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{}\n",m.confusion(d).nrows()); // 2
  /// @endcode
  /// @see staged_evaluation()
  [[nodiscard]] dstruct::DataFrame confusion(const dstruct::DataFrame& data) const;
  /// @brief Returns confusion for a fixed fitted prefix, including zero.
  /// @param data Borrowed labelled data with staged_evaluation() validity rules.
  /// @param rounds Prefix length in 0..n_trees().
  /// @return Owned K rows: actual class and pred_<class> count columns, in classes() order.
  /// @throws std::invalid_argument Invalid schema/response class, retained nonfinite
  /// predictors, or unrepresentable score/metric arithmetic.
  /// @throws std::runtime_error No complete nonempty rows or an unseen categorical predictor level.
  /// @throws std::out_of_range Missing predictor columns or an invalid round prefix.
  /// @throws std::domain_error A Formula transform is outside its domain.
  /// @par Example
  /// See the full-model overload, passing an explicit prefix.
  /// @see staged_evaluation()
  [[nodiscard]] dstruct::DataFrame confusion(const dstruct::DataFrame& data, std::size_t rounds)
      const;
  /// @brief Plots observed classes and highlights full-model misclassifications.
  /// @param data Borrowed labelled plotting frame; predictions use the frozen model.
  /// @param x_feature Existing numeric plotting column.
  /// @param y_feature Existing numeric plotting column.
  /// @return Owned scatterplot; rows missing coordinates/labels are omitted. This can be
  /// training or held-out data; it does not perform splitting or estimate uncertainty.
  /// @throws std::invalid_argument Invalid schema/response class, retained nonfinite
  /// predictors, or unrepresentable score/metric arithmetic.
  /// @throws std::runtime_error No complete nonempty rows or an unseen categorical predictor level.
  /// @throws std::out_of_range Missing predictor columns or an invalid round prefix.
  /// @throws std::domain_error A Formula transform is outside its domain.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::GBMClassifier m(d,"Species ~ Petal.Length + Petal.Width",{2});
  /// const auto plot=m.plot_classification(d,"Petal.Length","Petal.Width");
  /// std::print("{}\n",!plot.series().empty()); // true
  /// @endcode
  /// @see plot_decision_regions()
  [[nodiscard]] graphics::RPlot plot_classification(
      const dstruct::DataFrame& data,
      const std::string&        x_feature,
      const std::string&        y_feature) const;
  /// @brief Plots full-model labels on a padded regular two-predictor grid.
  /// @param x_feature First distinct expanded numeric predictor name.
  /// @param y_feature Second distinct expanded numeric predictor name.
  /// @param grid_resolution At least two points per axis, default 60.
  /// @return Owned pastel grid-point plot, not observed data or a probability heatmap.
  /// @throws std::invalid_argument Not exactly two matching distinct predictor names or grid<2.
  /// @throws std::length_error Squared grid size overflows size_t.
  /// @par Example
  /// See the prefix overload.
  /// @see plot_classification()
  [[nodiscard]] graphics::RPlot plot_decision_regions(
      const std::string& x_feature,
      const std::string& y_feature,
      std::size_t        grid_resolution = 60) const;
  /// @brief Plots a fixed prefix's decision regions with no new fitted model.
  /// @param x_feature First distinct expanded numeric predictor name.
  /// @param y_feature Second distinct expanded numeric predictor name.
  /// @param grid_resolution Number of grid points per axis, at least two.
  /// @param rounds Prefix length in 0..n_trees(); zero shows the prior classifier.
  /// @return Owned pastel grid-point plot; interpolation is not inferred between observations.
  /// @throws std::invalid_argument Invalid axes/grid/prefix or nonfinite grid arithmetic.
  /// @throws std::length_error Squared grid size overflows size_t.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::GBMClassifier m(d,"Species ~ Petal.Length + Petal.Width",{2});
  /// const auto p=m.plot_decision_regions("Petal.Length","Petal.Width",10,1);
  /// std::print("{}\n",p.series().size()); // 3
  /// @endcode
  /// @see predict_detail()
  [[nodiscard]] graphics::RPlot plot_decision_regions(
      const std::string& x_feature,
      const std::string& y_feature,
      std::size_t        grid_resolution,
      std::size_t        rounds) const;
  /// @brief Plots training mean negative log likelihood after rounds one through M.
  /// @return Owned loss plot, excluding prior round zero; no held-out validity follows.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{0,1,2,3});
  /// d.add_column("label",std::vector<std::string>{"a","a","b","b"});
  /// const dm::stats::GBMClassifier m(d,"label ~ x",{2,.5,1});
  /// std::print("{}\n",m.plot_training_deviance().series().size()); // 1
  /// @endcode
  /// @see staged_evaluation()
  [[nodiscard]] graphics::RPlot plot_training_deviance() const;

 private:
  /// @brief Encodes complete rows, establishes priors and fits all requested class trees.
  /// @param data Borrowed constructor frame.
  /// @throws std::invalid_argument Invalid options/classes/design or unrepresentable arithmetic.
  /// @throws std::runtime_error No complete rows.
  /// @throws std::domain_error Invalid Formula transform.
  void fit(const dstruct::DataFrame& data);
  /// @brief Accumulates a valid prefix on an already encoded x0..x(p-1) frame.
  /// @param predictor_frame Borrowed finite synthetic predictor columns.
  /// @param n_rows Matching row count.
  /// @param rounds Prefix length, prevalidated in 0..n_trees().
  /// @return Owned n_rows-by-K raw scores.
  /// @throws std::invalid_argument Unrepresentable accumulated score.
  linalg::DenseMatrix<double> score_frame(
      const dstruct::DataFrame& predictor_frame,
      std::size_t               n_rows,
      std::size_t               rounds) const;
  /// @brief Records stages on already encoded complete labelled rows.
  /// @param predictors Synthetic x0..x(p-1) frame.
  /// @param labels Complete response labels, validated against classes().
  /// @param input_rows Original frame row count for skipped-row reporting.
  /// @param rounds Valid prefix endpoint.
  /// @return Owned prefix evaluation table including zero.
  /// @throws std::invalid_argument Unknown label or unrepresentable score/metric.
  dstruct::DataFrame evaluate_frame(
      const dstruct::DataFrame&       predictors,
      const std::vector<std::string>& labels,
      std::size_t                     input_rows,
      std::size_t                     rounds) const;
  Formula                  formula_;         ///< Parsed original formula.
  DesignInfo               design_;          ///< Frozen encoded predictor and response schema.
  GBMClassifierOptions     options_;         ///< Validated round, tree and sampling controls.
  std::vector<std::string> classes_;         ///< Lexical retained response classes.
  std::vector<std::string> predictor_names_; ///< Expanded design column order.
  std::vector<double>      feature_min_;     ///< Training encoded minima for decision-region axes.
  std::vector<double>      feature_max_;     ///< Training encoded maxima for decision-region axes.
  std::vector<double>      initial_scores_;  ///< Log empirical class priors.
  std::vector<std::vector<DecisionTreeRegressor>>
      trees_; ///< Round-major/class-minor residual trees.
  std::vector<std::vector<std::size_t>>
      sampled_indices_; ///< Per-round compact sampled indices; empty subvector means full sample.
  dstruct::DataFrame
      training_predictors_; ///< Owned encoded training frame for inspecting pseudo-residuals.
  std::vector<std::size_t> kept_row_indices_;  ///< Original training indices after null dropping.
  std::vector<std::string> training_labels_;   ///< Retained original response labels.
  std::vector<std::string> fitted_classes_;    ///< Full-model retained-row predictions.
  std::vector<double>      training_deviance_; ///< Per-completed-round stable mean log losses.
  std::size_t              observations_{0};   ///< Retained training row count.
};
/// @brief Owned learning-rate/round search result, including a possible prior-only choice.
/// selected_options can refit on the caller's chosen data. If selected_rounds is zero,
/// selected_options.n_trees is one solely to satisfy the fit constructor; use explicit
/// prefix zero when evaluating/predicting that selected model.
/// @par Example
/// @code
/// const auto d=dm::datasets::iris();
/// const auto search=dm::stats::gbm_validation_search(d.slice(0,120),d.slice(120,150),
///     "Species ~ Petal.Length",{2,.1,1},{.1});
/// std::print("{}\n",search.candidates.nrows()); // 3
/// @endcode
/// @see gbm_validation_search()
struct GBMValidationResult {
  GBMClassifierOptions selected_options; ///< Selected rate and refit budget max(1,selected_rounds).
  std::size_t          selected_rounds{
      0}; ///< Selected prefix, with zero representing the empirical-prior model.
  dstruct::DataFrame
      candidates; ///< All stages with learning_rate, training_mean_log_loss and training_accuracy.
};
/// @brief Selects learning rate and prefix by validation mean log loss, without touching test data.
/// Fits one full-budget model per supplied rate on train, then evaluates its fixed stages
/// on validation. Exact ties prefer the first supplied rate and earliest round, including
/// zero. All other options remain fixed. No cross-validation, calibration or refitting on
/// validation occurs here. The caller is responsible for disjoint train/validation rows;
/// no record-identity check or test frame is part of this API. Cost is one training plus
/// one staged evaluation per rate, rather than refitting every round count.
/// @param train Borrowed labelled fitting rows.
/// @param validation Borrowed labelled selection rows with the fitted classes/schema.
/// @param formula Categorical response formula.
/// @param options Maximum round budget and fixed CART/sampling controls.
/// @param learning_rates Nonempty ordered finite positive grid, default {.03,.1,.3}.
/// @return Owned selected_options, selected_rounds and per-rate/stage validation metric rows,
/// plus learning_rate, training_mean_log_loss and training_accuracy.
/// @throws std::invalid_argument Invalid grid/options/schema/classes or nonfinite arithmetic.
/// @throws std::runtime_error No complete rows or an unseen categorical predictor level.
/// @throws std::out_of_range Missing predictor columns.
/// @throws std::domain_error Invalid Formula transform.
/// @par Example
/// @code
/// const auto d=dm::datasets::iris();
/// const auto search=dm::stats::gbm_validation_search(d.slice(0,120),d.slice(120,150),
///     "Species ~ Petal.Length",{2,.1,1},{.1,.3});
/// std::print("{}\n",search.candidates.nrows()); // 6
/// @endcode
///
/// @par References
/// - J. H. Friedman (2001), Greedy function approximation: A gradient boosting machine.
///   https://doi.org/10.1214/aos/1013203451
/// @see GBMClassifier::staged_evaluation(), GBMValidationResult
[[nodiscard]] GBMValidationResult gbm_validation_search(
    const dstruct::DataFrame&  train,
    const dstruct::DataFrame&  validation,
    const std::string&         formula,
    GBMClassifierOptions       options        = {},
    const std::vector<double>& learning_rates = {.03, .1, .3});
} // namespace datamunge::stats
