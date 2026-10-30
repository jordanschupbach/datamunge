#pragma once

/// @file conformal.hpp
/// @brief Split conformal prediction: prediction sets for classifiers and prediction intervals
///        for regressors with a distribution-free, finite-sample coverage guarantee, plus the
///        exact distribution of the coverage a calibration set delivers.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief The finite-sample split-conformal quantile of nonconformity scores.
///
/// The \f$k\f$-th smallest score with \f$k=\lceil(n+1)(1-\alpha)\rceil\f$, or \f$+\infty\f$ when
/// \f$k>n\f$ (too few scores to certify the level, so every candidate must be admitted).
/// @param scores  Calibration scores (any order).
/// @param alpha   Miscoverage level in \f$(0,1)\f$.
/// @return The threshold.
/// @throws std::invalid_argument  If @p scores is empty or @p alpha is outside \f$(0,1)\f$.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::stats::conformal_quantile({5, 1, 4, 2, 3, 9, 7, 8, 6}, 0.2));  // 8 (k = 8)
/// @endcode
double conformal_quantile(std::vector<double> scores, double alpha);

/// @brief The exact coverage law of split conformal prediction with @p n calibration points.
///
/// For exchangeable data with continuous scores (no ties), the coverage of a test point is
/// \f$k/(n+1)\f$ on average over calibration and test data, so
/// \f$1-\alpha\le k/(n+1)<1-\alpha+1/(n+1)\f$. Conditional on the calibration set, the coverage
/// (the probability a fresh test point is covered) is Beta\f$(k,\,n+1-k)\f$ distributed.
struct ConformalCoverage {
    std::size_t rank{0};       ///< \f$k=\lceil(n+1)(1-\alpha)\rceil\f$ (\f$>n\f$ means the trivial set).
    double      mean{1.0};     ///< \f$k/(n+1)\f$, or 1 when \f$k>n\f$.
    double      beta_a{0.0};   ///< First Beta parameter, \f$k\f$ (0 when \f$k>n\f$).
    double      beta_b{0.0};   ///< Second Beta parameter, \f$n+1-k\f$ (0 when \f$k>n\f$).
    double      sd{0.0};       ///< Standard deviation of the conditional coverage.
};

/// @brief The coverage law for @p n calibration points at level @p alpha.
/// @param n      Calibration size (at least 1).
/// @param alpha  Miscoverage level in \f$(0,1)\f$.
/// @return Rank, mean and the Beta law of the conditional coverage.
/// @throws std::invalid_argument  If @p n is 0 or @p alpha is outside \f$(0,1)\f$.
///
/// @par Example
/// @code
/// const auto c = dm::stats::conformal_coverage(19, 0.1);
/// std::print("{} {:.4f} {:.4f}\n", c.rank, c.mean, c.sd);  // 18 0.9000 0.0655
/// @endcode
ConformalCoverage conformal_coverage(std::size_t n, double alpha);

/// @brief Settings for SplitConformalClassifier.
struct ConformalClassifierOptions {
    /// @brief Target miscoverage level: the true class lies in the prediction set with
    ///        probability at least 1 - alpha.
    double alpha{0.1};
    /// @brief Nonconformity score: "lac" (\f$1-\hat p_y\f$, the least ambiguous set-valued
    ///        classifier) or "aps" (adaptive prediction sets: the total probability of the
    ///        classes at least as likely as \f$y\f$).
    std::string score{"lac"};
    /// @brief Calibrate one threshold per class from that class's calibration rows (Mondrian
    ///        conformal), so that coverage holds within every class, not only on average.
    bool class_conditional{false};
    /// @brief Add the most probable class to a set that would otherwise be empty. The sets only
    ///        grow, so the guarantee still holds (with coverage a little above \f$k/(n+1)\f$).
    bool nonempty{false};
};

/// @brief Split (inductive) conformal prediction for classification.
///
/// Wraps any classifier that outputs class probabilities. Each calibration example's true class
/// is scored, and a test point's set holds every class whose score is at most the
/// conformal_quantile() of the calibration scores. If calibration and test data are
/// exchangeable, the set contains the true class with probability at least \f$1-\alpha\f$, whatever
/// the model and the data distribution. With `class_conditional`, the guarantee holds for each
/// class separately.
///
/// @par Example
/// @code
/// const dm::stats::SplitConformalClassifier cp({{0.9, 0.1}, {0.2, 0.8}, {0.6, 0.4}, {0.3, 0.7}},
///                                             {"no", "yes", "no", "yes"}, {"no", "yes"}, {.alpha = 0.2});
/// std::print("{}\n", cp.predict_set({0.65, 0.35})[0]);  // no (threshold 0.4, the largest of 4 scores)
/// @endcode
/// @see SplitConformalRegressor, conformal_coverage()
/// @par References
/// - V. Vovk, A. Gammerman and G. Shafer (2005). *Algorithmic Learning in a Random World*. Springer.
/// - M. Sadinle, J. Lei and L. Wasserman (2019). "Least ambiguous set-valued classifiers with
///   bounded error levels". JASA 114, 223–234.
/// - Y. Romano, M. Sesia and E. Candès (2020). "Classification with valid and adaptive coverage".
///   NeurIPS 33.
class SplitConformalClassifier {
public:
    /// @brief Calibrates from a probability matrix and class indices.
    /// @param calibration_probabilities  Rows are calibration examples, columns class probabilities
    ///                                   in the order of @p classes.
    /// @param calibration_labels         True class index of each row.
    /// @param classes                    Class names (at least 2).
    /// @param options                    Level, score and class conditioning.
    /// @throws std::invalid_argument  On bad options, mismatched sizes or labels out of range, or
    ///                                (class-conditional) a class with no calibration rows.
    SplitConformalClassifier(const linalg::DenseMatrix<double>& calibration_probabilities,
                             const std::vector<std::size_t>& calibration_labels, std::vector<std::string> classes,
                             ConformalClassifierOptions options = {});
    /// @brief Calibrates from probability rows (e.g. a classifier's `predict_detail().probability`)
    ///        and class names.
    /// @param calibration_probabilities  One probability row per calibration example.
    /// @param calibration_labels         True class name of each row.
    /// @param classes                    Class names, in column order.
    /// @param options                    Level, score and class conditioning.
    /// @throws std::invalid_argument  As the matrix constructor, or for an unknown label.
    SplitConformalClassifier(const std::vector<std::vector<double>>& calibration_probabilities,
                             const std::vector<std::string>& calibration_labels, std::vector<std::string> classes,
                             ConformalClassifierOptions options = {});

    /// @brief The miscoverage level.
    /// @return \f$\alpha\f$.
    [[nodiscard]] double alpha() const { return options_.alpha; }
    /// @brief The nonconformity score name.
    /// @return "lac" or "aps".
    [[nodiscard]] const std::string& score() const { return options_.score; }
    /// @brief The options used.
    /// @return The options.
    [[nodiscard]] const ConformalClassifierOptions& options() const { return options_; }
    /// @brief The marginal threshold: the conformal_quantile() of all calibration scores.
    /// @return \f$\hat q\f$ (\f$+\infty\f$ when the calibration set is too small).
    [[nodiscard]] double quantile() const { return quantile_; }
    /// @brief The thresholds in use, one per class (all equal to quantile() unless class-conditional).
    /// @return One threshold per class.
    [[nodiscard]] const std::vector<double>& class_quantiles() const { return class_quantiles_; }
    /// @brief Number of calibration rows.
    /// @return \f$n\f$.
    [[nodiscard]] std::size_t n_calibration() const { return n_cal_; }
    /// @brief Class names in column order.
    /// @return The classes.
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    /// @brief The score of each calibration row's true class, in input order.
    /// @return \f$n\f$ scores.
    [[nodiscard]] const std::vector<double>& calibration_scores() const { return scores_; }
    /// @brief The marginal coverage law for this calibration size and level.
    /// @return conformal_coverage(n_calibration(), alpha()).
    [[nodiscard]] ConformalCoverage coverage() const;

    /// @brief Nonconformity score of class @p k for one probability row.
    /// @param probabilities  Class probabilities.
    /// @param k              Class index.
    /// @return The score (larger means less conforming).
    [[nodiscard]] double nonconformity(const std::vector<double>& probabilities, std::size_t k) const;
    /// @brief Per-class set membership (1/0), aligned with classes(), for one probability row:
    ///        the classes whose score is at most their threshold, plus the most probable class
    ///        when `nonempty` is set and no class qualifies.
    /// @param probabilities  Class probabilities.
    /// @return One flag per class.
    [[nodiscard]] std::vector<char> membership(const std::vector<double>& probabilities) const;
    /// @brief The prediction set as class names.
    /// @param probabilities  Class probabilities.
    /// @return The classes admitted, in classes() order.
    [[nodiscard]] std::vector<std::string> predict_set(const std::vector<double>& probabilities) const;
    /// @brief Prediction sets for many rows.
    /// @param probabilities  One probability row per test point.
    /// @return One set per row.
    [[nodiscard]] std::vector<std::vector<std::string>> predict_sets(const std::vector<std::vector<double>>& probabilities) const;
    /// @brief Conformal p-value of each class for one probability row.
    ///
    /// \f$p_k=(1+\#\{i:s_i\ge s_k\})/(n+1)\f$ over the calibration scores (of class \f$k\f$ only,
    /// when class-conditional). Under exchangeability the true class's p-value is
    /// super-uniform, \f$P(p_y\le t)\le t\f$, and the set is \f$\{k:p_k>\alpha\}\f$ up to ties (before
    /// the `nonempty` fallback).
    /// @param probabilities  Class probabilities.
    /// @return One p-value per class.
    [[nodiscard]] std::vector<double> p_values(const std::vector<double>& probabilities) const;

    /// @brief Coverage and set-size statistics on a labelled test set.
    struct Evaluation {
        double                   coverage{0.0};          ///< Share of test points whose set holds the truth.
        double                   average_set_size{0.0};  ///< Mean set size.
        double                   empty_rate{0.0};        ///< Share of empty sets.
        std::vector<std::size_t> set_sizes;              ///< Set size per test point.
        std::vector<double>      class_coverage;         ///< Coverage among test points of each class (NaN if none).
        std::vector<std::size_t> size_counts;            ///< Number of sets of each size 0..K.
    };
    /// @brief Evaluates on a test set given as a matrix and class indices.
    /// @param test_probabilities  One probability row per test point.
    /// @param test_labels         True class index per row.
    /// @return The statistics.
    /// @throws std::invalid_argument  If the sizes differ.
    [[nodiscard]] Evaluation evaluate(const linalg::DenseMatrix<double>& test_probabilities,
                                      const std::vector<std::size_t>& test_labels) const;
    /// @brief Evaluates on a test set given as probability rows and class names.
    /// @param test_probabilities  One probability row per test point.
    /// @param test_labels         True class name per row.
    /// @return The statistics.
    /// @throws std::invalid_argument  If the sizes differ or a label is unknown.
    [[nodiscard]] Evaluation evaluate(const std::vector<std::vector<double>>& test_probabilities,
                                      const std::vector<std::string>& test_labels) const;

private:
    /// @brief Shared calibration once the inputs are rows and indices.
    /// @param rows    Probability rows.
    /// @param labels  Class indices.
    void calibrate(const std::vector<std::vector<double>>& rows, const std::vector<std::size_t>& labels);
    /// @brief Index of a class name.
    /// @param label  Class name.
    /// @return Its position in classes().
    [[nodiscard]] std::size_t class_index(const std::string& label) const;

    ConformalClassifierOptions options_;          ///< Settings.
    std::vector<std::string>   classes_;          ///< Class names.
    std::size_t                n_cal_{0};         ///< Calibration size.
    double                     quantile_{0.0};    ///< Marginal threshold.
    std::vector<double>        class_quantiles_;  ///< Threshold used per class.
    std::vector<double>        scores_;           ///< Calibration scores.
    std::vector<std::size_t>   labels_;           ///< Calibration labels.
};

/// @brief Settings for SplitConformalRegressor.
struct ConformalRegressorOptions {
    /// @brief Target miscoverage level.
    double alpha{0.1};
};

/// @brief Split conformal prediction intervals for regression.
///
/// The calibration score is \f$s_i=\max(\ell_i-y_i,\;y_i-u_i)/\sigma_i\f$ for a base band
/// \f$[\ell_i,u_i]\f$ and a positive scale \f$\sigma_i\f$; with \f$\hat q\f$ the conformal_quantile()
/// of the scores, the interval for a new point is \f$[\ell-\hat q\sigma,\;u+\hat q\sigma]\f$. Three
/// choices of band and scale give the standard methods:
/// - *absolute residual*: \f$\ell=u=\hat y\f$, \f$\sigma=1\f$, a constant width \f$2\hat q\f$ (the
///   predictions-and-truth constructor);
/// - *normalised residual*: \f$\ell=u=\hat y\f$ and \f$\sigma\f$ an estimate of the spread at
///   \f$x\f$, so widths adapt to heteroscedastic noise (normalized());
/// - *conformalized quantile regression* (CQR): \f$\ell,u\f$ fitted lower and upper quantiles,
///   \f$\sigma=1\f$, so the band is widened or narrowed by \f$\hat q\f$ (quantile_band()).
///
/// All three cover a new response with probability at least \f$1-\alpha\f$ under exchangeability.
///
/// @par Example
/// @code
/// const dm::stats::SplitConformalRegressor cp({0, 0, 0, 0}, {0.5, -1.0, 2.0, -0.2}, {.alpha = 0.2});
/// const auto [lo, hi] = cp.predict_interval(10.0);
/// std::print("{} {}\n", lo, hi);  // 8 12
/// @endcode
/// @see SplitConformalClassifier, conformal_coverage()
/// @par References
/// - J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani and L. Wasserman (2018). "Distribution-free
///   predictive inference for regression". JASA 113, 1094–1111.
/// - Y. Romano, E. Patterson and E. Candès (2019). "Conformalized quantile regression". NeurIPS 32.
class SplitConformalRegressor {
public:
    /// @brief Absolute-residual calibration.
    /// @param calibration_predictions  Point predictions \f$\hat y_i\f$.
    /// @param calibration_truth        Responses \f$y_i\f$.
    /// @param options                  Level.
    /// @throws std::invalid_argument  On a bad level, mismatched or empty inputs.
    SplitConformalRegressor(const std::vector<double>& calibration_predictions,
                            const std::vector<double>& calibration_truth, ConformalRegressorOptions options = {});
    /// @brief General calibration with band and scale.
    /// @param calibration_lower  Band lower ends \f$\ell_i\f$.
    /// @param calibration_upper  Band upper ends \f$u_i\f$.
    /// @param calibration_scale  Positive scales \f$\sigma_i\f$.
    /// @param calibration_truth  Responses \f$y_i\f$.
    /// @param options            Level.
    /// @throws std::invalid_argument  On a bad level, mismatched or empty inputs, or a scale \f$\le0\f$.
    SplitConformalRegressor(const std::vector<double>& calibration_lower, const std::vector<double>& calibration_upper,
                            const std::vector<double>& calibration_scale, const std::vector<double>& calibration_truth,
                            ConformalRegressorOptions options = {});
    /// @brief Normalised-residual calibration, \f$|y_i-\hat y_i|/\sigma_i\f$.
    /// @param calibration_predictions  Point predictions.
    /// @param calibration_scale        Positive spread estimates.
    /// @param calibration_truth        Responses.
    /// @param options                  Level.
    /// @return The calibrated regressor.
    [[nodiscard]] static SplitConformalRegressor normalized(const std::vector<double>& calibration_predictions,
                                                            const std::vector<double>& calibration_scale,
                                                            const std::vector<double>& calibration_truth,
                                                            ConformalRegressorOptions options = {});
    /// @brief Conformalized quantile regression, \f$\max(\ell_i-y_i,\,y_i-u_i)\f$.
    /// @param calibration_lower  Fitted lower quantiles.
    /// @param calibration_upper  Fitted upper quantiles.
    /// @param calibration_truth  Responses.
    /// @param options            Level.
    /// @return The calibrated regressor.
    [[nodiscard]] static SplitConformalRegressor quantile_band(const std::vector<double>& calibration_lower,
                                                               const std::vector<double>& calibration_upper,
                                                               const std::vector<double>& calibration_truth,
                                                               ConformalRegressorOptions options = {});

    /// @brief The miscoverage level.
    /// @return \f$\alpha\f$.
    [[nodiscard]] double alpha() const { return options_.alpha; }
    /// @brief The calibrated score quantile (the half-width for absolute residuals).
    /// @return \f$\hat q\f$ (\f$+\infty\f$ when the calibration set is too small).
    [[nodiscard]] double quantile() const { return quantile_; }
    /// @brief Number of calibration points.
    /// @return \f$n\f$.
    [[nodiscard]] std::size_t n_calibration() const { return n_cal_; }
    /// @brief The calibration scores, in input order.
    /// @return \f$n\f$ scores.
    [[nodiscard]] const std::vector<double>& calibration_scores() const { return scores_; }
    /// @brief The coverage law for this calibration size and level.
    /// @return conformal_coverage(n_calibration(), alpha()).
    [[nodiscard]] ConformalCoverage coverage() const;

    /// @brief \f$[\hat y-\hat q,\;\hat y+\hat q]\f$.
    /// @param prediction  Point prediction.
    /// @return The interval.
    [[nodiscard]] std::pair<double, double> predict_interval(double prediction) const;
    /// @brief \f$[\ell-\hat q\sigma,\;u+\hat q\sigma]\f$.
    /// @param lower  Band lower end.
    /// @param upper  Band upper end.
    /// @param scale  Positive scale.
    /// @return The interval.
    [[nodiscard]] std::pair<double, double> predict_interval(double lower, double upper, double scale) const;

    /// @brief Coverage and width statistics on a test set.
    struct Evaluation {
        double            coverage{0.0};       ///< Share of responses inside their interval.
        double            average_width{0.0};  ///< Mean interval width.
        std::vector<char> covered;             ///< 1 where the response is inside, per test point.
        std::vector<double> widths;            ///< Interval width per test point.
    };
    /// @brief Evaluates absolute-residual intervals.
    /// @param test_predictions  Point predictions.
    /// @param test_truth        Responses.
    /// @return The statistics.
    /// @throws std::invalid_argument  If the sizes differ.
    [[nodiscard]] Evaluation evaluate(const std::vector<double>& test_predictions,
                                      const std::vector<double>& test_truth) const;
    /// @brief Evaluates general band-and-scale intervals.
    /// @param test_lower  Band lower ends.
    /// @param test_upper  Band upper ends.
    /// @param test_scale  Scales.
    /// @param test_truth  Responses.
    /// @return The statistics.
    /// @throws std::invalid_argument  If the sizes differ.
    [[nodiscard]] Evaluation evaluate(const std::vector<double>& test_lower, const std::vector<double>& test_upper,
                                      const std::vector<double>& test_scale, const std::vector<double>& test_truth) const;

private:
    ConformalRegressorOptions options_;        ///< Settings.
    std::size_t               n_cal_{0};       ///< Calibration size.
    double                    quantile_{0.0};  ///< Score quantile.
    std::vector<double>       scores_;         ///< Calibration scores.
};

} // namespace datamunge::stats
