#pragma once

/// @file knn_classifier.hpp
/// @brief k-nearest-neighbour classification: majority (or inverse-distance weighted) vote of
///        the k training points closest to the query, with exact leave-one-out evaluation.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/detail/knn_common.hpp>
#include <datamunge/stats/formula.hpp>

#include <array>
#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Settings for KNNClassifier.
struct KNNClassifierOptions {
    /// @brief Number of neighbours that vote; at least 1.
    std::size_t k{5};
    /// @brief Distance between standardised predictor vectors.
    DistanceMetric metric{DistanceMetric::Euclidean};
    /// @brief Uniform votes (false) or votes weighted by \f$1/d\f$ (true).
    bool weighted{false};
    /// @brief Rescale each predictor to mean 0 and standard deviation 1 (divisor \f$n\f$, as
    ///        scikit-learn's `StandardScaler`) before measuring distances, so no predictor
    ///        dominates because of its units.
    bool standardize{true};
};

/// @brief Predicted classes with the vote shares behind them.
struct KNNClassifierPrediction {
    /// @brief Predicted class per row of the new data; "" where a predictor was missing.
    std::vector<std::string> class_label;
    /// @brief Share of the (weighted) votes per class, in classes() order; empty where a
    ///        predictor was missing.
    std::vector<std::vector<double>> vote_share;
};

/// @brief k-nearest-neighbour classifier from a formula `class ~ x1 + x2 + ...`.
///
/// The model stores the (standardised) training points. A query is assigned the class with the
/// largest share of votes among its k nearest training points. Ties in distance are broken by
/// training-row order and ties in the vote by the order of classes() (sorted labels), as in
/// scikit-learn. Each standardised difference is computed as the raw difference times
/// \f$1/s_j\f$, so points tied in the raw data stay exactly tied. Factor predictors enter as their dummy columns.
///
/// There is no meaningful resubstitution accuracy (with \f$k=1\f$ every training point is its own
/// nearest neighbour), so fitted_classes(), training_accuracy() and confusion_matrix() are
/// leave-one-out: each row is classified by the other \f$n-1\f$ rows, with the standardisation
/// recomputed without it, exactly as refitting \f$n\f$ times would.
///
/// @par Example
/// @code
/// const auto iris = dm::datasets::iris();
/// const dm::stats::KNNClassifier knn(iris, "Species ~ Petal.Length + Petal.Width", {.k = 5});
/// std::print("{:.4f}\n", knn.training_accuracy());  // 0.9667 (leave-one-out)
/// @endcode
///
/// @par References
/// - T. M. Cover and P. E. Hart (1967). "Nearest neighbor pattern classification". *IEEE
///   Transactions on Information Theory* 13(1), 21–27.
/// - scikit-learn `KNeighborsClassifier`:
///   https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
class KNNClassifier {
 public:
    /// @brief Fits the classifier.
    /// @param data     Data frame holding the response (a string column) and the predictors.
    /// @param formula  `class ~ predictors`; rows with a missing value are dropped.
    /// @param options  k, metric, voting and standardisation.
    /// @throws std::invalid_argument  If k is 0 or the response has fewer than 2 classes.
    KNNClassifier(const dstruct::DataFrame& data, const std::string& formula, KNNClassifierOptions options = {});

    /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
    ///
    /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
    /// converts to Formula implicitly, so `KNNClassifier("y ~ x", data)` also works.
    /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
    /// @param data  Data frame holding every column the formula names.
    /// @param options  Same as in the `(data, formula, ...)` constructor.
    /// @throws std::invalid_argument  As the `(data, formula, ...)` constructor.
    KNNClassifier(const Formula& formula, const dstruct::DataFrame& data, KNNClassifierOptions options = {})
        : KNNClassifier(data, formula.text(), std::move(options)) {}

    /// @brief The formula the model was fitted with.
    /// @return Formula text.
    [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
    /// @brief The class labels, sorted; every vote-share vector follows this order.
    /// @return Sorted distinct labels.
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    /// @brief Names of the predictor columns (dummy columns for factors).
    /// @return Predictor names.
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    /// @brief Number of training rows used (after dropping missing values).
    /// @return \f$n\f$.
    [[nodiscard]] std::size_t observations() const { return observations_; }
    /// @brief Number of voting neighbours.
    /// @return k.
    [[nodiscard]] std::size_t k() const { return options_.k; }
    /// @brief The options the model was fitted with.
    /// @return Options.
    [[nodiscard]] const KNNClassifierOptions& options() const { return options_; }

    /// @brief Leave-one-out predicted class of each training row.
    /// @return \f$n\f$ labels.
    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    /// @brief Observed class of each training row.
    /// @return \f$n\f$ labels.
    [[nodiscard]] const std::vector<std::string>& training_labels() const { return training_labels_; }
    /// @brief Leave-one-out accuracy (not a resubstitution accuracy).
    /// @return Share of rows whose leave-one-out prediction is correct.
    [[nodiscard]] double training_accuracy() const;
    /// @brief Leave-one-out confusion matrix.
    /// @return Counts; rows are the observed class and columns the predicted class, both in
    ///         classes() order.
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;
    /// @brief Leave-one-out accuracy for every \f$k=1,\dots,K\f$ at once, with this model's metric,
    ///        voting and standardisation.
    ///
    /// Each row's neighbours are sorted once, so the whole path costs about as much as one
    /// leave-one-out evaluation.
    /// @param max_k  \f$K\f$; reduced to \f$n-1\f$ if larger.
    /// @return Accuracies; element \f$k-1\f$ is the accuracy with \f$k\f$ neighbours.
    /// @throws std::invalid_argument  If @p max_k is 0.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::KNNClassifier knn(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    /// std::print("{:.4f}\n", knn.loo_accuracy_path(15)[4]);  // 0.9667, the same as k = 5
    /// @endcode
    [[nodiscard]] std::vector<double> loo_accuracy_path(std::size_t max_k) const;

    /// @brief The k nearest training rows of each row of @p newdata.
    /// @param newdata  Rows to query; needs the predictor columns.
    /// @return Training-row positions and distances, nearest first (empty where a predictor is
    ///         missing). Distances are on the standardised scale when `standardize` is set.
    [[nodiscard]] KNNNeighbors kneighbors(const dstruct::DataFrame& newdata) const;

    /// @brief A short text summary (formula, k, metric, classes, leave-one-out accuracy).
    /// @return Summary text.
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to @p os.
    /// @param os  Output stream.
    void print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    void print_summary() const;

    /// @brief Predicted classes for new rows.
    /// @param newdata  Rows to classify.
    /// @return One label per row ("" where a predictor is missing).
    /// @throws std::invalid_argument  If a factor predictor has a level unseen in training.
    [[nodiscard]] std::vector<std::string> predict(const dstruct::DataFrame& newdata) const;
    /// @brief Predicted classes with their vote shares.
    /// @param newdata  Rows to classify.
    /// @return Labels and vote shares.
    /// @throws std::invalid_argument  If a factor predictor has a level unseen in training.
    [[nodiscard]] KNNClassifierPrediction predict_detail(const dstruct::DataFrame& newdata) const;

    /// @brief Scatter of @p data in the (@p x_feature, @p y_feature) plane coloured by observed
    ///        class, with misclassified points (by predict()) marked.
    /// @param data       Rows to draw.
    /// @param x_feature  Column on the x axis.
    /// @param y_feature  Column on the y axis.
    /// @return The plot.
    [[nodiscard]] graphics::RPlot plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                      const std::string& y_feature) const;

    /// @brief The predicted class over the plane of the two predictors, shaded in pale class
    ///        colours, with the training points on top.
    ///
    /// The plane is @p limits, or else the training range padded by 5% in x and 10% in y, cut
    /// into @p grid_resolution × @p grid_resolution cells; each cell takes the class predicted at
    /// its centre, and runs of equal cells along a row are drawn as one rectangle.
    /// @param x_feature        Predictor on the x axis.
    /// @param y_feature        Predictor on the y axis.
    /// @param grid_resolution  Cells per axis (at least 2).
    /// @param limits           Optional box {x_min, x_max, y_min, y_max}, e.g. to leave room for
    ///                         the legend at the top right.
    /// @return The plot, with its axis limits set to the shaded box.
    /// @throws std::invalid_argument  If the model does not have exactly these two predictors, or
    ///         @p limits is not increasing.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::KNNClassifier knn(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    /// knn.plot_decision_regions("Petal.Length", "Petal.Width").save_svg("regions.svg");
    /// @endcode
    [[nodiscard]] graphics::RPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                        std::size_t grid_resolution = 150,
                                                        std::optional<std::array<double, 4>> limits = std::nullopt) const;

 private:
    /// @brief Builds the design, the scaling and the leave-one-out predictions.
    /// @param data  Training data.
    void fit(const dstruct::DataFrame& data);
    /// @brief Vote shares of the k nearest training rows to a raw query point.
    /// @param query        Raw predictor values.
    /// @param exclude_row  Training row to skip (leave-one-out), or `SIZE_MAX`.
    /// @param mult         Per-predictor multipliers; empty for the fitted \f$1/s_j\f$.
    /// @return Shares in classes() order.
    std::vector<double> vote_shares(const std::vector<double>& query, std::size_t exclude_row = SIZE_MAX,
                                    const std::vector<double>& mult = {}) const;
    /// @brief Row @p row of @p X as a vector.
    /// @param X    Matrix.
    /// @param row  Row index.
    /// @return The row.
    std::vector<double> row_of(const linalg::DenseMatrix<double>& X, std::size_t row) const;

    Formula              formula_;          ///< The model formula.
    DesignInfo           design_;           ///< Resolved predictor columns.
    KNNClassifierOptions options_;          ///< Fitting options.

    std::vector<std::string> classes_;          ///< Sorted class labels.
    std::vector<std::string> predictor_names_;  ///< Predictor column names.
    detail::KnnScaling       scaling_;          ///< Means and scales of the predictors.
    std::vector<double>      feature_min_;      ///< Per-predictor minimum, for plot_decision_regions().
    std::vector<double>      feature_max_;      ///< Per-predictor maximum, for plot_decision_regions().

    linalg::DenseMatrix<double> training_X_;            ///< Raw training predictors.
    std::vector<double>         multipliers_;           ///< \f$1/s_j\f$ per predictor.
    std::vector<std::size_t>    training_class_index_;  ///< Class of each row, as an index into classes_.
    std::vector<std::string>    training_labels_;       ///< Class of each row.
    std::vector<std::string>    fitted_classes_;        ///< Leave-one-out predictions.
    std::size_t                 observations_{0};       ///< Number of training rows.
};

/// @brief Large-sample conditional error of the k-nearest-neighbour rule for two classes.
///
/// As \f$n\to\infty\f$ the k neighbours of \f$x\f$ converge to \f$x\f$, so their labels are
/// independent Bernoulli(\f$\eta\f$) draws with \f$\eta=P(Y=1\mid x)\f$. With \f$B\sim\text{Bin}(k,\eta)\f$
/// votes for class 1, the rule errs with probability
/// \f$\eta P(B<k/2)+(1-\eta)P(B>k/2)+\tfrac12P(B=k/2)\f$ (a tie, possible for even k, is
/// counted as a fair coin). For \f$k=1\f$ this is \f$2\eta(1-\eta)\f$; averaging over \f$x\f$ gives the
/// asymptotic risk, which Cover and Hart (1967) bound by \f$2R^*(1-R^*)\f$ for \f$k=1\f$, where \f$R^*\f$
/// is the Bayes risk.
/// @param eta  \f$P(Y=1\mid x)\f$, in \f$[0,1]\f$.
/// @param k    Number of neighbours, at least 1.
/// @return The conditional error probability.
/// @throws std::invalid_argument  If @p eta is outside \f$[0,1]\f$ or @p k is 0.
///
/// @par Example
/// @code
/// std::print("{:.4f} {:.4f}\n", dm::stats::knn_asymptotic_risk(0.2, 1),
///            dm::stats::knn_asymptotic_risk(0.2, 5));  // 0.3200 0.2348
/// @endcode
[[nodiscard]] double knn_asymptotic_risk(double eta, std::size_t k);

} // namespace datamunge::stats
