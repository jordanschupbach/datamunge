#pragma once
/// @file random_forest_classifier.hpp
/// @brief Random forest classification: bagged CART trees with random predictor subsets at every
///        split, majority voting, out-of-bag accuracy, margins, Breiman's strength and
///        correlation, and impurity and permutation importance.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/decision_tree_classifier.hpp>
#include <datamunge/stats/formula.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Options for RandomForestClassifier.
struct RandomForestClassifierOptions {
    std::size_t    n_trees{100};          ///< Number of trees \f$B\f$ (at least 1).
    std::size_t    max_depth{10};         ///< Depth limit of each tree.
    std::size_t    min_samples_split{2};  ///< A node with fewer rows is not split.
    std::size_t    min_samples_leaf{1};   ///< Each child of a split keeps at least this many rows.
    /** @brief Predictors \f$m\f$ drawn at random as split candidates at every node. 0 means
        \f$\lfloor\sqrt p\rfloor\f$ (at least 1), the usual classification default; \f$m=p\f$ is bagging. */
    std::size_t    max_features{0};
    SplitCriterion criterion{SplitCriterion::Gini}; ///< Impurity used to choose splits.
    /** @brief Draw each tree's rows with replacement (true, bagging) or as a subsample without
        replacement (false). */
    bool           bootstrap{true};
    double         sample_fraction{1.0};  ///< Rows drawn per tree, as a fraction of the training rows.
    std::uint64_t  seed{42};              ///< Seed for the row draws and the predictor subsets.
};

/// @brief Predictions of a RandomForestClassifier for a data frame.
struct RandomForestClassifierPrediction {
    std::vector<std::string>         class_label; ///< Majority-vote class per row; "" if a predictor was missing.
    std::vector<std::vector<double>> vote_share;  ///< [row][class] fraction of trees voting for each class.
};

/// @brief Breiman's (2001) strength and correlation of a forest, estimated out of bag.
///
/// The margin of a row is \f$\mathrm{mr}=Q_y-\max_{j\ne y}Q_j\f$, where \f$Q_j\f$ is its out-of-bag
/// vote share for class \f$j\f$. The strength is \f$s=\bar{\mathrm{mr}}\f$, and the mean correlation
/// between trees is \f$\bar\rho=\mathrm{var}(\mathrm{mr})/(\overline{\mathrm{sd}})^2\f$, where
/// \f$\mathrm{sd}_t=\sqrt{p_{1t}+p_{2t}-(p_{1t}-p_{2t})^2}\f$ and \f$p_{1t}\f$, \f$p_{2t}\f$ are the shares of
/// tree \f$t\f$'s out-of-bag rows on which it votes for the true class and for the strongest wrong
/// class. Chebyshev's inequality gives the bound \f$\mathrm{PE}^*\le\bar\rho(1-s^2)/s^2\f$ on the
/// generalisation error (when \f$s>0\f$).
struct RandomForestStrength {
    double strength{0.0};         ///< \f$s\f$, the mean out-of-bag margin.
    double margin_variance{0.0};  ///< \f$\mathrm{var}(\mathrm{mr})\f$ over rows.
    double mean_tree_sd{0.0};     ///< \f$\overline{\mathrm{sd}}\f$, the mean over trees of \f$\mathrm{sd}_t\f$.
    double correlation{0.0};      ///< \f$\bar\rho\f$.
    double error_bound{0.0};      ///< \f$\bar\rho(1-s^2)/s^2\f$; infinity when \f$s\le0\f$.
    std::size_t rows{0};          ///< Rows that were out of bag for at least one tree.
};

/// @brief Out-of-bag permutation importance (Breiman's "mean decrease in accuracy").
struct RandomForestPermutationImportance {
    std::vector<std::string> predictor_names; ///< Design columns, as RandomForestClassifier::predictor_names().
    /** @brief Mean over trees of (tree's out-of-bag accuracy) minus (its accuracy after permuting
        the predictor among its out-of-bag rows). */
    std::vector<double> importance;
    std::vector<double> standard_error;       ///< Standard deviation over trees divided by \f$\sqrt B\f$.
};

/// @brief Random forest classifier: \f$B\f$ CART trees, each fit to a bootstrap sample of the rows and
///        restricted to \f$m\f$ random predictors at every split, combined by majority vote.
///
/// Built from a data frame and an R-style formula whose response is a string column, like the
/// other stats classifiers. Rows with a missing predictor or response are dropped before fitting,
/// and every per-row accessor (fitted classes, out-of-bag votes, margins) refers to the remaining
/// rows in order. Each row is out of bag for the trees whose bootstrap sample missed it, about
/// \f$e^{-1}\approx37\%\f$ of them, and the votes of those trees give an accuracy estimate that needs no
/// held-out data (oob_accuracy()). Ties in a vote go to the class that sorts first.
///
/// @par Example
/// @code
/// const dm::stats::RandomForestClassifier forest(dm::datasets::iris(), "Species ~ .", {.n_trees = 200});
/// std::print("{} trees, m = {}, OOB accuracy {:.4f}\n", forest.n_trees(), forest.max_features_used(),
///            forest.oob_accuracy()); // 200 trees, m = 2, OOB accuracy 0.9600
/// @endcode
class RandomForestClassifier {
 public:
    /// @brief Fits the forest.
    /// @param data     Training data.
    /// @param formula  Formula `class ~ predictors`; the response must be a string column.
    /// @param options  Forest options.
    /// @throws std::invalid_argument  If n_trees is 0, sample_fraction is not positive, the
    ///         response has fewer than two classes, or the formula does not resolve.
    /// @throws std::runtime_error  If no bootstrap sample with two classes is drawn in 25 tries.
    RandomForestClassifier(const dstruct::DataFrame& data, const std::string& formula,
                           RandomForestClassifierOptions options = {});

    /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
    ///
    /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
    /// converts to Formula implicitly, so `RandomForestClassifier("y ~ x", data)` also works.
    /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
    /// @param data  Data frame holding every column the formula names.
    /// @param options  Same as in the `(data, formula, ...)` constructor.
    /// @throws std::invalid_argument  As the `(data, formula, ...)` constructor.
    RandomForestClassifier(const Formula& formula, const dstruct::DataFrame& data, RandomForestClassifierOptions options = {})
        : RandomForestClassifier(data, formula.text(), std::move(options)) {}

    /// @brief The formula text the forest was fit with.
    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    /// @brief The classes, sorted; class indices everywhere refer to this order.
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    /// @brief The design columns (predictors, with categorical ones expanded to dummies).
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    /// @brief Training rows used (after dropping rows with missing values).
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    /// @brief Number of trees \f$B\f$.
    [[nodiscard]] std::size_t                     n_trees() const { return trees_.size(); }
    /// @brief Predictors tried at each split, \f$m\f$ (the resolved max_features).
    [[nodiscard]] std::size_t                     max_features_used() const { return max_features_used_; }
    /// @brief The options the forest was fit with.
    [[nodiscard]] const RandomForestClassifierOptions& options() const { return options_; }

    /// @brief Tree @p t of the forest.
    /// @param t  Tree index, below n_trees().
    /// @return The tree.
    /// @throws std::out_of_range  If @p t is out of range.
    [[nodiscard]] const DecisionTreeClassifier& tree(std::size_t t) const;

    /// @brief How many times each training row was drawn into tree @p t's sample (0 = out of bag).
    /// @param t  Tree index, below n_trees().
    /// @return One count per training row.
    /// @throws std::out_of_range  If @p t is out of range.
    [[nodiscard]] std::vector<int> in_bag_counts(std::size_t t) const;

    /// @brief Mean decrease in impurity: each tree's normalised impurity-decrease importance,
    ///        averaged over trees and normalised to sum to 1.
    /// @return One value per predictor_names() entry.
    [[nodiscard]] std::vector<double> feature_importance() const;

    /// @brief Out-of-bag permutation importance.
    ///
    /// For each tree and predictor, the predictor's values are shuffled among the tree's
    /// out-of-bag rows and the drop in that tree's out-of-bag accuracy is recorded. Unlike
    /// feature_importance(), a predictor the response does not depend on scores about 0.
    /// @param seed  Seed for the shuffles.
    /// @return Mean decrease and standard error per predictor.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::RandomForestClassifier forest(dm::datasets::iris(), "Species ~ .");
    /// const auto pi = forest.permutation_importance();
    /// for (std::size_t j = 0; j < pi.importance.size(); ++j)
    ///     std::print("{} {:.3f}\n", pi.predictor_names[j], pi.importance[j]);
    /// @endcode
    [[nodiscard]] RandomForestPermutationImportance permutation_importance(std::uint64_t seed = 1) const;

    /// @brief Resubstitution predictions: the forest's vote for each training row.
    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    /// @brief Share of training rows whose resubstitution prediction is correct (optimistic).
    [[nodiscard]] double                          training_accuracy() const;
    /// @brief Out-of-bag accuracy: each row is predicted by majority vote of the trees that did
    ///        not train on it; rows never out of bag are skipped. A near-unbiased estimate of
    ///        test accuracy.
    [[nodiscard]] double                          oob_accuracy() const { return oob_accuracy_; }
    /// @brief Resubstitution confusion matrix: rows are the actual class, columns the predicted
    ///        class, both in classes() order.
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;

    /// @brief Number of trees for which each training row is out of bag.
    [[nodiscard]] std::vector<int> oob_tree_counts() const;
    /// @brief Out-of-bag vote shares, [row][class]; an empty row if the row was never out of bag.
    [[nodiscard]] std::vector<std::vector<double>> oob_vote_share() const;
    /// @brief Out-of-bag majority-vote class per training row; "" if never out of bag.
    [[nodiscard]] std::vector<std::string> oob_predictions() const;
    /// @brief Out-of-bag margin \f$Q_y-\max_{j\ne y}Q_j\f$ per training row, in \f$[-1,1]\f$; NaN if never
    ///        out of bag. A negative margin means the row is misclassified out of bag.
    [[nodiscard]] std::vector<double> oob_margins() const;
    /// @brief Out-of-bag accuracy of the first \f$t\f$ trees, for \f$t=1,\dots,B\f$, from the one fit.
    ///
    /// Element \f$t-1\f$ uses the rows that are out of bag for at least one of the first \f$t\f$ trees.
    /// The last element equals oob_accuracy().
    [[nodiscard]] std::vector<double> oob_accuracy_path() const;
    /// @brief Each tree's own accuracy on its out-of-bag rows (NaN if it has none).
    [[nodiscard]] std::vector<double> oob_tree_accuracies() const;
    /// @brief Breiman's strength, correlation and error bound, estimated out of bag.
    /// @return The estimates; see RandomForestStrength.
    [[nodiscard]] RandomForestStrength strength_correlation() const;

    /// @brief Summary in the style of R's `print(randomForest)`: call, classes, importance,
    ///        training and out-of-bag accuracy.
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to @p os.
    /// @param os  Output stream.
    void                      print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    void                      print_summary() const;

    /// @brief Majority-vote class per row of @p newdata ("" if a predictor is missing).
    /// @param newdata  Data frame with the predictor columns.
    /// @return One class label per row.
    /// @throws std::runtime_error  If a categorical predictor has an unseen level.
    [[nodiscard]] std::vector<std::string>        predict(const dstruct::DataFrame& newdata) const;
    /// @brief Majority-vote class and vote shares per row of @p newdata.
    /// @param newdata  Data frame with the predictor columns.
    /// @return The predictions.
    /// @throws std::runtime_error  If a categorical predictor has an unseen level.
    [[nodiscard]] RandomForestClassifierPrediction predict_detail(const dstruct::DataFrame& newdata) const;

    /// @brief Each tree's vote at one point given as numbers.
    /// @param x  Predictor values, ordered as predictor_names().
    /// @return Class index (into classes()) voted by each tree, in tree order.
    /// @throws std::invalid_argument  If @p x does not have one value per predictor.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::RandomForestClassifier forest(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    /// const auto votes = forest.tree_votes({4.8, 1.7});
    /// std::print("tree 0 says {}\n", forest.classes()[votes[0]]);
    /// @endcode
    [[nodiscard]] std::vector<std::size_t> tree_votes(const std::vector<double>& x) const;
    /// @brief Vote share per class at one point given as numbers.
    /// @param x  Predictor values, ordered as predictor_names().
    /// @return Fraction of trees voting for each class, in classes() order.
    /// @throws std::invalid_argument  If @p x does not have one value per predictor.
    [[nodiscard]] std::vector<double> vote_share(const std::vector<double>& x) const;

    /// @brief Scatter of @p data in the (x_feature, y_feature) plane, coloured by true class, with
    ///        a black marker on rows the forest misclassifies (resubstitution).
    /// @param data       Data frame with the response and both features.
    /// @param x_feature  Column on the x axis.
    /// @param y_feature  Column on the y axis.
    /// @return The plot.
    [[nodiscard]] graphics::RPlot plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                      const std::string& y_feature) const;

    /// @brief The majority-vote class over the plane of the forest's two predictors, shaded by
    ///        how unanimous the vote is, with the training points on top.
    ///
    /// Each class has three tints: darkest where at least 90% of trees agree, medium for 60% to
    /// 90%, palest below 60%.
    /// @param x_feature        Predictor on the x axis.
    /// @param y_feature        Predictor on the y axis.
    /// @param grid_resolution  Cells per axis.
    /// @param limits           Optional box {x_min, x_max, y_min, y_max}; default the training range
    ///                         padded by 5% in x and 10% in y.
    /// @return The plot, with its axis limits set to the box.
    /// @throws std::invalid_argument  If the model does not have exactly these two predictors.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::RandomForestClassifier forest(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    /// forest.plot_decision_regions("Petal.Length", "Petal.Width").save_svg("forest_regions.svg");
    /// @endcode
    [[nodiscard]] graphics::RPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                        std::size_t grid_resolution = 150,
                                                        std::optional<std::array<double, 4>> limits = std::nullopt) const;

 private:
    /// @brief Draws the samples, grows the trees and computes the out-of-bag votes.
    /// @param data     Training data.
    /// @param options  Forest options.
    void fit(const dstruct::DataFrame& data, RandomForestClassifierOptions options);
    /// @brief Tree @p t's vote (forest class index) for a point in forest column order.
    /// @param t  Tree index.
    /// @param x  Predictor values, ordered as predictor_names().
    /// @param buffer  Scratch space for the tree's own column order.
    /// @return Index into classes().
    [[nodiscard]] std::size_t tree_vote(std::size_t t, const std::vector<double>& x, std::vector<double>& buffer) const;
    /// @brief Out-of-bag vote counts, [row][class].
    [[nodiscard]] std::vector<std::vector<double>> oob_vote_counts() const;

    Formula    formula_;                        ///< The parsed formula.
    DesignInfo design_;                         ///< Resolved design (columns, levels).
    RandomForestClassifierOptions options_;     ///< Options as given.
    std::size_t max_features_used_{0};          ///< Resolved \f$m\f$.

    std::vector<DecisionTreeClassifier>    trees_;          ///< The trees.
    std::vector<std::vector<int>>          in_bag_counts_;  ///< [tree][row] times drawn.
    std::vector<std::vector<std::size_t>>  tree_columns_;   ///< [tree][tree column] forest column index.
    std::vector<std::vector<std::size_t>>  tree_classes_;   ///< [tree][tree class] forest class index.
    std::vector<std::vector<std::size_t>>  training_votes_; ///< [tree][row] vote on the training rows.

    std::vector<std::string> classes_;          ///< Sorted class labels.
    std::vector<std::string> predictor_names_;  ///< Design column names.
    std::vector<double>      feature_min_;      ///< Per-predictor training minimum, for plots.
    std::vector<double>      feature_max_;      ///< Per-predictor training maximum, for plots.
    linalg::DenseMatrix<double> training_x_;    ///< Training predictors, rows x predictors.
    std::vector<std::size_t>    training_class_; ///< Class index per training row.

    std::vector<std::string> training_labels_;  ///< Class label per training row.
    std::vector<std::string> fitted_classes_;   ///< Resubstitution predictions.
    double                    oob_accuracy_{0.0}; ///< Out-of-bag accuracy.
    std::size_t               observations_{0}; ///< Training rows.
};

} // namespace datamunge::stats
