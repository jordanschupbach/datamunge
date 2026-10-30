#pragma once
/// @file decision_tree_classifier.hpp
/// @brief CART classification trees: recursive binary splits chosen by Gini or entropy decrease,
///        the building block of RandomForestClassifier.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Impurity a classification tree minimises when choosing a split.
enum class SplitCriterion {
    Gini,    ///< \f$1-\sum_k p_k^2\f$.
    Entropy  ///< \f$-\sum_k p_k\log_2p_k\f$.
};

/// @brief Options for DecisionTreeClassifier.
struct DecisionTreeClassifierOptions {
    std::size_t     max_depth{5};                 ///< Depth limit (the root has depth 0; at least 1).
    std::size_t     min_samples_split{2};         ///< A node with fewer rows is not split.
    std::size_t     min_samples_leaf{1};          ///< Each child of a split keeps at least this many rows.
    double          min_impurity_decrease{0.0};   ///< A split must lower the impurity by more than this.
    SplitCriterion  criterion{SplitCriterion::Gini}; ///< Impurity used to choose splits.
    /** @brief Predictors tried at each node: 0 means all (ordinary CART); otherwise a random subset
        of this many, drawn afresh at every node from random_seed. RandomForestClassifier uses
        this to decorrelate its trees. */
    std::size_t     max_features{0};
    std::uint64_t   random_seed{0};               ///< Seed for the predictor subsets.
};

/// @brief Predictions of a DecisionTreeClassifier for a data frame.
struct DecisionTreeClassifierPrediction {
    std::vector<std::string>         class_label; ///< Leaf majority class per row; "" if a predictor was missing.
    std::vector<std::vector<double>> probability; ///< [row][class] class shares in the row's leaf.
};

/// @brief CART binary classification tree, fit from a data frame and an R-style formula whose
///        response is a string column.
///
/// Each node is split on one predictor at a threshold \f$x_j\le c\f$, choosing the predictor and the
/// midpoint \f$c\f$ between consecutive distinct values that most reduce the weighted impurity of the
/// children. Each predictor's rows are sorted once per node and swept from left to right, so a
/// node of \f$n\f$ rows costs \f$O(pn\log n)\f$. Ties between equally good splits go to the predictor that
/// comes first and the smaller threshold. Growing stops at max_depth, at pure nodes, at nodes
/// smaller than min_samples_split, or when no split decreases the impurity by more than
/// min_impurity_decrease. Any intercept in the formula is ignored, and rows with a missing
/// predictor or response are dropped.
///
/// @par Example
/// @code
/// const dm::stats::DecisionTreeClassifier tree(dm::datasets::iris(), "Species ~ .", {.max_depth = 30});
/// std::print("{} nodes, depth {}, training accuracy {:.2f}\n", tree.node_count(), tree.depth(),
///            tree.training_accuracy()); // 17 nodes, depth 5, training accuracy 1.00
/// @endcode
///
/// @par References
/// - L. Breiman, J. H. Friedman, R. A. Olshen and C. J. Stone (1984). *Classification and
///   Regression Trees*. Wadsworth.
class DecisionTreeClassifier {
 public:
    /// @brief Grows the tree.
    /// @param data     Training data.
    /// @param formula  Formula `class ~ predictors`; the response must be a string column.
    /// @param options  Tree options.
    /// @throws std::invalid_argument  If the response has fewer than two classes, max_depth is 0,
    ///         or the formula does not resolve.
    DecisionTreeClassifier(const dstruct::DataFrame& data, const std::string& formula,
                           DecisionTreeClassifierOptions options = {});

    /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
    ///
    /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
    /// converts to Formula implicitly, so `DecisionTreeClassifier("y ~ x", data)` also works.
    /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
    /// @param data  Data frame holding every column the formula names.
    /// @param options  Same as in the `(data, formula, ...)` constructor.
    /// @throws std::invalid_argument  As the `(data, formula, ...)` constructor.
    DecisionTreeClassifier(const Formula& formula, const dstruct::DataFrame& data, DecisionTreeClassifierOptions options = {})
        : DecisionTreeClassifier(data, formula.text(), std::move(options)) {}

    /// @brief The formula text the tree was fit with.
    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    /// @brief The classes, sorted; class indices refer to this order.
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    /// @brief The design columns (predictors, with categorical ones expanded to dummies).
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    /// @brief Training rows used (after dropping rows with missing values).
    [[nodiscard]] std::size_t                     observations() const { return observations_; }

    /// @brief Number of nodes, leaves included.
    [[nodiscard]] std::size_t node_count() const { return nodes_.size(); }
    /// @brief Number of leaves.
    [[nodiscard]] std::size_t leaf_count() const;
    /// @brief Depth of the deepest leaf (a lone root has depth 0).
    [[nodiscard]] std::size_t depth() const;

    /// @brief Impurity-decrease importance per predictor: the sum over the predictor's splits of
    ///        \f$(n_{\text{node}}/n)\Delta\f$, normalised to sum to 1.
    /// @return One value per predictor_names() entry.
    [[nodiscard]] std::vector<double> feature_importance() const;

    /// @brief Resubstitution predictions: the leaf class of each training row.
    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    /// @brief Share of training rows whose leaf class is correct.
    [[nodiscard]] double                          training_accuracy() const;
    /// @brief Resubstitution confusion matrix: rows are the actual class, columns the predicted
    ///        class, both in classes() order.
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;

    /// @brief Summary in the style of R's `print(tree)`: call, classes, the tree as an indented
    ///        list of splits, importance and training accuracy.
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to @p os.
    /// @param os  Output stream.
    void                      print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    void                      print_summary() const;

    /// @brief Leaf class per row of @p newdata ("" if a predictor is missing).
    /// @param newdata  Data frame with the predictor columns.
    /// @return One class label per row.
    /// @throws std::runtime_error  If a categorical predictor has an unseen level.
    [[nodiscard]] std::vector<std::string>          predict(const dstruct::DataFrame& newdata) const;
    /// @brief Leaf class and leaf class shares per row of @p newdata.
    /// @param newdata  Data frame with the predictor columns.
    /// @return The predictions.
    /// @throws std::runtime_error  If a categorical predictor has an unseen level.
    [[nodiscard]] DecisionTreeClassifierPrediction   predict_detail(const dstruct::DataFrame& newdata) const;

    /// @brief Predicts one row given as numbers, without building a data frame.
    ///
    /// Fast path for ensembles and plots that evaluate the tree at many points.
    /// @param x  The predictor values, ordered as predictor_names() (dummy columns included).
    /// @return Index of the predicted class in classes().
    /// @throws std::invalid_argument  If @p x does not have one value per predictor.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::DecisionTreeClassifier tree(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    /// std::print("{}\n", tree.classes()[tree.predict_row({1.4, 0.2})]); // setosa
    /// @endcode
    [[nodiscard]] std::size_t predict_row(const std::vector<double>& x) const;

    /// @brief The tree as a table, one row per node in depth-first order (node 0 is the root).
    ///
    /// Columns: `node`, `depth`, `feature` ("" for a leaf), `threshold` (null for a leaf; rows with
    /// a value at most the threshold go left), `left` and `right` (child node ids, -1 for a
    /// leaf), `n` (training rows reaching the node), `impurity`, and `class` (majority class).
    /// @return The node table.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::DecisionTreeClassifier tree(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width",
    ///                                              {.max_depth = 2});
    /// std::print("{}\n", tree.node_frame().to_md());
    /// @endcode
    [[nodiscard]] dstruct::DataFrame node_frame() const;

    /// @brief Scatter of @p data in the (x_feature, y_feature) plane, coloured by true class, with
    ///        a black marker on rows the tree misclassifies.
    /// @param data       Data frame with the response and both features.
    /// @param x_feature  Column on the x axis.
    /// @param y_feature  Column on the y axis.
    /// @return The plot.
    [[nodiscard]] graphics::RPlot plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                      const std::string& y_feature) const;

    /// @brief The predicted class over a grid of the tree's two predictors, drawn as pale points.
    /// @param x_feature        Predictor on the x axis.
    /// @param y_feature        Predictor on the y axis.
    /// @param grid_resolution  Grid points per axis.
    /// @return The plot.
    /// @throws std::invalid_argument  If the model does not have exactly these two predictors.
    [[nodiscard]] graphics::RPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                        std::size_t grid_resolution = 60) const;

 private:
    /// @brief One node of the tree.
    struct Node {
        bool                 is_leaf{true};             ///< True for a leaf.
        std::size_t          feature_index{0};          ///< Split predictor (internal nodes).
        double               threshold{0.0};            ///< Split threshold; \f$x\le c\f$ goes left.
        std::size_t          left{0};                   ///< Left child node index.
        std::size_t          right{0};                  ///< Right child node index.
        std::size_t          depth{0};                  ///< Depth (root 0).
        std::size_t          n_samples{0};              ///< Training rows reaching the node.
        double               impurity{0.0};             ///< Impurity of those rows.
        std::vector<double>  class_counts;              ///< Rows per class, in classes() order.
        std::size_t          predicted_class_index{0};  ///< Majority class.
    };

    /// @brief Resolves the formula and grows the tree.
    /// @param data     Training data.
    /// @param options  Tree options.
    void        fit(const dstruct::DataFrame& data, DecisionTreeClassifierOptions options);
    /// @brief Grows the subtree for @p rows and returns its root's index.
    /// @param X            Training predictors.
    /// @param class_index  Class index of every training row.
    /// @param rows         Rows reaching this node.
    /// @param depth        Depth of this node.
    /// @return Index of the new node in nodes_.
    std::size_t build_node(const linalg::DenseMatrix<double>& X, const std::vector<std::size_t>& class_index,
                           std::vector<std::size_t> rows, std::size_t depth);
    /// @brief Impurity of a node with @p class_counts over @p n rows.
    /// @param class_counts  Rows per class.
    /// @param n             Total rows.
    /// @return Gini or entropy impurity.
    double      impurity_of(const std::vector<double>& class_counts, std::size_t n) const;
    /// @brief The leaf reached by @p x.
    /// @param x  Predictor values in predictor_names() order.
    /// @return Leaf node index.
    std::size_t classify_row(const std::vector<double>& x) const;
    /// @brief Writes the subtree at @p node_index as indented text.
    /// @param os          Output stream.
    /// @param node_index  Subtree root.
    /// @param indent      Indentation level.
    void        dump_node(std::ostream& os, std::size_t node_index, std::size_t indent) const;
    /// @brief The predictors to try at a node: all, or a random max_features of them.
    /// @param p  Number of predictors.
    /// @return Predictor indices.
    std::vector<std::size_t> feature_candidates(std::size_t p);

    Formula    formula_;                        ///< The parsed formula.
    DesignInfo design_;                         ///< Resolved design (columns, levels).
    DecisionTreeClassifierOptions options_;     ///< Options as given.
    random::SplitMix64 rng_{0};                 ///< Generator for predictor subsets.

    std::vector<std::string> classes_;          ///< Sorted class labels.
    std::vector<std::string> predictor_names_;  ///< Design column names.
    std::vector<double>      feature_min_;      ///< Per-predictor training minimum, for plots.
    std::vector<double>      feature_max_;      ///< Per-predictor training maximum, for plots.
    std::vector<Node>        nodes_;            ///< The nodes; nodes_[0] is the root.

    std::vector<std::string> fitted_classes_;   ///< Resubstitution predictions.
    std::vector<std::string> training_labels_;  ///< Class label per training row.
    std::size_t               observations_{0}; ///< Training rows.
};

} // namespace datamunge::stats
