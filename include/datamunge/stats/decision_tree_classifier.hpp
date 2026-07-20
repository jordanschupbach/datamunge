#pragma once

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

enum class SplitCriterion { Gini, Entropy };

struct DecisionTreeClassifierOptions {
    std::size_t     max_depth{5};
    std::size_t     min_samples_split{2};
    std::size_t     min_samples_leaf{1};
    double          min_impurity_decrease{0.0};
    SplitCriterion  criterion{SplitCriterion::Gini};
    // 0 = consider every predictor at each split (ordinary CART). When
    // nonzero, each split considers a random subset of this many predictors
    // (drawn fresh per node, seeded by random_seed) — this is the mechanism
    // RandomForestClassifier uses to decorrelate its trees.
    std::size_t     max_features{0};
    std::uint64_t   random_seed{0};
};

struct DecisionTreeClassifierPrediction {
    std::vector<std::string>         class_label; // "" if predictors were missing
    std::vector<std::vector<double>> probability;  // [row][class]
};

// CART-style binary decision tree classifier, fit from a DataFrame and an
// R-style formula whose left-hand side is a categorical (string) column:
//
//   DecisionTreeClassifier model(iris, "Species ~ Petal.Length + Petal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// Recursively splits on a single predictor and threshold at each node,
// choosing the split that most reduces Gini impurity (or entropy). As with
// LDA/SVM, any intercept term in the formula is ignored and rows with a
// null predictor or null class label are dropped before fitting.
class DecisionTreeClassifier {
 public:
    DecisionTreeClassifier(const dstruct::DataFrame& data, const std::string& formula,
                           DecisionTreeClassifierOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }

    [[nodiscard]] std::size_t node_count() const { return nodes_.size(); }
    [[nodiscard]] std::size_t leaf_count() const;
    [[nodiscard]] std::size_t depth() const;

    // Gini/entropy-decrease-based importance per predictor, normalized to sum to 1.
    [[nodiscard]] std::vector<double> feature_importance() const;

    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    [[nodiscard]] double                          training_accuracy() const;
    // Rows = actual class, columns = predicted class, both ordered as classes().
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<std::string>          predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] DecisionTreeClassifierPrediction   predict_detail(const dstruct::DataFrame& newdata) const;

    // Scatter of `data` in the (x_feature, y_feature) plane, colored by
    // true class, with a distinct marker overlaid on misclassified points.
    [[nodiscard]] plot::RPlot plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                         const std::string& y_feature) const;

    // Background grid of predicted class regions in the (x_feature,
    // y_feature) plane overlaid with the training points. Only valid when
    // the model has exactly two predictors (which must be x_feature and
    // y_feature, in either order) — otherwise throws.
    [[nodiscard]] plot::RPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                          std::size_t grid_resolution = 60) const;

 private:
    struct Node {
        bool                 is_leaf{true};
        std::size_t          feature_index{0};
        double               threshold{0.0};
        std::size_t          left{0};
        std::size_t          right{0};
        std::size_t          depth{0};
        std::size_t          n_samples{0};
        double               impurity{0.0};
        std::vector<double>  class_counts; // aligned with classes_
        std::size_t          predicted_class_index{0};
    };

    void        fit(const dstruct::DataFrame& data, DecisionTreeClassifierOptions options);
    std::size_t build_node(const linalg::DenseMatrix<double>& X, const std::vector<std::size_t>& class_index,
                           std::vector<std::size_t> rows, std::size_t depth);
    double      impurity_of(const std::vector<double>& class_counts, std::size_t n) const;
    std::size_t classify_row(const std::vector<double>& x) const; // returns leaf node index
    void        dump_node(std::ostream& os, std::size_t node_index, std::size_t indent) const;
    std::vector<std::size_t> feature_candidates(std::size_t p);

    Formula    formula_;
    DesignInfo design_;
    DecisionTreeClassifierOptions options_;
    random::SplitMix64 rng_{0};

    std::vector<std::string> classes_;
    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_min_; // per-predictor training data range, for plot_decision_regions
    std::vector<double>      feature_max_;
    std::vector<Node>        nodes_; // nodes_[0] is the root

    std::vector<std::string> fitted_classes_;
    std::vector<std::string> training_labels_;
    std::size_t               observations_{0};
};

} // namespace datamunge::stats
