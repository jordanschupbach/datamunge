#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/decision_tree_classifier.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct RandomForestClassifierOptions {
    std::size_t    n_trees{100};
    std::size_t    max_depth{10};
    std::size_t    min_samples_split{2};
    std::size_t    min_samples_leaf{1};
    // 0 = auto: floor(sqrt(number of predictors)), at least 1 (the usual
    // classification default). Each tree considers this many randomly
    // chosen predictors at every split.
    std::size_t    max_features{0};
    SplitCriterion criterion{SplitCriterion::Gini};
    // Draw each tree's training sample with replacement (true, the usual
    // "bagging" random forest) or as a without-replacement subsample
    // (false).
    bool           bootstrap{true};
    // Fraction of the training rows drawn per tree.
    double         sample_fraction{1.0};
    std::uint64_t  seed{42};
};

struct RandomForestClassifierPrediction {
    std::vector<std::string>         class_label; // "" if predictors were missing
    std::vector<std::vector<double>> vote_share;  // [row][class], fraction of trees voting each class
};

// Bagged ensemble of DecisionTreeClassifier trees, each fit on a bootstrap
// resample of the training rows and restricted to a random subset of
// predictors at every split (the standard random forest recipe). Uses the
// same formula + DataFrame constructor style as the other stats classes:
//
//   RandomForestClassifier model(iris, "Species ~ Petal.Length + Petal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// Predictions are the majority vote across trees. Out-of-bag rows (rows a
// given tree never trained on) give a built-in accuracy estimate via
// oob_accuracy() without needing a held-out test set.
class RandomForestClassifier {
 public:
    RandomForestClassifier(const dstruct::DataFrame& data, const std::string& formula,
                           RandomForestClassifierOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] std::size_t                     n_trees() const { return trees_.size(); }
    [[nodiscard]] std::size_t                     max_features_used() const { return max_features_used_; }

    // Per-tree Gini/entropy-decrease importance, averaged across trees and
    // normalized to sum to 1.
    [[nodiscard]] std::vector<double> feature_importance() const;

    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    [[nodiscard]] double                          training_accuracy() const;
    // Out-of-bag accuracy: for each row, majority-vote only among the trees
    // that did not train on it. A near-unbiased estimate of test accuracy.
    [[nodiscard]] double                          oob_accuracy() const { return oob_accuracy_; }
    // Rows = actual class, columns = predicted class, both ordered as classes().
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<std::string>           predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] RandomForestClassifierPrediction    predict_detail(const dstruct::DataFrame& newdata) const;

    // Scatter of `data` in the (x_feature, y_feature) plane, colored by
    // true class, with a distinct marker overlaid on misclassified points.
    [[nodiscard]] plot::RPlot plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                         const std::string& y_feature) const;

    // Background grid of majority-vote predicted class regions in the
    // (x_feature, y_feature) plane overlaid with the training points. Only
    // valid when the model has exactly two predictors.
    [[nodiscard]] plot::RPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                          std::size_t grid_resolution = 60) const;

 private:
    void fit(const dstruct::DataFrame& data, RandomForestClassifierOptions options);

    Formula    formula_;
    DesignInfo design_;
    RandomForestClassifierOptions options_;
    std::size_t max_features_used_{0};

    std::vector<DecisionTreeClassifier> trees_;
    std::vector<std::vector<bool>>      in_bag_; // [tree][row] -- true if row was sampled into that tree's training set

    std::vector<std::string> classes_;
    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_min_;
    std::vector<double>      feature_max_;

    std::vector<std::string> training_labels_;
    std::vector<std::string> fitted_classes_;
    double                    oob_accuracy_{0.0};
    std::size_t               observations_{0};
};

} // namespace datamunge::stats
