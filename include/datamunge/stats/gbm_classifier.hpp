#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/decision_tree_regressor.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct GBMClassifierOptions {
    std::size_t   n_trees{100}; // boosting rounds; each round fits one tree per class
    double        learning_rate{0.1};
    std::size_t   max_depth{3};
    std::size_t   min_samples_split{2};
    std::size_t   min_samples_leaf{1};
    double        subsample{1.0};
    std::uint64_t seed{42};
};

struct GBMClassifierPrediction {
    std::vector<std::string>         class_label; // "" if predictors were missing
    std::vector<std::vector<double>> probability;  // [row][class], softmax over class scores
};

// Multiclass gradient boosting classifier: at each of n_trees rounds, one
// shallow regression tree per class is fit to that class's pseudo-residual
// (one-hot indicator minus current softmax probability) -- functional
// gradient descent on multinomial deviance. Generalizes cleanly to binary
// classification (K=2). Same formula + DataFrame constructor style as the
// other stats classes:
//
//   GBMClassifier model(iris, "Species ~ Petal.Length + Petal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// As with GBMRegressor, training_accuracy()/confusion_matrix() are
// ordinary (in-sample) training metrics; use training_deviance() to see
// how the fit evolved round by round.
class GBMClassifier {
 public:
    GBMClassifier(const dstruct::DataFrame& data, const std::string& formula, GBMClassifierOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] std::size_t                     n_trees() const { return trees_.size(); }

    // Per-class-tree SSE-decrease importance, averaged across every tree in every round, normalized to sum to 1.
    [[nodiscard]] std::vector<double> feature_importance() const;

    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    [[nodiscard]] double                          training_accuracy() const;
    // Rows = actual class, columns = predicted class, both ordered as classes().
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;
    // Multinomial deviance (mean negative log-likelihood) on the training set after each boosting round.
    [[nodiscard]] const std::vector<double>& training_deviance() const { return training_deviance_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<std::string>          predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] GBMClassifierPrediction           predict_detail(const dstruct::DataFrame& newdata) const;

    [[nodiscard]] plot::ScatterPlot plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                         const std::string& y_feature) const;
    [[nodiscard]] plot::ScatterPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                          std::size_t grid_resolution = 60) const;
    [[nodiscard]] plot::ScatterPlot plot_training_deviance() const;

 private:
    void fit(const dstruct::DataFrame& data);
    // Softmax class scores for a batch of rows already expressed as a synthetic x0..x{p-1} predictor frame.
    linalg::DenseMatrix<double> score_frame(const dstruct::DataFrame& predictor_frame, std::size_t n_rows) const;

    Formula               formula_;
    DesignInfo            design_;
    GBMClassifierOptions  options_;

    std::vector<std::string> classes_;
    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_min_;
    std::vector<double>      feature_max_;
    std::vector<double>      initial_scores_; // per-class log-prior, aligned with classes_

    // trees_[round][class]
    std::vector<std::vector<DecisionTreeRegressor>> trees_;

    std::vector<std::string> training_labels_;
    std::vector<std::string> fitted_classes_;
    std::vector<double>      training_deviance_;
    std::size_t               observations_{0};
};

} // namespace datamunge::stats
