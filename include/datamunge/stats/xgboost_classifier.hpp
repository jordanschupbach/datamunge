#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/detail/xgboost_tree.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct XGBoostClassifierOptions {
    std::size_t   n_trees{100}; // boosting rounds; each round fits one tree per class
    double        learning_rate{0.3};
    std::size_t   max_depth{6};
    double        lambda{1.0};
    double        alpha{0.0};
    double        gamma{0.0};
    double        min_child_weight{1.0};
    std::size_t   min_samples_leaf{1};
    double        subsample{1.0};
    double        colsample_bytree{1.0};
    std::uint64_t seed{42};
};

struct XGBoostClassifierPrediction {
    std::vector<std::string>         class_label;
    std::vector<std::vector<double>> probability;
};

// XGBoost-style regularized multiclass gradient boosting classifier: at
// each of n_trees rounds, one tree per class is grown (via
// detail::XGBoostTree) against that class's softmax gradient/Hessian --
// gradient_ik = p_ik - y_ik (one-hot), Hessian_ik = p_ik*(1 - p_ik) (the
// standard diagonal approximation to the softmax Hessian used for
// multiclass boosting, since the exact Hessian is a dense K x K matrix per
// row). Generalizes cleanly to binary classification (K=2). Same formula +
// DataFrame constructor style as the other stats classes:
//
//   XGBoostClassifier model(iris, "Species ~ Petal.Length + Petal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// As with GBMClassifier, training_accuracy()/confusion_matrix() are
// ordinary (in-sample) training metrics; use training_deviance() to see
// the fit evolve round by round.
class XGBoostClassifier {
 public:
    XGBoostClassifier(const dstruct::DataFrame& data, const std::string& formula,
                      XGBoostClassifierOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] std::size_t                     n_trees() const { return trees_.size(); }

    // Gain-based importance summed across all trees in all rounds, normalized to sum to 1.
    [[nodiscard]] std::vector<double> feature_importance() const;

    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    [[nodiscard]] double                          training_accuracy() const;
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;
    [[nodiscard]] const std::vector<double>&  training_deviance() const { return training_deviance_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<std::string>          predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] XGBoostClassifierPrediction       predict_detail(const dstruct::DataFrame& newdata) const;

    [[nodiscard]] plot::ScatterPlot plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                         const std::string& y_feature) const;
    [[nodiscard]] plot::ScatterPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                          std::size_t grid_resolution = 60) const;
    [[nodiscard]] plot::ScatterPlot plot_training_deviance() const;

 private:
    void fit(const dstruct::DataFrame& data);
    linalg::DenseMatrix<double> score_matrix(const linalg::DenseMatrix<double>& X, std::size_t n_rows) const;

    Formula                    formula_;
    DesignInfo                 design_;
    XGBoostClassifierOptions   options_;

    std::vector<std::string> classes_;
    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_min_;
    std::vector<double>      feature_max_;
    std::vector<double>      base_scores_; // per-class log-prior, aligned with classes_

    // trees_[round][class]
    std::vector<std::vector<detail::XGBoostTree>> trees_;

    std::vector<std::string> training_labels_;
    std::vector<std::string> fitted_classes_;
    std::vector<double>      training_deviance_;
    std::size_t               observations_{0};
};

} // namespace datamunge::stats
