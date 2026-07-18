#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct NaiveBayesClassifierOptions {
    // Additive (Laplace/Lidstone) smoothing for categorical predictors' frequency tables.
    double laplace_smoothing{1.0};
    // Relative variance floor for numeric predictors' per-class Gaussians: the actual floor added to every
    // per-class variance is var_smoothing * (largest overall per-feature variance across the training set).
    // Matches scikit-learn's GaussianNB var_smoothing convention exactly (same default).
    double var_smoothing{1e-9};
};

struct NaiveBayesClassifierPrediction {
    std::vector<std::string>         class_label; // "" if predictors were missing
    std::vector<std::vector<double>> probability;  // [row][class]
};

// Naive Bayes classifier fit from a DataFrame and an R-style formula whose
// left-hand side is a categorical (string) column, using the same formula
// + DataFrame constructor style as the other stats classes:
//
//   NaiveBayesClassifier model(penguins, "species ~ bill_length_mm + bill_depth_mm + island");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// Each predictor is modeled independently given the class (the "naive"
// conditional-independence assumption): numeric predictors get a per-class
// Gaussian likelihood, categorical predictors get a per-class frequency
// table over their observed levels. Unlike a dummy-encoded design matrix,
// a categorical predictor's levels are modeled jointly as one variable
// (not as separate independent 0/1 features), which is what the
// independence assumption actually requires. Formula terms must be plain
// column references -- functions, interactions, and transforms aren't
// supported (they don't have a natural per-feature-independent meaning
// here) and are rejected at construction. Rows with a null predictor or
// null class label are dropped before fitting.
class NaiveBayesClassifier {
 public:
    NaiveBayesClassifier(const dstruct::DataFrame& data, const std::string& formula,
                         NaiveBayesClassifierOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] const std::vector<double>&      class_priors() const { return class_priors_; }

    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    [[nodiscard]] double                          training_accuracy() const;
    // Rows = actual class, columns = predicted class, both ordered as classes().
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<std::string>         predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] NaiveBayesClassifierPrediction   predict_detail(const dstruct::DataFrame& newdata) const;

    [[nodiscard]] plot::ScatterPlot plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                         const std::string& y_feature) const;
    // Background grid of predicted class regions; requires exactly two predictors, both numeric.
    [[nodiscard]] plot::ScatterPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                          std::size_t grid_resolution = 60) const;

 private:
    void                      fit(const dstruct::DataFrame& data);
    std::vector<std::size_t> filter_complete_rows(const dstruct::DataFrame& frame, bool require_response) const;
    std::vector<double>       log_posteriors_for_row(const dstruct::DataFrame& frame, std::size_t row) const;

    Formula                        formula_;
    DesignInfo                      design_;
    NaiveBayesClassifierOptions     options_;

    std::vector<std::string> predictor_names_;
    std::vector<bool>         predictor_is_categorical_;
    std::vector<int>          numeric_slot_;     // per predictor: index into numeric_* arrays, or -1
    std::vector<int>          categorical_slot_; // per predictor: index into categorical_* arrays, or -1

    std::vector<std::string> classes_;
    std::vector<double>      class_priors_;

    std::vector<std::vector<double>> numeric_mean_;     // [slot][class]
    std::vector<std::vector<double>> numeric_variance_; // [slot][class]
    std::vector<double>              numeric_min_;      // [slot], for plot_decision_regions
    std::vector<double>              numeric_max_;      // [slot]

    std::vector<std::vector<std::string>>         categorical_levels_;   // [slot] -> sorted levels
    std::vector<std::vector<std::vector<double>>> categorical_log_prob_; // [slot][class][level]

    std::vector<std::string> training_labels_;
    std::vector<std::string> fitted_classes_;
    std::size_t                observations_{0};
};

} // namespace datamunge::stats
