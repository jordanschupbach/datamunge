#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

enum class SVMKernel { Linear, Polynomial, Radial, Sigmoid };

struct SVMOptions {
    SVMKernel   kernel{SVMKernel::Radial};
    double      cost{1.0};           // C: box constraint on the dual alphas
    double      gamma{-1.0};         // <= 0 means "auto" = 1 / number of predictors
    double      coef0{0.0};          // used by Polynomial and Sigmoid kernels
    int         degree{3};           // used by the Polynomial kernel
    double      tolerance{1e-3};     // SMO KKT-violation tolerance
    std::size_t max_passes{50};      // consecutive no-change sweeps before declaring convergence
    bool        scale{true};         // standardize numeric predictors (zero mean, unit variance)
};

struct SVMPrediction {
    std::vector<std::string>         class_label; // predicted class per row ("" if predictors were missing)
    std::vector<std::vector<double>> votes;        // [row][class] one-vs-one vote counts
};

// Multi-class support vector machine classifier (soft-margin, one-vs-one),
// fit from a DataFrame and an R-style formula whose left-hand side is a
// categorical (string) column — matching R's e1071::svm() in spirit:
//
//   SVM model(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// Trains one binary classifier per pair of classes via a simplified
// Sequential Minimal Optimization (SMO) solver and classifies new points by
// majority vote across all pairs. Numeric predictors are standardized
// internally by default (SVMOptions::scale). As with LDA, any intercept
// term in the formula is ignored and rows with a null predictor or null
// class label are dropped before fitting.
class SVM {
 public:
    SVM(const dstruct::DataFrame& data, const std::string& formula, SVMOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }

    [[nodiscard]] std::size_t                num_support_vectors() const;
    [[nodiscard]] std::vector<std::size_t>    support_vectors_per_class() const; // aligned with classes()

    // Training-set fit diagnostics.
    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    [[nodiscard]] double                          training_accuracy() const;
    // Rows = actual class, columns = predicted class, both ordered as classes().
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<std::string> predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] SVMPrediction            predict_detail(const dstruct::DataFrame& newdata) const;

 private:
    struct PairModel {
        std::size_t               class_i{0};
        std::size_t               class_j{0}; // votes for class_j when the decision value is negative
        std::vector<std::size_t>  support_indices; // row indices into scaled_training_X_
        std::vector<double>       support_alpha_y;  // alpha_k * y_k for each support vector
        double                    bias{0.0};
    };

    void           fit(const dstruct::DataFrame& data, SVMOptions options);
    double         kernel(const std::vector<double>& a, const std::vector<double>& b) const;
    SVMPrediction  classify(const linalg::DenseMatrix<double>& X) const;
    std::vector<double> scale_row(const std::vector<double>& raw) const;

    Formula    formula_;
    DesignInfo design_;
    SVMOptions options_;
    double     effective_gamma_{1.0};

    std::vector<std::string> classes_;
    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_mean_;
    std::vector<double>      feature_scale_;

    linalg::DenseMatrix<double> scaled_training_X_;
    std::vector<std::string>    training_labels_;
    std::vector<PairModel>      pair_models_;

    std::vector<std::string> fitted_classes_;
    std::size_t               observations_{0};
};

} // namespace datamunge::stats
