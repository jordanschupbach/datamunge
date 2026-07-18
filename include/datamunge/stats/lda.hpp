#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

struct LDAOptions {
    // Class prior probabilities, in the same order as LDA::classes() once
    // sorted alphabetically. Must be positive and sum to ~1. Defaults to
    // the observed class proportions when unset.
    std::optional<std::vector<double>> priors;
};

struct LDAPrediction {
    std::vector<std::string>         class_label;   // predicted class per row ("" if predictors were missing)
    std::vector<std::vector<double>> posterior;      // [row][class] posterior probabilities
    std::vector<std::vector<double>> discriminants;  // [row][LD] projected discriminant scores
};

// Classic multi-class linear discriminant analysis (Fisher/Bayes LDA,
// matching R's MASS::lda), fit from a DataFrame and an R-style formula
// whose left-hand side is a categorical (string) column, e.g.:
//
//   LDA model(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// Assumes a common (pooled) within-class covariance across all classes.
// Rows with a null predictor or null class label are dropped before
// fitting. The right-hand side supports the same term syntax as LM
// (interactions, log/sqrt/poly/I(...), categorical predictors), but any
// intercept term is ignored — classification designs never include one.
class LDA {
 public:
    LDA(const dstruct::DataFrame& data, const std::string& formula, LDAOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] std::size_t                     num_discriminants() const { return scaling_.cols(); }

    [[nodiscard]] const std::vector<double>&              priors() const { return priors_; }
    [[nodiscard]] const std::vector<std::vector<double>>& group_means() const { return group_means_; }
    [[nodiscard]] const linalg::DenseMatrix<double>&      scaling() const { return scaling_; }
    [[nodiscard]] const std::vector<double>&              proportion_of_trace() const { return proportion_of_trace_; }

    // Training-set fit diagnostics.
    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    [[nodiscard]] double                          training_accuracy() const;
    // Rows = actual class, columns = predicted class, both ordered as classes().
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<std::string> predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] LDAPrediction            predict_detail(const dstruct::DataFrame& newdata) const;

    // Scatter of the training data's first two discriminant scores
    // (LD1 vs LD2; LD1 vs a jittered baseline if there is only one),
    // colored by true class.
    [[nodiscard]] plot::ScatterPlot plot_discriminants() const;
    void                            save_discriminant_plot(const std::string& path) const;

 private:
    void          fit(const dstruct::DataFrame& data, LDAOptions options);
    LDAPrediction classify(const linalg::DenseMatrix<double>& X) const;

    Formula    formula_;
    DesignInfo design_;

    std::vector<std::string>         classes_;
    std::vector<std::string>         predictor_names_;
    std::vector<double>              priors_;
    std::vector<std::vector<double>> group_means_;
    std::vector<double>              overall_mean_;

    // Cached per-class terms of the Bayes discriminant function
    // delta_k(x) = x . precision_means_[k] - 0.5*mahalanobis_offset_[k] + log(priors_[k]).
    std::vector<std::vector<double>> precision_means_;   // Sigma_w^-1 * mu_k
    std::vector<double>              mahalanobis_offset_; // mu_k^T Sigma_w^-1 mu_k

    linalg::DenseMatrix<double> scaling_;             // p x num_discriminants
    std::vector<double>         proportion_of_trace_;

    std::vector<std::string>         fitted_classes_;
    std::vector<std::string>         training_labels_;
    std::vector<std::vector<double>> training_discriminants_; // [row][LD], for plot_discriminants()
    std::size_t                      observations_{0};
};

} // namespace datamunge::stats
