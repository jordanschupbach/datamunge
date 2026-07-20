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

enum class DistanceMetric { Euclidean, Manhattan };

struct KNNClassifierOptions {
    std::size_t    k{5};
    DistanceMetric metric{DistanceMetric::Euclidean};
    // Uniform (every one of the k neighbors votes equally) or distance-weighted
    // (closer neighbors vote more, weight = 1/distance) voting.
    bool weighted{false};
    // Standardize predictors (mean 0, unit variance) before computing
    // distances, so no single predictor dominates just from having a larger
    // numeric scale.
    bool standardize{true};
};

struct KNNClassifierPrediction {
    std::vector<std::string>         class_label; // "" if predictors were missing
    std::vector<std::vector<double>> vote_share;  // [row][class]
};

// K-nearest-neighbors classifier fit from a DataFrame and an R-style
// formula whose left-hand side is a categorical (string) column, using the
// same formula + DataFrame constructor style as the other stats classes:
//
//   KNNClassifier model(iris, "Species ~ Petal.Length + Petal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// KNN has no separate "training" phase beyond standardization/storage, so
// there is no honest resubstitution accuracy (a point's nearest neighbor
// is always itself, at distance zero). training_accuracy() and
// confusion_matrix() therefore report leave-one-out performance -- each
// training row is classified using every *other* training row as a
// candidate neighbor.
class KNNClassifier {
 public:
    KNNClassifier(const dstruct::DataFrame& data, const std::string& formula, KNNClassifierOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] std::size_t                     k() const { return options_.k; }

    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    // Leave-one-out accuracy (see class docs) -- not a resubstitution accuracy.
    [[nodiscard]] double                          training_accuracy() const;
    // Rows = actual class, columns = predicted class, both ordered as classes(); leave-one-out.
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<std::string>         predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] KNNClassifierPrediction          predict_detail(const dstruct::DataFrame& newdata) const;

    // Scatter of `data` in the (x_feature, y_feature) plane, colored by
    // true class, with a distinct marker overlaid on leave-one-out
    // misclassified points.
    [[nodiscard]] plot::RPlot plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                         const std::string& y_feature) const;

    // Background grid of predicted class regions in the (x_feature,
    // y_feature) plane overlaid with the training points. Only valid when
    // the model has exactly two predictors.
    [[nodiscard]] plot::RPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                          std::size_t grid_resolution = 60) const;

 private:
    void fit(const dstruct::DataFrame& data);
    // Returns vote shares (aligned with classes_) for a standardized query point, optionally excluding one
    // training row index (used for leave-one-out evaluation).
    std::vector<double> vote_shares(const std::vector<double>& query_std, std::size_t exclude_row = SIZE_MAX) const;

    Formula               formula_;
    DesignInfo            design_;
    KNNClassifierOptions  options_;

    std::vector<std::string> classes_;
    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_mean_;
    std::vector<double>      feature_scale_;
    std::vector<double>      feature_min_; // per-predictor training data range, for plot_decision_regions
    std::vector<double>      feature_max_;

    linalg::DenseMatrix<double> training_X_std_; // standardized training predictors
    std::vector<std::size_t>    training_class_index_;
    std::vector<std::string>    training_labels_;
    std::vector<std::string>    fitted_classes_; // leave-one-out predictions
    std::size_t                  observations_{0};
};

} // namespace datamunge::stats
