#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>
#include <datamunge/stats/knn_classifier.hpp> // reuses DistanceMetric

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct KNNRegressorOptions {
    std::size_t    k{5};
    DistanceMetric metric{DistanceMetric::Euclidean};
    // Uniform (every one of the k neighbors contributes equally) or
    // distance-weighted (closer neighbors contribute more, weight =
    // 1/distance) averaging.
    bool weighted{false};
    bool standardize{true};
};

// K-nearest-neighbors regressor fit from a DataFrame and an R-style
// formula whose left-hand side is numeric, using the same formula +
// DataFrame constructor style as the other stats classes:
//
//   KNNRegressor model(iris, "Petal.Length ~ Sepal.Length + Sepal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// As with KNNClassifier, a point's nearest neighbor is always itself at
// distance zero, so fitted_values()/r_squared()/rmse() report
// leave-one-out performance rather than a trivial resubstitution fit.
class KNNRegressor {
 public:
    KNNRegressor(const dstruct::DataFrame& data, const std::string& formula, KNNRegressorOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] std::size_t                     k() const { return options_.k; }

    // Leave-one-out predictions/metrics (see class docs) -- not resubstitution.
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    [[nodiscard]] double                     r_squared() const;
    [[nodiscard]] double                     rmse() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    [[nodiscard]] plot::ScatterPlot plot_predicted_vs_actual() const;
    [[nodiscard]] plot::ScatterPlot plot_residuals_vs_fitted() const;

 private:
    void   fit(const dstruct::DataFrame& data);
    double predict_one(const std::vector<double>& query_std, std::size_t exclude_row = SIZE_MAX) const;

    Formula              formula_;
    DesignInfo           design_;
    KNNRegressorOptions  options_;

    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_mean_;
    std::vector<double>      feature_scale_;

    linalg::DenseMatrix<double> training_X_std_;
    std::vector<double>         training_y_;
    std::vector<double>         fitted_; // leave-one-out predictions
    std::size_t                  observations_{0};
};

} // namespace datamunge::stats
