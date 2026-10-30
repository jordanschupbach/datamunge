#pragma once

/// @file knn_regressor.hpp
/// @brief k-nearest-neighbour regression: the (optionally inverse-distance weighted) mean
///        response of the k training points closest to the query, with exact leave-one-out
///        evaluation.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/detail/knn_common.hpp>
#include <datamunge/stats/formula.hpp>
#include <datamunge/stats/knn_classifier.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Settings for KNNRegressor.
struct KNNRegressorOptions {
    /// @brief Number of neighbours averaged; at least 1.
    std::size_t k{5};
    /// @brief Distance between standardised predictor vectors.
    DistanceMetric metric{DistanceMetric::Euclidean};
    /// @brief Plain mean (false) or mean weighted by \f$1/d\f$ (true).
    bool weighted{false};
    /// @brief Rescale each predictor to mean 0 and standard deviation 1 (divisor \f$n\f$)
    ///        before measuring distances.
    bool standardize{true};
};

/// @brief k-nearest-neighbour regressor from a formula `y ~ x1 + x2 + ...`.
///
/// The prediction at \f$x\f$ is \f$\hat f(x)=\frac1k\sum_{j\in N_k(x)}y_j\f$ (or the
/// \f$1/d\f$-weighted mean), a step function of \f$x\f$. Ties in distance are broken by training-row
/// order; differences are taken on the raw values before scaling, so raw ties stay exact. fitted_values(), r_squared() and rmse() are leave-one-out: each row is predicted from
/// the other \f$n-1\f$ rows, with the standardisation recomputed without it.
///
/// @par Example
/// @code
/// const dm::stats::KNNRegressor knn(dm::datasets::iris(),
///                                   "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
/// std::print("{:.4f}\n", knn.r_squared());  // leave-one-out R^2, about 0.96
/// @endcode
///
/// @par References
/// - scikit-learn `KNeighborsRegressor`:
///   https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
/// - T. Hastie, R. Tibshirani and J. Friedman (2009). *The Elements of Statistical Learning*,
///   2nd ed., section 2.3.2 and 7.3. Springer.
class KNNRegressor {
 public:
    /// @brief Fits the regressor.
    /// @param data     Data frame holding the numeric response and the predictors.
    /// @param formula  `y ~ predictors`; rows with a missing value are dropped. No intercept column is made.
    /// @param options  k, metric, weighting and standardisation.
    /// @throws std::invalid_argument  If k is 0.
    KNNRegressor(const dstruct::DataFrame& data, const std::string& formula, KNNRegressorOptions options = {});

    /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
    ///
    /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
    /// converts to Formula implicitly, so `KNNRegressor("y ~ x", data)` also works.
    /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
    /// @param data  Data frame holding every column the formula names.
    /// @param options  Same as in the `(data, formula, ...)` constructor.
    /// @throws std::invalid_argument  As the `(data, formula, ...)` constructor.
    KNNRegressor(const Formula& formula, const dstruct::DataFrame& data, KNNRegressorOptions options = {})
        : KNNRegressor(data, formula.text(), std::move(options)) {}

    /// @brief The formula the model was fitted with.
    /// @return Formula text.
    [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
    /// @brief Names of the predictor columns.
    /// @return Predictor names.
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    /// @brief Number of training rows used.
    /// @return \f$n\f$.
    [[nodiscard]] std::size_t observations() const { return observations_; }
    /// @brief Number of neighbours averaged.
    /// @return k.
    [[nodiscard]] std::size_t k() const { return options_.k; }
    /// @brief The options the model was fitted with.
    /// @return Options.
    [[nodiscard]] const KNNRegressorOptions& options() const { return options_; }

    /// @brief Leave-one-out prediction of each training row.
    /// @return \f$n\f$ predictions.
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    /// @brief Observed response of each training row.
    /// @return \f$n\f$ values.
    [[nodiscard]] const std::vector<double>& training_response() const { return training_y_; }
    /// @brief Leave-one-out \f$R^2=1-\sum(y_i-\hat y_{-i})^2/\sum(y_i-\bar y)^2\f$.
    /// @return \f$R^2\f$ (can be negative).
    [[nodiscard]] double r_squared() const;
    /// @brief Leave-one-out root mean squared error.
    /// @return RMSE.
    [[nodiscard]] double rmse() const;
    /// @brief Leave-one-out RMSE for every \f$k=1,\dots,K\f$ at once, with this model's metric,
    ///        weighting and standardisation.
    /// @param max_k  \f$K\f$; reduced to \f$n-1\f$ if larger.
    /// @return RMSEs; element \f$k-1\f$ is the RMSE with \f$k\f$ neighbours.
    /// @throws std::invalid_argument  If @p max_k is 0.
    [[nodiscard]] std::vector<double> loo_rmse_path(std::size_t max_k) const;

    /// @brief The k nearest training rows of each row of @p newdata.
    /// @param newdata  Rows to query; needs the predictor columns.
    /// @return Training-row positions and distances, nearest first (empty where a predictor is
    ///         missing). Distances are on the standardised scale when `standardize` is set.
    [[nodiscard]] KNNNeighbors kneighbors(const dstruct::DataFrame& newdata) const;

    /// @brief A short text summary (formula, k, metric, leave-one-out \f$R^2\f$ and RMSE).
    /// @return Summary text.
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to @p os.
    /// @param os  Output stream.
    void print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    void print_summary() const;

    /// @brief Predictions for new rows.
    /// @param newdata  Rows to predict.
    /// @return One prediction per row (NaN where a predictor is missing).
    /// @throws std::invalid_argument  If a factor predictor has a level unseen in training.
    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    /// @brief Leave-one-out predictions against the observed responses.
    /// @return The plot.
    [[nodiscard]] graphics::RPlot plot_predicted_vs_actual() const;
    /// @brief Leave-one-out residuals against the leave-one-out predictions.
    /// @return The plot.
    [[nodiscard]] graphics::RPlot plot_residuals_vs_fitted() const;

 private:
    /// @brief Builds the design, the scaling and the leave-one-out predictions.
    /// @param data  Training data.
    void fit(const dstruct::DataFrame& data);
    /// @brief The (weighted) mean response of the k nearest training rows to a raw query point.
    /// @param query        Raw predictor values.
    /// @param exclude_row  Training row to skip (leave-one-out), or `SIZE_MAX`.
    /// @param mult         Per-predictor multipliers; empty for the fitted \f$1/s_j\f$.
    /// @return The prediction (NaN if there are no neighbours).
    double predict_one(const std::vector<double>& query, std::size_t exclude_row = SIZE_MAX,
                       const std::vector<double>& mult = {}) const;

    Formula             formula_;  ///< The model formula.
    DesignInfo          design_;   ///< Resolved predictor columns.
    KNNRegressorOptions options_;  ///< Fitting options.

    std::vector<std::string> predictor_names_;  ///< Predictor column names.
    detail::KnnScaling       scaling_;          ///< Means and scales of the predictors.

    linalg::DenseMatrix<double> training_X_;       ///< Raw training predictors.
    std::vector<double>         multipliers_;      ///< \f$1/s_j\f$ per predictor.
    std::vector<double>         training_y_;       ///< Training responses.
    std::vector<double>         fitted_;           ///< Leave-one-out predictions.
    std::size_t                 observations_{0};  ///< Number of training rows.
};

} // namespace datamunge::stats
