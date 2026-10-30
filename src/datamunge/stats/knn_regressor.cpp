/// @file knn_regressor.cpp
/// @brief KNNRegressor: fitting, exact leave-one-out evaluation, prediction and plots.

#include <datamunge/stats/knn_regressor.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Helpers private to this file.
namespace {

/// @brief Formats @p v with fixed @p precision ("NaN" for NaN).
/// @param v          Value.
/// @param precision  Digits after the point.
/// @return The text.
std::string format_stat(double v, int precision = 4) {
    if (std::isnan(v)) return "NaN";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

} // namespace

KNNRegressor::KNNRegressor(const dstruct::DataFrame& data, const std::string& formula, KNNRegressorOptions options)
    : formula_(formula), options_(options) {
    fit(data);
}

void KNNRegressor::fit(const dstruct::DataFrame& data) {
    if (options_.k == 0) throw std::invalid_argument("KNNRegressor: k must be at least 1");

    design_       = formula_.resolve(data, ResponseKind::Numeric, /*force_no_intercept=*/true);
    const auto dm = design_.build_matrix(data, /*require_response=*/true);
    const std::size_t n = dm.X.rows();
    const std::size_t p = dm.X.cols();

    predictor_names_ = design_.coefficient_names;
    observations_    = n;
    training_y_      = dm.y;

    scaling_        = detail::knn_scaling(dm.X, options_.standardize);
    training_X_  = dm.X;
    multipliers_ = detail::knn_multipliers(scaling_);

    fitted_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> query(p);
        for (std::size_t j = 0; j < p; ++j) query[j] = training_X_(i, j);
        fitted_[i] = predict_one(query, i, detail::knn_loo_multipliers(scaling_, training_X_, i, options_.standardize));
    }
}

double KNNRegressor::predict_one(const std::vector<double>& query, std::size_t exclude_row,
                                 const std::vector<double>& mult) const {
    double weighted_sum = 0.0, weight_total = 0.0;
    for (const auto& [d, row] : detail::knn_nearest(training_X_, query, options_.metric, options_.k,
                                                    mult.empty() ? multipliers_ : mult, exclude_row)) {
        const double w = detail::knn_weight(d, options_.weighted);
        weighted_sum += w * training_y_[row];
        weight_total += w;
    }
    return weight_total > 0.0 ? weighted_sum / weight_total : std::numeric_limits<double>::quiet_NaN();
}

std::vector<double> KNNRegressor::loo_rmse_path(std::size_t max_k) const {
    if (max_k == 0) throw std::invalid_argument("KNNRegressor::loo_rmse_path: max_k must be at least 1");
    const std::size_t n = training_X_.rows(), p = training_X_.cols();
    max_k = std::min(max_k, n - 1);
    std::vector<double> sse(max_k, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> query(p);
        for (std::size_t j = 0; j < p; ++j) query[j] = training_X_(i, j);
        const auto nearest = detail::knn_nearest(training_X_, query, options_.metric, max_k,
                                                 detail::knn_loo_multipliers(scaling_, training_X_, i,
                                                                             options_.standardize),
                                                 i);
        double sum = 0.0, total = 0.0;
        for (std::size_t k = 0; k < nearest.size(); ++k) {
            const double w = detail::knn_weight(nearest[k].first, options_.weighted);
            sum += w * training_y_[nearest[k].second];
            total += w;
            sse[k] += (training_y_[i] - sum / total) * (training_y_[i] - sum / total);
        }
    }
    for (double& s : sse) s = std::sqrt(s / static_cast<double>(n));
    return sse;
}

KNNNeighbors KNNRegressor::kneighbors(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "KNNRegressor::kneighbors");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);
    KNNNeighbors result;
    result.index.assign(newdata.nrows(), {});
    result.distance.assign(newdata.nrows(), {});
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        std::vector<double> query(dm.X.cols());
        for (std::size_t j = 0; j < dm.X.cols(); ++j) query[j] = dm.X(i, j);
        const auto row = dm.used_row_indices[i];
        for (const auto& [d, r] : detail::knn_nearest(training_X_, query, options_.metric, options_.k, multipliers_)) {
            result.index[row].push_back(r);
            result.distance[row].push_back(d);
        }
    }
    return result;
}

double KNNRegressor::r_squared() const {
    const double mean =
        std::accumulate(training_y_.begin(), training_y_.end(), 0.0) / static_cast<double>(training_y_.size());
    double tss = 0.0, rss = 0.0;
    for (std::size_t i = 0; i < training_y_.size(); ++i) {
        const double dt = training_y_[i] - mean;
        const double dr = training_y_[i] - fitted_[i];
        tss += dt * dt;
        rss += dr * dr;
    }
    return (tss > 0.0) ? (1.0 - rss / tss) : 1.0;
}

double KNNRegressor::rmse() const {
    double sse = 0.0;
    for (std::size_t i = 0; i < training_y_.size(); ++i) {
        const double d = training_y_[i] - fitted_[i];
        sse += d * d;
    }
    return std::sqrt(sse / static_cast<double>(training_y_.size()));
}

std::string KNNRegressor::summary() const {
    std::ostringstream out;
    out << "Call:\nknn(formula = " << formula_.text() << ", k = " << options_.k << ", metric = "
        << (options_.metric == DistanceMetric::Euclidean ? "euclidean" : "manhattan")
        << ", weighted = " << (options_.weighted ? "true" : "false") << ")\n\n";

    out << "Leave-one-out R-squared: " << format_stat(r_squared(), 4) << "\n";
    out << "Leave-one-out RMSE: " << format_stat(rmse(), 4) << "  (" << observations_ << " observations)\n";

    return out.str();
}

void KNNRegressor::print_summary(std::ostream& os) const { os << summary(); }

void KNNRegressor::print_summary() const { print_summary(std::cout); }

std::vector<double> KNNRegressor::predict(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "KNNRegressor::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);

    std::vector<double> result(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        std::vector<double> query(dm.X.cols());
        for (std::size_t j = 0; j < dm.X.cols(); ++j) query[j] = dm.X(i, j);
        result[dm.used_row_indices[i]] = predict_one(query);
    }
    return result;
}

graphics::RPlot KNNRegressor::plot_predicted_vs_actual() const {
    auto plot = graphics::RPlot::create();
    plot.points(training_y_, fitted_, "predictions");
    plot.title("Predicted vs Actual (leave-one-out)").x_label("Actual").y_label("Predicted");
    return plot;
}

graphics::RPlot KNNRegressor::plot_residuals_vs_fitted() const {
    std::vector<double> residuals(training_y_.size());
    for (std::size_t i = 0; i < training_y_.size(); ++i) residuals[i] = training_y_[i] - fitted_[i];

    auto plot = graphics::RPlot::create();
    plot.points(fitted_, residuals, "residuals");
    plot.title("Residuals vs Fitted (leave-one-out)").x_label("Fitted values").y_label("Residuals");
    return plot;
}

} // namespace datamunge::stats
