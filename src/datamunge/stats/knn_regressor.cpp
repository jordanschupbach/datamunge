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

namespace {

std::string format_stat(double v, int precision = 4) {
    if (std::isnan(v)) return "NaN";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

} // namespace

KNNRegressor::KNNRegressor(const dstruct::DataFrame& data, const std::string& formula, KNNRegressorOptions options)
    : formula_(formula) {
    options_ = options;
    fit(data);
}

void KNNRegressor::fit(const dstruct::DataFrame& data) {
    if (options_.k == 0) throw std::invalid_argument("KNNRegressor: k must be at least 1");

    design_       = formula_.resolve(data, ResponseKind::Numeric, /*force_no_intercept=*/true);
    const auto dm = design_.build_matrix(data, /*require_response=*/true);
    const std::size_t n = dm.X.rows();
    const std::size_t p = dm.X.cols();

    predictor_names_ = design_.coefficient_names;
    observations_     = n;
    training_y_       = dm.y;

    feature_mean_.assign(p, 0.0);
    feature_scale_.assign(p, 1.0);
    if (options_.standardize) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum += dm.X(i, j);
            feature_mean_[j] = sum / static_cast<double>(n);
        }
        for (std::size_t j = 0; j < p; ++j) {
            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double d = dm.X(i, j) - feature_mean_[j];
                ss += d * d;
            }
            const double scale = std::sqrt(ss / static_cast<double>(n));
            feature_scale_[j]  = (scale < 1e-12) ? 1.0 : scale;
        }
    }

    training_X_std_ = linalg::DenseMatrix<double>(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) training_X_std_(i, j) = (dm.X(i, j) - feature_mean_[j]) / feature_scale_[j];

    fitted_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> query(p);
        for (std::size_t j = 0; j < p; ++j) query[j] = training_X_std_(i, j);
        fitted_[i] = predict_one(query, /*exclude_row=*/i);
    }
}

double KNNRegressor::predict_one(const std::vector<double>& query_std, std::size_t exclude_row) const {
    const std::size_t n = training_X_std_.rows();
    const std::size_t p = training_X_std_.cols();

    std::vector<std::pair<double, std::size_t>> distances;
    distances.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (i == exclude_row) continue;
        double d = 0.0;
        if (options_.metric == DistanceMetric::Euclidean) {
            for (std::size_t j = 0; j < p; ++j) {
                const double diff = training_X_std_(i, j) - query_std[j];
                d += diff * diff;
            }
            d = std::sqrt(d);
        } else {
            for (std::size_t j = 0; j < p; ++j) d += std::fabs(training_X_std_(i, j) - query_std[j]);
        }
        distances.emplace_back(d, i);
    }

    const std::size_t kk = std::min(options_.k, distances.size());
    std::partial_sort(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(kk), distances.end());

    double weighted_sum = 0.0;
    double weight_total  = 0.0;
    for (std::size_t idx = 0; idx < kk; ++idx) {
        const double      d   = distances[idx].first;
        const std::size_t row = distances[idx].second;
        const double       w   = options_.weighted ? 1.0 / (d + 1e-12) : 1.0;
        weighted_sum += w * training_y_[row];
        weight_total += w;
    }
    return (weight_total > 0.0) ? (weighted_sum / weight_total) : std::numeric_limits<double>::quiet_NaN();
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
        for (std::size_t j = 0; j < dm.X.cols(); ++j) query[j] = (dm.X(i, j) - feature_mean_[j]) / feature_scale_[j];
        result[dm.used_row_indices[i]] = predict_one(query);
    }
    return result;
}

plot::ScatterPlot KNNRegressor::plot_predicted_vs_actual() const {
    auto plot = plot::ScatterPlot::create();
    plot.points(training_y_, fitted_, "predictions");
    plot.title("Predicted vs Actual (leave-one-out)").x_label("Actual").y_label("Predicted");
    return plot;
}

plot::ScatterPlot KNNRegressor::plot_residuals_vs_fitted() const {
    std::vector<double> residuals(training_y_.size());
    for (std::size_t i = 0; i < training_y_.size(); ++i) residuals[i] = training_y_[i] - fitted_[i];

    auto plot = plot::ScatterPlot::create();
    plot.points(fitted_, residuals, "residuals");
    plot.title("Residuals vs Fitted (leave-one-out)").x_label("Fitted values").y_label("Residuals");
    return plot;
}

} // namespace datamunge::stats
