#include <datamunge/stats/kernel_regression.hpp>

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

const char* kernel_name(KernelRegressionKernel kernel) {
    switch (kernel) {
        case KernelRegressionKernel::Gaussian: return "gaussian";
        case KernelRegressionKernel::Epanechnikov: return "epanechnikov";
        case KernelRegressionKernel::Uniform: return "uniform";
        case KernelRegressionKernel::Triangular: return "triangular";
    }
    return "unknown";
}

double kernel_weight(double u, KernelRegressionKernel kernel) {
    switch (kernel) {
        case KernelRegressionKernel::Gaussian: return std::exp(-0.5 * u * u);
        case KernelRegressionKernel::Epanechnikov: return (std::fabs(u) <= 1.0) ? 0.75 * (1.0 - u * u) : 0.0;
        case KernelRegressionKernel::Uniform: return (std::fabs(u) <= 1.0) ? 1.0 : 0.0;
        case KernelRegressionKernel::Triangular: return (std::fabs(u) <= 1.0) ? (1.0 - std::fabs(u)) : 0.0;
    }
    return 0.0;
}

} // namespace

KernelRegression::KernelRegression(const dstruct::DataFrame& data, const std::string& formula,
                                   KernelRegressionOptions options)
    : formula_(formula) {
    options_ = options;
    fit(data);
}

void KernelRegression::fit(const dstruct::DataFrame& data) {
    if (options_.n_bandwidth == 0) throw std::invalid_argument("KernelRegression: n_bandwidth must be at least 1");
    if (options_.bandwidth == 0.0) throw std::invalid_argument("KernelRegression: bandwidth must not be zero");

    design_       = formula_.resolve(data, ResponseKind::Numeric, /*force_no_intercept=*/true);
    const auto dm = design_.build_matrix(data, /*require_response=*/true);
    const std::size_t n = dm.X.rows();
    const std::size_t p = dm.X.cols();
    if (p == 0) throw std::invalid_argument("KernelRegression: at least one predictor is required");

    predictor_names_ = design_.coefficient_names;
    observations_     = n;
    training_y_       = dm.y;

    feature_min_.assign(p, std::numeric_limits<double>::infinity());
    feature_max_.assign(p, -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            feature_min_[j] = std::min(feature_min_[j], dm.X(i, j));
            feature_max_[j] = std::max(feature_max_[j], dm.X(i, j));
        }

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

    if (options_.bandwidth > 0.0) {
        bandwidth_was_selected_ = false;
        bandwidth_grid_.clear();
        cv_mse_.clear();
        bandwidth_used_ = options_.bandwidth;
    } else {
        bandwidth_was_selected_ = true;

        const double h_min = 0.01 * std::sqrt(static_cast<double>(p));
        const double h_max = 2.0 * std::sqrt(static_cast<double>(p));
        bandwidth_grid_.assign(options_.n_bandwidth, 0.0);
        for (std::size_t k = 0; k < options_.n_bandwidth; ++k) {
            const double t = (options_.n_bandwidth == 1)
                               ? 0.0
                               : static_cast<double>(k) / static_cast<double>(options_.n_bandwidth - 1);
            bandwidth_grid_[k] = std::exp(std::log(h_min) + t * (std::log(h_max) - std::log(h_min)));
        }

        cv_mse_.assign(options_.n_bandwidth, std::numeric_limits<double>::infinity());
        for (std::size_t k = 0; k < options_.n_bandwidth; ++k) {
            double      sse   = 0.0;
            std::size_t count = 0;
            for (std::size_t i = 0; i < n; ++i) {
                std::vector<double> query(p);
                for (std::size_t j = 0; j < p; ++j) query[j] = training_X_std_(i, j);
                const double pred = predict_one(query, bandwidth_grid_[k], i);
                if (std::isnan(pred)) continue;
                const double d = training_y_[i] - pred;
                sse += d * d;
                ++count;
            }
            if (count > 0) cv_mse_[k] = sse / static_cast<double>(count);
        }

        std::size_t best_k     = 0;
        double      best_error = std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < options_.n_bandwidth; ++k) {
            if (cv_mse_[k] < best_error) {
                best_error = cv_mse_[k];
                best_k     = k;
            }
        }
        bandwidth_used_ = bandwidth_grid_[best_k];
    }

    fitted_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> query(p);
        for (std::size_t j = 0; j < p; ++j) query[j] = training_X_std_(i, j);
        fitted_[i] = predict_one(query, bandwidth_used_, i);
    }
}

double KernelRegression::predict_one(const std::vector<double>& query_std, double bandwidth,
                                     std::size_t exclude_row) const {
    const std::size_t n = training_X_std_.rows();
    const std::size_t p = training_X_std_.cols();

    double weighted_sum = 0.0;
    double weight_total  = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (i == exclude_row) continue;
        double dist = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            const double diff = training_X_std_(i, j) - query_std[j];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);
        const double w = kernel_weight(dist / bandwidth, options_.kernel);
        weighted_sum += w * training_y_[i];
        weight_total += w;
    }
    return (weight_total > 0.0) ? (weighted_sum / weight_total) : std::numeric_limits<double>::quiet_NaN();
}

double KernelRegression::r_squared() const {
    const double mean =
        std::accumulate(training_y_.begin(), training_y_.end(), 0.0) / static_cast<double>(training_y_.size());
    double tss = 0.0, rss = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < training_y_.size(); ++i) {
        if (std::isnan(fitted_[i])) continue;
        const double dt = training_y_[i] - mean;
        const double dr = training_y_[i] - fitted_[i];
        tss += dt * dt;
        rss += dr * dr;
        ++count;
    }
    return (count > 0 && tss > 0.0) ? (1.0 - rss / tss) : 1.0;
}

double KernelRegression::rmse() const {
    double      sse   = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < training_y_.size(); ++i) {
        if (std::isnan(fitted_[i])) continue;
        const double d = training_y_[i] - fitted_[i];
        sse += d * d;
        ++count;
    }
    return (count > 0) ? std::sqrt(sse / static_cast<double>(count)) : std::numeric_limits<double>::quiet_NaN();
}

std::string KernelRegression::summary() const {
    std::ostringstream out;
    out << "Call:\nkernelRegression(formula = " << formula_.text() << ", kernel = \"" << kernel_name(options_.kernel)
        << "\")\n\n";

    if (bandwidth_was_selected_)
        out << "Bandwidth: " << format_stat(bandwidth_used_, 4) << "  (selected by leave-one-out cross-validation "
            << "over " << bandwidth_grid_.size() << " values)\n";
    else
        out << "Bandwidth: " << format_stat(bandwidth_used_, 4) << "  (fixed)\n";

    out << "\nLeave-one-out R-squared: " << format_stat(r_squared(), 4) << "\n";
    out << "Leave-one-out RMSE: " << format_stat(rmse(), 4) << "  (" << observations_ << " observations)\n";

    return out.str();
}

void KernelRegression::print_summary(std::ostream& os) const { os << summary(); }

void KernelRegression::print_summary() const { print_summary(std::cout); }

std::vector<double> KernelRegression::predict(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "KernelRegression::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);

    std::vector<double> result(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        std::vector<double> query(dm.X.cols());
        for (std::size_t j = 0; j < dm.X.cols(); ++j) query[j] = (dm.X(i, j) - feature_mean_[j]) / feature_scale_[j];
        result[dm.used_row_indices[i]] = predict_one(query, bandwidth_used_);
    }
    return result;
}

plot::ScatterPlot KernelRegression::plot_fit(const dstruct::DataFrame& data, std::size_t grid_resolution) const {
    if (predictor_names_.size() != 1)
        throw std::invalid_argument("KernelRegression::plot_fit: only supported for a single-predictor model");
    const auto& x_feature = predictor_names_[0];

    double x_min = feature_min_[0], x_max = feature_max_[0];
    const double pad = std::max(1e-6, (x_max - x_min) * 0.05);
    x_min -= pad;
    x_max += pad;

    std::vector<double> grid_x(grid_resolution);
    for (std::size_t k = 0; k < grid_resolution; ++k)
        grid_x[k] = x_min + (x_max - x_min) * static_cast<double>(k) / static_cast<double>(grid_resolution - 1);

    dstruct::DataFrame grid;
    grid.add_column(x_feature, grid_x);
    const auto grid_y = predict(grid);

    auto plot = plot::ScatterPlot::create();

    std::vector<double> actual_x, actual_y;
    for (std::size_t i = 0; i < data.nrows(); ++i) {
        const auto x = data.optional_double_at(x_feature, i);
        const auto y = data.optional_double_at(design_.response_name, i);
        if (!x || !y) continue;
        actual_x.push_back(*x);
        actual_y.push_back(*y);
    }
    plot.points(actual_x, actual_y, "observed", {156, 163, 175}, 4.0);
    plot.line(grid_x, grid_y, "kernel fit", {220, 38, 38}, 2.5);

    plot.title("Kernel Regression Fit").x_label(x_feature).y_label(design_.response_name);
    return plot;
}

plot::ScatterPlot KernelRegression::plot_predicted_vs_actual() const {
    auto plot = plot::ScatterPlot::create();
    plot.points(training_y_, fitted_, "predictions");
    plot.title("Predicted vs Actual (leave-one-out)").x_label("Actual").y_label("Predicted");
    return plot;
}

plot::ScatterPlot KernelRegression::plot_residuals_vs_fitted() const {
    std::vector<double> residuals(training_y_.size());
    for (std::size_t i = 0; i < training_y_.size(); ++i) residuals[i] = training_y_[i] - fitted_[i];

    auto plot = plot::ScatterPlot::create();
    plot.points(fitted_, residuals, "residuals");
    plot.title("Residuals vs Fitted (leave-one-out)").x_label("Fitted values").y_label("Residuals");
    return plot;
}

plot::ScatterPlot KernelRegression::plot_cv_curve() const {
    if (!bandwidth_was_selected_)
        throw std::invalid_argument("KernelRegression::plot_cv_curve: no cross-validation curve for a fixed-"
                                    "bandwidth fit");

    const auto best_it = std::min_element(cv_mse_.begin(), cv_mse_.end());
    const auto best_k   = static_cast<std::size_t>(best_it - cv_mse_.begin());

    auto plot = plot::ScatterPlot::create();
    plot.line(bandwidth_grid_, cv_mse_, "LOO CV mean squared error");
    plot.points({bandwidth_grid_[best_k]}, {cv_mse_[best_k]}, "selected bandwidth", {220, 38, 38}, 8.0);
    plot.title("Bandwidth Cross-Validation Curve").x_label("Bandwidth").y_label("LOO mean squared error");
    return plot;
}

} // namespace datamunge::stats
