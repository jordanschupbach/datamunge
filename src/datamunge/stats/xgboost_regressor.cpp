#include <datamunge/stats/xgboost_regressor.hpp>

#include <datamunge/random/random.hpp>

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

linalg::DenseMatrix<double> take_rows(const linalg::DenseMatrix<double>& X, const std::vector<std::size_t>& rows) {
    linalg::DenseMatrix<double> out(rows.size(), X.cols(), 0.0);
    for (std::size_t r = 0; r < rows.size(); ++r)
        for (std::size_t j = 0; j < X.cols(); ++j) out(r, j) = X(rows[r], j);
    return out;
}

template <typename T>
std::vector<T> take_rows(const std::vector<T>& v, const std::vector<std::size_t>& rows) {
    std::vector<T> out(rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) out[r] = v[rows[r]];
    return out;
}

} // namespace

XGBoostRegressor::XGBoostRegressor(const dstruct::DataFrame& data, const std::string& formula,
                                   XGBoostRegressorOptions options)
    : formula_(formula) {
    options_ = options;
    fit(data);
}

void XGBoostRegressor::fit(const dstruct::DataFrame& data) {
    if (options_.n_trees == 0) throw std::invalid_argument("XGBoostRegressor: n_trees must be at least 1");
    if (options_.subsample <= 0.0 || options_.subsample > 1.0)
        throw std::invalid_argument("XGBoostRegressor: subsample must be in (0, 1]");
    if (options_.colsample_bytree <= 0.0 || options_.colsample_bytree > 1.0)
        throw std::invalid_argument("XGBoostRegressor: colsample_bytree must be in (0, 1]");

    design_       = formula_.resolve(data, ResponseKind::Numeric, /*force_no_intercept=*/true);
    const auto dm = design_.build_matrix(data, /*require_response=*/true);
    const std::size_t n = dm.X.rows();

    predictor_names_ = design_.coefficient_names;
    observations_     = n;
    training_y_       = dm.y;

    base_score_ = std::accumulate(training_y_.begin(), training_y_.end(), 0.0) / static_cast<double>(n);

    detail::XGBoostTreeOptions tree_options;
    tree_options.max_depth        = options_.max_depth;
    tree_options.lambda           = options_.lambda;
    tree_options.alpha            = options_.alpha;
    tree_options.gamma            = options_.gamma;
    tree_options.min_child_weight = options_.min_child_weight;
    tree_options.min_samples_leaf = options_.min_samples_leaf;
    tree_options.colsample_bytree = options_.colsample_bytree;

    std::vector<double> scores(n, base_score_);
    const auto           sample_size = static_cast<std::size_t>(
        std::max(1.0, std::round(options_.subsample * static_cast<double>(n))));
    random::SplitMix64 sampler(options_.seed);

    trees_.clear();
    trees_.reserve(options_.n_trees);
    training_deviance_.reserve(options_.n_trees);

    for (std::size_t m = 0; m < options_.n_trees; ++m) {
        std::vector<double> gradient(n), hessian(n, 1.0); // squared-error loss: dL/dF = F - y, d2L/dF2 = 1
        for (std::size_t i = 0; i < n; ++i) gradient[i] = scores[i] - training_y_[i];

        detail::XGBoostTree tree;
        tree_options.seed = options_.seed * 1000003ULL + m * 7919ULL + 1;

        if (options_.subsample < 1.0) {
            std::vector<std::size_t> all(n);
            std::iota(all.begin(), all.end(), 0);
            for (std::size_t i = 0; i < sample_size; ++i) {
                const std::size_t j = i + static_cast<std::size_t>(sampler.next_u64() % (n - i));
                std::swap(all[i], all[j]);
            }
            std::vector<std::size_t> rows(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(sample_size));
            std::sort(rows.begin(), rows.end());
            tree.fit(take_rows(dm.X, rows), take_rows(gradient, rows), take_rows(hessian, rows), tree_options);
        } else {
            tree.fit(dm.X, gradient, hessian, tree_options);
        }

        const auto contribution = tree.predict(dm.X);
        for (std::size_t i = 0; i < n; ++i) scores[i] += options_.learning_rate * contribution[i];
        trees_.push_back(std::move(tree));

        double sse = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = training_y_[i] - scores[i];
            sse += d * d;
        }
        training_deviance_.push_back(sse / static_cast<double>(n));
    }

    fitted_ = scores;
}

std::vector<double> XGBoostRegressor::feature_importance() const {
    const std::size_t   p = predictor_names_.size();
    std::vector<double> importance(p, 0.0);
    for (const auto& tree : trees_) {
        const auto gain = tree.feature_gain(p);
        for (std::size_t j = 0; j < p; ++j) importance[j] += gain[j];
    }
    const double total = std::accumulate(importance.begin(), importance.end(), 0.0);
    if (total > 0.0)
        for (double& v : importance) v /= total;
    return importance;
}

double XGBoostRegressor::r_squared() const {
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

double XGBoostRegressor::rmse() const {
    double sse = 0.0;
    for (std::size_t i = 0; i < training_y_.size(); ++i) {
        const double d = training_y_[i] - fitted_[i];
        sse += d * d;
    }
    return std::sqrt(sse / static_cast<double>(training_y_.size()));
}

std::string XGBoostRegressor::summary() const {
    std::ostringstream out;
    out << "Call:\nxgboost(formula = " << formula_.text() << ", n_trees = " << options_.n_trees
        << ", learning_rate = " << format_stat(options_.learning_rate, 3) << ", max_depth = " << options_.max_depth
        << ", lambda = " << format_stat(options_.lambda, 3) << ", alpha = " << format_stat(options_.alpha, 3)
        << ", gamma = " << format_stat(options_.gamma, 3) << ")\n\n";

    out << "Feature importance (gain):\n";
    const auto importance = feature_importance();
    for (std::size_t j = 0; j < predictor_names_.size(); ++j)
        out << "  " << predictor_names_[j] << ": " << format_stat(importance[j], 4) << "\n";

    out << "\nR-squared: " << format_stat(r_squared(), 4) << "\n";
    out << "RMSE: " << format_stat(rmse(), 4) << "  (" << observations_ << " observations)\n";
    if (!training_deviance_.empty())
        out << "Training MSE: " << format_stat(training_deviance_.front(), 4) << " (round 1) -> "
            << format_stat(training_deviance_.back(), 4) << " (round " << training_deviance_.size() << ")\n";

    return out.str();
}

void XGBoostRegressor::print_summary(std::ostream& os) const { os << summary(); }

void XGBoostRegressor::print_summary() const { print_summary(std::cout); }

std::vector<double> XGBoostRegressor::predict(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "XGBoostRegressor::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);

    std::vector<double> compact(dm.X.rows(), base_score_);
    for (const auto& tree : trees_) {
        const auto contribution = tree.predict(dm.X);
        for (std::size_t i = 0; i < compact.size(); ++i) compact[i] += options_.learning_rate * contribution[i];
    }

    std::vector<double> result(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) result[dm.used_row_indices[i]] = compact[i];
    return result;
}

plot::RPlot XGBoostRegressor::plot_predicted_vs_actual() const {
    auto plot = plot::RPlot::create();
    plot.points(training_y_, fitted_, "predictions");
    plot.title("Predicted vs Actual").x_label("Actual").y_label("Predicted");
    return plot;
}

plot::RPlot XGBoostRegressor::plot_residuals_vs_fitted() const {
    std::vector<double> residuals(training_y_.size());
    for (std::size_t i = 0; i < training_y_.size(); ++i) residuals[i] = training_y_[i] - fitted_[i];

    auto plot = plot::RPlot::create();
    plot.points(fitted_, residuals, "residuals");
    plot.title("Residuals vs Fitted").x_label("Fitted values").y_label("Residuals");
    return plot;
}

plot::RPlot XGBoostRegressor::plot_training_deviance() const {
    std::vector<double> iteration(training_deviance_.size());
    for (std::size_t i = 0; i < training_deviance_.size(); ++i) iteration[i] = static_cast<double>(i + 1);

    auto plot = plot::RPlot::create();
    plot.line(iteration, training_deviance_, "training MSE");
    plot.title("Training Deviance").x_label("Boosting iteration").y_label("Mean squared error");
    return plot;
}

} // namespace datamunge::stats
