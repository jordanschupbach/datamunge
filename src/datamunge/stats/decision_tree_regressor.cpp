#include <datamunge/stats/decision_tree_regressor.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
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

double sse_of(const std::vector<double>& y, const std::vector<std::size_t>& rows, double& mean_out) {
    double sum = 0.0;
    for (const auto r : rows) sum += y[r];
    const double mean = rows.empty() ? 0.0 : sum / static_cast<double>(rows.size());
    mean_out           = mean;
    double sse         = 0.0;
    for (const auto r : rows) {
        const double d = y[r] - mean;
        sse += d * d;
    }
    return sse;
}

} // namespace

DecisionTreeRegressor::DecisionTreeRegressor(const dstruct::DataFrame& data, const std::string& formula,
                                             DecisionTreeRegressorOptions options)
    : formula_(formula) {
    fit(data, std::move(options));
}

void DecisionTreeRegressor::fit(const dstruct::DataFrame& data, DecisionTreeRegressorOptions options) {
    options_ = options;
    rng_     = random::SplitMix64(options_.random_seed);
    design_  = formula_.resolve(data, ResponseKind::Numeric, /*force_no_intercept=*/true);

    auto              dm = design_.build_matrix(data, /*require_response=*/true);
    const std::size_t n  = dm.X.rows();

    predictor_names_ = design_.coefficient_names;
    observations_    = n;
    training_y_       = dm.y;
    if (options_.max_depth == 0) throw std::invalid_argument("DecisionTreeRegressor: max_depth must be at least 1");

    nodes_.clear();
    std::vector<std::size_t> all_rows(n);
    for (std::size_t i = 0; i < n; ++i) all_rows[i] = i;
    build_node(dm.X, dm.y, std::move(all_rows), 0);

    fitted_values_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> row(dm.X.cols());
        for (std::size_t j = 0; j < dm.X.cols(); ++j) row[j] = dm.X(i, j);
        fitted_values_[i] = nodes_[leaf_for(row)].mean_value;
    }
}

std::vector<std::size_t> DecisionTreeRegressor::feature_candidates(std::size_t p) {
    std::vector<std::size_t> all(p);
    std::iota(all.begin(), all.end(), 0);
    if (options_.max_features == 0 || options_.max_features >= p) return all;
    for (std::size_t i = 0; i < options_.max_features; ++i) {
        const std::size_t j = i + static_cast<std::size_t>(rng_.next_u64() % (p - i));
        std::swap(all[i], all[j]);
    }
    all.resize(options_.max_features);
    return all;
}

std::size_t DecisionTreeRegressor::build_node(const linalg::DenseMatrix<double>& X, const std::vector<double>& y,
                                              std::vector<std::size_t> rows, std::size_t depth) {
    const std::size_t p = X.cols();

    Node node;
    node.depth     = depth;
    node.n_samples = rows.size();
    node.sse       = sse_of(y, rows, node.mean_value);

    const std::size_t my_index = nodes_.size();
    nodes_.push_back(node);

    if (depth >= options_.max_depth || rows.size() < options_.min_samples_split || node.sse <= 1e-12) return my_index;

    double      best_gain      = 0.0;
    std::size_t best_feature   = 0;
    double      best_threshold = 0.0;
    bool        found          = false;

    for (const auto feature : feature_candidates(p)) {
        std::set<double> unique_values;
        for (const auto r : rows) unique_values.insert(X(r, feature));
        if (unique_values.size() < 2) continue;

        std::vector<double> sorted_values(unique_values.begin(), unique_values.end());
        for (std::size_t v = 0; v + 1 < sorted_values.size(); ++v) {
            const double threshold = 0.5 * (sorted_values[v] + sorted_values[v + 1]);

            std::vector<std::size_t> left_rows, right_rows;
            for (const auto r : rows) (X(r, feature) <= threshold ? left_rows : right_rows).push_back(r);
            if (left_rows.size() < options_.min_samples_leaf || right_rows.size() < options_.min_samples_leaf) continue;

            double left_mean = 0.0, right_mean = 0.0;
            const double left_sse  = sse_of(y, left_rows, left_mean);
            const double right_sse = sse_of(y, right_rows, right_mean);
            const double gain      = node.sse - (left_sse + right_sse);
            if (gain > best_gain + 1e-12) {
                best_gain      = gain;
                best_feature   = feature;
                best_threshold = threshold;
                found          = true;
            }
        }
    }

    if (!found || best_gain <= options_.min_impurity_decrease) return my_index;

    std::vector<std::size_t> left_rows, right_rows;
    for (const auto r : rows) (X(r, best_feature) <= best_threshold ? left_rows : right_rows).push_back(r);

    const std::size_t left_index  = build_node(X, y, std::move(left_rows), depth + 1);
    const std::size_t right_index = build_node(X, y, std::move(right_rows), depth + 1);

    nodes_[my_index].is_leaf       = false;
    nodes_[my_index].feature_index = best_feature;
    nodes_[my_index].threshold     = best_threshold;
    nodes_[my_index].left          = left_index;
    nodes_[my_index].right         = right_index;

    return my_index;
}

std::size_t DecisionTreeRegressor::leaf_for(const std::vector<double>& x) const {
    std::size_t index = 0;
    while (!nodes_[index].is_leaf)
        index = (x[nodes_[index].feature_index] <= nodes_[index].threshold) ? nodes_[index].left : nodes_[index].right;
    return index;
}

std::size_t DecisionTreeRegressor::leaf_count() const {
    return static_cast<std::size_t>(std::count_if(nodes_.begin(), nodes_.end(), [](const Node& n) { return n.is_leaf; }));
}

std::size_t DecisionTreeRegressor::depth() const {
    std::size_t max_depth = 0;
    for (const auto& node : nodes_)
        if (node.is_leaf) max_depth = std::max(max_depth, node.depth);
    return max_depth;
}

std::vector<double> DecisionTreeRegressor::feature_importance() const {
    std::vector<double> importance(predictor_names_.size(), 0.0);
    const double         total = static_cast<double>(observations_);
    for (const auto& node : nodes_) {
        if (node.is_leaf) continue;
        const auto& left  = nodes_[node.left];
        const auto& right = nodes_[node.right];
        importance[node.feature_index] += (static_cast<double>(node.n_samples) / total)
                                         * (node.sse - (left.sse + right.sse)) / static_cast<double>(node.n_samples);
    }
    const double sum = std::accumulate(importance.begin(), importance.end(), 0.0);
    if (sum > 0.0)
        for (double& v : importance) v /= sum;
    return importance;
}

double DecisionTreeRegressor::r_squared() const {
    const double mean = std::accumulate(training_y_.begin(), training_y_.end(), 0.0) / static_cast<double>(training_y_.size());
    double        tss = 0.0, rss = 0.0;
    for (std::size_t i = 0; i < training_y_.size(); ++i) {
        const double dt = training_y_[i] - mean;
        const double dr = training_y_[i] - fitted_values_[i];
        tss += dt * dt;
        rss += dr * dr;
    }
    return (tss > 0.0) ? (1.0 - rss / tss) : 1.0;
}

double DecisionTreeRegressor::rmse() const {
    double sse = 0.0;
    for (std::size_t i = 0; i < training_y_.size(); ++i) {
        const double d = training_y_[i] - fitted_values_[i];
        sse += d * d;
    }
    return std::sqrt(sse / static_cast<double>(training_y_.size()));
}

void DecisionTreeRegressor::dump_node(std::ostream& os, std::size_t index, std::size_t indent) const {
    const auto& node = nodes_[index];
    os << std::string(indent * 2, ' ');
    if (node.is_leaf) {
        os << "* value=" << format_stat(node.mean_value, 3) << "  n=" << node.n_samples
           << "  sse=" << format_stat(node.sse, 3) << "\n";
        return;
    }
    os << predictor_names_[node.feature_index] << " <= " << format_stat(node.threshold, 3) << "?  (n=" << node.n_samples
       << ", sse=" << format_stat(node.sse, 3) << ")\n";
    dump_node(os, node.left, indent + 1);
    dump_node(os, node.right, indent + 1);
}

std::string DecisionTreeRegressor::summary() const {
    std::ostringstream out;
    out << "Call:\ntree(formula = " << formula_.text() << ", max_depth = " << options_.max_depth << ")\n\n";

    out << "Tree: " << node_count() << " nodes, " << leaf_count() << " leaves, depth " << depth() << "\n\n";
    dump_node(out, 0, 0);

    out << "\nFeature importance:\n";
    const auto importance = feature_importance();
    for (std::size_t j = 0; j < predictor_names_.size(); ++j)
        out << "  " << predictor_names_[j] << ": " << format_stat(importance[j], 4) << "\n";

    out << "\nR-squared: " << format_stat(r_squared(), 4) << "\n";
    out << "RMSE: " << format_stat(rmse(), 4) << "  (" << observations_ << " observations)\n";

    return out.str();
}

void DecisionTreeRegressor::print_summary(std::ostream& os) const { os << summary(); }

void DecisionTreeRegressor::print_summary() const { print_summary(std::cout); }

std::vector<double> DecisionTreeRegressor::predict(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "DecisionTreeRegressor::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);

    std::vector<double> result(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        std::vector<double> row(dm.X.cols());
        for (std::size_t j = 0; j < dm.X.cols(); ++j) row[j] = dm.X(i, j);
        result[dm.used_row_indices[i]] = nodes_[leaf_for(row)].mean_value;
    }
    return result;
}

plot::ScatterPlot DecisionTreeRegressor::plot_predicted_vs_actual() const {
    auto plot = plot::ScatterPlot::create();
    plot.points(training_y_, fitted_values_, "predictions");
    plot.title("Predicted vs Actual").x_label("Actual").y_label("Predicted");
    return plot;
}

plot::ScatterPlot DecisionTreeRegressor::plot_residuals_vs_fitted() const {
    std::vector<double> residuals(training_y_.size());
    for (std::size_t i = 0; i < training_y_.size(); ++i) residuals[i] = training_y_[i] - fitted_values_[i];

    auto plot = plot::ScatterPlot::create();
    plot.points(fitted_values_, residuals, "residuals");
    plot.title("Residuals vs Fitted").x_label("Fitted values").y_label("Residuals");
    return plot;
}

} // namespace datamunge::stats
