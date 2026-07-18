#include <datamunge/stats/random_forest_regressor.hpp>

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

RandomForestRegressor::RandomForestRegressor(const dstruct::DataFrame& data, const std::string& formula,
                                             RandomForestRegressorOptions options)
    : formula_(formula) {
    fit(data, std::move(options));
}

void RandomForestRegressor::fit(const dstruct::DataFrame& data, RandomForestRegressorOptions options) {
    options_ = options;
    if (options_.n_trees == 0) throw std::invalid_argument("RandomForestRegressor: n_trees must be at least 1");
    if (options_.sample_fraction <= 0.0)
        throw std::invalid_argument("RandomForestRegressor: sample_fraction must be positive");

    design_ = formula_.resolve(data, ResponseKind::Numeric, /*force_no_intercept=*/true);
    auto              dm = design_.build_matrix(data, /*require_response=*/true);
    const std::size_t n  = dm.X.rows();
    const std::size_t p  = dm.X.cols();

    predictor_names_ = design_.coefficient_names;
    observations_    = n;
    training_y_      = dm.y;

    max_features_used_ = options_.max_features;
    if (max_features_used_ == 0)
        max_features_used_ = std::max<std::size_t>(1, static_cast<std::size_t>(static_cast<double>(p) / 3.0));
    max_features_used_ = std::min(max_features_used_, p);

    const auto sample_size = static_cast<std::size_t>(
        std::max(1.0, std::round(options_.sample_fraction * static_cast<double>(n))));

    random::SplitMix64 sampler(options_.seed);

    trees_.clear();
    trees_.reserve(options_.n_trees);
    in_bag_.assign(options_.n_trees, std::vector<bool>(n, false));

    DecisionTreeRegressorOptions tree_options;
    tree_options.max_depth         = options_.max_depth;
    tree_options.min_samples_split = options_.min_samples_split;
    tree_options.min_samples_leaf  = options_.min_samples_leaf;
    tree_options.max_features      = max_features_used_;

    for (std::size_t t = 0; t < options_.n_trees; ++t) {
        std::vector<std::size_t> indices;
        indices.reserve(sample_size);
        std::vector<bool> in_bag(n, false);

        if (options_.bootstrap) {
            for (std::size_t i = 0; i < sample_size; ++i) {
                const auto idx = static_cast<std::size_t>(sampler.next_u64() % n);
                indices.push_back(idx);
                in_bag[idx] = true;
            }
        } else {
            std::vector<std::size_t> all(n);
            std::iota(all.begin(), all.end(), 0);
            const auto take = std::min(sample_size, n);
            for (std::size_t i = 0; i < take; ++i) {
                const std::size_t j = i + static_cast<std::size_t>(sampler.next_u64() % (n - i));
                std::swap(all[i], all[j]);
                indices.push_back(all[i]);
                in_bag[all[i]] = true;
            }
        }

        const auto bootstrap_frame = data.take_rows(indices);
        tree_options.random_seed   = options_.seed * 1000003ULL + t * 7919ULL + 1;

        trees_.emplace_back(bootstrap_frame, formula_.text(), tree_options);
        in_bag_[t] = std::move(in_bag);
    }

    fitted_values_ = predict(data);

    std::vector<double>      oob_sum(n, 0.0);
    std::vector<std::size_t> oob_count(n, 0);
    for (std::size_t t = 0; t < trees_.size(); ++t) {
        const auto preds = trees_[t].predict(data);
        for (std::size_t i = 0; i < n; ++i) {
            if (in_bag_[t][i] || std::isnan(preds[i])) continue;
            oob_sum[i] += preds[i];
            ++oob_count[i];
        }
    }
    double sse = 0.0, tss = 0.0;
    std::size_t considered = 0;
    const double mean_y =
        std::accumulate(training_y_.begin(), training_y_.end(), 0.0) / static_cast<double>(training_y_.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (oob_count[i] == 0) continue;
        const double pred = oob_sum[i] / static_cast<double>(oob_count[i]);
        const double dr   = training_y_[i] - pred;
        const double dt   = training_y_[i] - mean_y;
        sse += dr * dr;
        tss += dt * dt;
        ++considered;
    }
    oob_rmse_      = considered > 0 ? std::sqrt(sse / static_cast<double>(considered))
                                     : std::numeric_limits<double>::quiet_NaN();
    oob_r_squared_ = (considered > 0 && tss > 0.0) ? (1.0 - sse / tss) : std::numeric_limits<double>::quiet_NaN();
}

std::vector<double> RandomForestRegressor::feature_importance() const {
    std::vector<double>      importance(predictor_names_.size(), 0.0);
    std::vector<std::size_t> counts(predictor_names_.size(), 0);
    for (const auto& tree : trees_) {
        const auto& tree_predictors = tree.predictor_names();
        const auto  tree_importance = tree.feature_importance();
        for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
            const auto it = std::find(tree_predictors.begin(), tree_predictors.end(), predictor_names_[j]);
            if (it == tree_predictors.end()) continue;
            importance[j] += tree_importance[static_cast<std::size_t>(it - tree_predictors.begin())];
            ++counts[j];
        }
    }
    for (std::size_t j = 0; j < importance.size(); ++j)
        if (counts[j] > 0) importance[j] /= static_cast<double>(counts[j]);
    const double sum = std::accumulate(importance.begin(), importance.end(), 0.0);
    if (sum > 0.0)
        for (double& v : importance) v /= sum;
    return importance;
}

double RandomForestRegressor::r_squared() const {
    const double mean =
        std::accumulate(training_y_.begin(), training_y_.end(), 0.0) / static_cast<double>(training_y_.size());
    double tss = 0.0, rss = 0.0;
    for (std::size_t i = 0; i < training_y_.size(); ++i) {
        const double dt = training_y_[i] - mean;
        const double dr = training_y_[i] - fitted_values_[i];
        tss += dt * dt;
        rss += dr * dr;
    }
    return (tss > 0.0) ? (1.0 - rss / tss) : 1.0;
}

double RandomForestRegressor::rmse() const {
    double sse = 0.0;
    for (std::size_t i = 0; i < training_y_.size(); ++i) {
        const double d = training_y_[i] - fitted_values_[i];
        sse += d * d;
    }
    return std::sqrt(sse / static_cast<double>(training_y_.size()));
}

std::string RandomForestRegressor::summary() const {
    std::ostringstream out;
    out << "Call:\nrandomForest(formula = " << formula_.text() << ", n_trees = " << options_.n_trees
        << ", max_features = " << max_features_used_ << ", max_depth = " << options_.max_depth << ")\n\n";

    out << "Trees: " << trees_.size() << "  (max_features per split: " << max_features_used_ << " of "
        << predictor_names_.size() << ")\n\n";

    out << "Feature importance:\n";
    const auto importance = feature_importance();
    for (std::size_t j = 0; j < predictor_names_.size(); ++j)
        out << "  " << predictor_names_[j] << ": " << format_stat(importance[j], 4) << "\n";

    out << "\nR-squared: " << format_stat(r_squared(), 4) << "\n";
    out << "RMSE: " << format_stat(rmse(), 4) << "  (" << observations_ << " observations)\n";
    out << "Out-of-bag R-squared: " << format_stat(oob_r_squared_, 4) << "\n";
    out << "Out-of-bag RMSE: " << format_stat(oob_rmse_, 4) << "\n";

    return out.str();
}

void RandomForestRegressor::print_summary(std::ostream& os) const { os << summary(); }

void RandomForestRegressor::print_summary() const { print_summary(std::cout); }

std::vector<double> RandomForestRegressor::predict(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "RandomForestRegressor::predict");
    const std::size_t n = newdata.nrows();

    std::vector<double>      sum(n, 0.0);
    std::vector<std::size_t> count(n, 0);
    for (const auto& tree : trees_) {
        const auto preds = tree.predict(newdata);
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(preds[i])) continue;
            sum[i] += preds[i];
            ++count[i];
        }
    }

    std::vector<double> result(n, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < n; ++i)
        if (count[i] > 0) result[i] = sum[i] / static_cast<double>(count[i]);
    return result;
}

plot::ScatterPlot RandomForestRegressor::plot_predicted_vs_actual() const {
    auto plot = plot::ScatterPlot::create();
    plot.points(training_y_, fitted_values_, "predictions");
    plot.title("Predicted vs Actual").x_label("Actual").y_label("Predicted");
    return plot;
}

plot::ScatterPlot RandomForestRegressor::plot_residuals_vs_fitted() const {
    std::vector<double> residuals(training_y_.size());
    for (std::size_t i = 0; i < training_y_.size(); ++i) residuals[i] = training_y_[i] - fitted_values_[i];

    auto plot = plot::ScatterPlot::create();
    plot.points(fitted_values_, residuals, "residuals");
    plot.title("Residuals vs Fitted").x_label("Fitted values").y_label("Residuals");
    return plot;
}

} // namespace datamunge::stats
