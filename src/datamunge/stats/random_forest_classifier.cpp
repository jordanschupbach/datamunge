#include <datamunge/stats/random_forest_classifier.hpp>

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

const plot::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

} // namespace

RandomForestClassifier::RandomForestClassifier(const dstruct::DataFrame& data, const std::string& formula,
                                               RandomForestClassifierOptions options)
    : formula_(formula) {
    fit(data, std::move(options));
}

void RandomForestClassifier::fit(const dstruct::DataFrame& data, RandomForestClassifierOptions options) {
    options_ = options;
    if (options_.n_trees == 0) throw std::invalid_argument("RandomForestClassifier: n_trees must be at least 1");
    if (options_.sample_fraction <= 0.0)
        throw std::invalid_argument("RandomForestClassifier: sample_fraction must be positive");

    design_ = formula_.resolve(data, ResponseKind::Categorical);
    auto              cd = design_.build_classification_matrix(data);
    const std::size_t n  = cd.X.rows();
    const std::size_t p  = cd.X.cols();

    predictor_names_ = design_.coefficient_names;
    observations_    = n;
    training_labels_ = cd.labels;

    classes_.clear();
    for (const auto& label : cd.labels)
        if (std::find(classes_.begin(), classes_.end(), label) == classes_.end()) classes_.push_back(label);
    std::sort(classes_.begin(), classes_.end());
    if (classes_.size() < 2)
        throw std::invalid_argument("RandomForestClassifier: response must have at least 2 distinct classes, found "
                                    + std::to_string(classes_.size()));

    feature_min_.assign(p, std::numeric_limits<double>::infinity());
    feature_max_.assign(p, -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            feature_min_[j] = std::min(feature_min_[j], cd.X(i, j));
            feature_max_[j] = std::max(feature_max_[j], cd.X(i, j));
        }

    max_features_used_ = options_.max_features;
    if (max_features_used_ == 0)
        max_features_used_ = std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(static_cast<double>(p))));
    max_features_used_ = std::min(max_features_used_, p);

    const auto sample_size = static_cast<std::size_t>(
        std::max(1.0, std::round(options_.sample_fraction * static_cast<double>(n))));

    random::SplitMix64 sampler(options_.seed);

    trees_.clear();
    trees_.reserve(options_.n_trees);
    in_bag_.assign(options_.n_trees, std::vector<bool>(n, false));

    DecisionTreeClassifierOptions tree_options;
    tree_options.max_depth          = options_.max_depth;
    tree_options.min_samples_split  = options_.min_samples_split;
    tree_options.min_samples_leaf   = options_.min_samples_leaf;
    tree_options.criterion          = options_.criterion;
    tree_options.max_features       = max_features_used_;

    for (std::size_t t = 0; t < options_.n_trees; ++t) {
        // A bootstrap sample can, in rare cases (small n, imbalanced
        // classes), miss a class entirely -- retry with fresh draws rather
        // than letting the whole forest fail to fit.
        constexpr int kMaxAttempts = 25;
        bool          built        = false;
        for (int attempt = 0; attempt < kMaxAttempts && !built; ++attempt) {
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
            tree_options.random_seed   = options_.seed * 1000003ULL + t * 7919ULL + static_cast<std::uint64_t>(attempt) + 1;

            try {
                trees_.emplace_back(bootstrap_frame, formula_.text(), tree_options);
                in_bag_[t] = std::move(in_bag);
                built      = true;
            } catch (const std::invalid_argument&) {
                // Likely a single-class bootstrap sample; retry.
            }
        }
        if (!built)
            throw std::runtime_error("RandomForestClassifier: could not draw a bootstrap sample containing at "
                                     "least 2 classes after repeated attempts");
    }

    fitted_classes_ = predict(data);

    std::vector<std::vector<double>> oob_vote_counts(n, std::vector<double>(classes_.size(), 0.0));
    std::vector<bool> ever_oob(n, false);
    for (std::size_t t = 0; t < trees_.size(); ++t) {
        const auto preds = trees_[t].predict(data);
        for (std::size_t i = 0; i < n; ++i) {
            if (in_bag_[t][i] || preds[i].empty()) continue;
            const auto it = std::lower_bound(classes_.begin(), classes_.end(), preds[i]);
            if (it == classes_.end() || *it != preds[i]) continue;
            oob_vote_counts[i][static_cast<std::size_t>(it - classes_.begin())] += 1.0;
            ever_oob[i] = true;
        }
    }
    std::size_t correct = 0, considered = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!ever_oob[i]) continue;
        const auto best =
            static_cast<std::size_t>(std::max_element(oob_vote_counts[i].begin(), oob_vote_counts[i].end())
                                     - oob_vote_counts[i].begin());
        ++considered;
        if (classes_[best] == training_labels_[i]) ++correct;
    }
    oob_accuracy_ = considered > 0 ? static_cast<double>(correct) / static_cast<double>(considered)
                                    : std::numeric_limits<double>::quiet_NaN();
}

std::vector<double> RandomForestClassifier::feature_importance() const {
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

double RandomForestClassifier::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
        if (fitted_classes_[i] == training_labels_[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> RandomForestClassifier::confusion_matrix() const {
    const std::size_t k = classes_.size();
    linalg::DenseMatrix<double> matrix(k, k, 0.0);
    for (std::size_t i = 0; i < training_labels_.size(); ++i) {
        const auto actual_it    = std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]);
        const auto predicted_it = std::lower_bound(classes_.begin(), classes_.end(), fitted_classes_[i]);
        matrix(static_cast<std::size_t>(actual_it - classes_.begin()),
               static_cast<std::size_t>(predicted_it - classes_.begin())) += 1.0;
    }
    return matrix;
}

std::string RandomForestClassifier::summary() const {
    std::ostringstream out;
    out << "Call:\nrandomForest(formula = " << formula_.text() << ", n_trees = " << options_.n_trees
        << ", max_features = " << max_features_used_
        << ", criterion = " << (options_.criterion == SplitCriterion::Gini ? "gini" : "entropy")
        << ", max_depth = " << options_.max_depth << ")\n\n";

    out << "Number of Classes: " << classes_.size() << "\n";
    out << "Levels: ";
    for (const auto& c : classes_) out << c << " ";
    out << "\n\n";

    out << "Trees: " << trees_.size() << "  (max_features per split: " << max_features_used_ << " of "
        << predictor_names_.size() << ")\n\n";

    out << "Feature importance:\n";
    const auto importance = feature_importance();
    for (std::size_t j = 0; j < predictor_names_.size(); ++j)
        out << "  " << predictor_names_[j] << ": " << format_stat(importance[j], 4) << "\n";

    out << "\nTraining accuracy: " << format_stat(training_accuracy() * 100.0, 2) << "%\n";
    out << "Out-of-bag accuracy: " << format_stat(oob_accuracy_ * 100.0, 2) << "%  (" << observations_
        << " observations)\n";

    return out.str();
}

void RandomForestClassifier::print_summary(std::ostream& os) const { os << summary(); }

void RandomForestClassifier::print_summary() const { print_summary(std::cout); }

std::vector<std::string> RandomForestClassifier::predict(const dstruct::DataFrame& newdata) const {
    return predict_detail(newdata).class_label;
}

RandomForestClassifierPrediction RandomForestClassifier::predict_detail(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "RandomForestClassifier::predict");
    const std::size_t n = newdata.nrows();

    std::vector<std::vector<double>> vote_counts(n, std::vector<double>(classes_.size(), 0.0));
    std::vector<bool>                any_vote(n, false);

    for (const auto& tree : trees_) {
        const auto preds = tree.predict(newdata);
        for (std::size_t i = 0; i < n; ++i) {
            if (preds[i].empty()) continue;
            const auto it = std::lower_bound(classes_.begin(), classes_.end(), preds[i]);
            if (it == classes_.end() || *it != preds[i]) continue;
            vote_counts[i][static_cast<std::size_t>(it - classes_.begin())] += 1.0;
            any_vote[i] = true;
        }
    }

    RandomForestClassifierPrediction result;
    result.class_label.assign(n, "");
    result.vote_share.assign(n, {});
    for (std::size_t i = 0; i < n; ++i) {
        if (!any_vote[i]) continue;
        const double total = std::accumulate(vote_counts[i].begin(), vote_counts[i].end(), 0.0);
        std::vector<double> share(classes_.size());
        for (std::size_t c = 0; c < classes_.size(); ++c) share[c] = vote_counts[i][c] / total;
        const auto best =
            static_cast<std::size_t>(std::max_element(vote_counts[i].begin(), vote_counts[i].end())
                                     - vote_counts[i].begin());
        result.class_label[i] = classes_[best];
        result.vote_share[i]  = std::move(share);
    }
    return result;
}

plot::ScatterPlot RandomForestClassifier::plot_classification(const dstruct::DataFrame& data,
                                                               const std::string& x_feature,
                                                               const std::string& y_feature) const {
    const auto  predictions = predict(data);
    const auto& response    = design_.response_name;

    auto plot = plot::ScatterPlot::create();
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        std::vector<double> xs, ys;
        for (std::size_t i = 0; i < data.nrows(); ++i) {
            const auto actual = data.optional_string_at(response, i);
            if (!actual || *actual != classes_[c]) continue;
            const auto x = data.optional_double_at(x_feature, i);
            const auto y = data.optional_double_at(y_feature, i);
            if (!x || !y) continue;
            xs.push_back(*x);
            ys.push_back(*y);
        }
        plot.points(xs, ys, classes_[c], kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
    }

    std::vector<double> mis_x, mis_y;
    for (std::size_t i = 0; i < data.nrows(); ++i) {
        const auto actual = data.optional_string_at(response, i);
        if (!actual || i >= predictions.size() || predictions[i].empty() || predictions[i] == *actual) continue;
        const auto x = data.optional_double_at(x_feature, i);
        const auto y = data.optional_double_at(y_feature, i);
        if (!x || !y) continue;
        mis_x.push_back(*x);
        mis_y.push_back(*y);
    }
    if (!mis_x.empty()) plot.points(mis_x, mis_y, "misclassified", {15, 15, 15}, 9.0);

    plot.title("Random Forest Classification").x_label(x_feature).y_label(y_feature);
    return plot;
}

plot::ScatterPlot RandomForestClassifier::plot_decision_regions(const std::string& x_feature,
                                                                 const std::string& y_feature,
                                                                 std::size_t grid_resolution) const {
    if (predictor_names_.size() != 2)
        throw std::invalid_argument("RandomForestClassifier::plot_decision_regions: model must have exactly 2 "
                                    "predictors");
    std::size_t xi = 2, yi = 2;
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_names_[j] == x_feature) xi = j;
        if (predictor_names_[j] == y_feature) yi = j;
    }
    if (xi == 2 || yi == 2)
        throw std::invalid_argument("RandomForestClassifier::plot_decision_regions: x_feature/y_feature must match "
                                    "the model's two predictors (" + predictor_names_[0] + ", " + predictor_names_[1]
                                    + ")");

    double x_min = feature_min_[xi], x_max = feature_max_[xi];
    double y_min = feature_min_[yi], y_max = feature_max_[yi];
    const double x_pad = std::max(1e-6, (x_max - x_min) * 0.1);
    const double y_pad = std::max(1e-6, (y_max - y_min) * 0.1);
    x_min -= x_pad;
    x_max += x_pad;
    y_min -= y_pad;
    y_max += y_pad;

    std::vector<double> col0, col1;
    col0.reserve(grid_resolution * grid_resolution);
    col1.reserve(grid_resolution * grid_resolution);
    for (std::size_t gx = 0; gx < grid_resolution; ++gx) {
        for (std::size_t gy = 0; gy < grid_resolution; ++gy) {
            const double x = x_min + (x_max - x_min) * static_cast<double>(gx) / static_cast<double>(grid_resolution - 1);
            const double y = y_min + (y_max - y_min) * static_cast<double>(gy) / static_cast<double>(grid_resolution - 1);
            col0.push_back(predictor_names_[0] == x_feature ? x : y);
            col1.push_back(predictor_names_[1] == x_feature ? x : y);
        }
    }
    dstruct::DataFrame grid;
    grid.add_column(predictor_names_[0], col0);
    grid.add_column(predictor_names_[1], col1);

    const auto detail = predict_detail(grid);

    std::vector<std::vector<double>> grid_x(classes_.size()), grid_y(classes_.size());
    for (std::size_t i = 0; i < detail.class_label.size(); ++i) {
        if (detail.class_label[i].empty()) continue;
        const auto it = std::lower_bound(classes_.begin(), classes_.end(), detail.class_label[i]);
        const auto c  = static_cast<std::size_t>(it - classes_.begin());
        grid_x[c].push_back(predictor_names_[0] == x_feature ? col0[i] : col1[i]);
        grid_y[c].push_back(predictor_names_[0] == y_feature ? col0[i] : col1[i]);
    }

    auto plot = plot::ScatterPlot::create();
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        auto color = kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))];
        color.r    = static_cast<std::uint8_t>(std::min(255, color.r + (255 - color.r) * 3 / 4));
        color.g    = static_cast<std::uint8_t>(std::min(255, color.g + (255 - color.g) * 3 / 4));
        color.b    = static_cast<std::uint8_t>(std::min(255, color.b + (255 - color.b) * 3 / 4));
        plot.points(grid_x[c], grid_y[c], classes_[c] + " region", color, 3.0);
    }

    plot.title("Random Forest Decision Regions").x_label(x_feature).y_label(y_feature);
    return plot;
}

} // namespace datamunge::stats
