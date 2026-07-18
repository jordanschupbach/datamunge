#include <datamunge/stats/gbm_classifier.hpp>

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

dstruct::DataFrame build_predictor_frame(const linalg::DenseMatrix<double>& X) {
    dstruct::DataFrame frame;
    for (std::size_t j = 0; j < X.cols(); ++j) {
        std::vector<double> column(X.rows());
        for (std::size_t i = 0; i < X.rows(); ++i) column[i] = X(i, j);
        frame.add_column("x" + std::to_string(j), std::move(column));
    }
    return frame;
}

std::string predictor_formula(std::size_t p) {
    std::ostringstream oss;
    oss << "y ~ x0";
    for (std::size_t j = 1; j < p; ++j) oss << " + x" << j;
    return oss.str();
}

std::vector<double> softmax_row(const linalg::DenseMatrix<double>& scores, std::size_t row, std::size_t K) {
    double max_score = scores(row, 0);
    for (std::size_t k = 1; k < K; ++k) max_score = std::max(max_score, scores(row, k));
    std::vector<double> p(K);
    double               sum_exp = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        p[k] = std::exp(scores(row, k) - max_score);
        sum_exp += p[k];
    }
    for (double& v : p) v /= sum_exp;
    return p;
}

} // namespace

GBMClassifier::GBMClassifier(const dstruct::DataFrame& data, const std::string& formula, GBMClassifierOptions options)
    : formula_(formula) {
    options_ = options;
    fit(data);
}

void GBMClassifier::fit(const dstruct::DataFrame& data) {
    if (options_.n_trees == 0) throw std::invalid_argument("GBMClassifier: n_trees must be at least 1");
    if (options_.subsample <= 0.0 || options_.subsample > 1.0)
        throw std::invalid_argument("GBMClassifier: subsample must be in (0, 1]");

    design_       = formula_.resolve(data, ResponseKind::Categorical);
    const auto cd = design_.build_classification_matrix(data);
    const std::size_t n = cd.X.rows();
    const std::size_t p = cd.X.cols();

    predictor_names_ = design_.coefficient_names;
    observations_    = n;
    training_labels_ = cd.labels;

    classes_.clear();
    for (const auto& label : cd.labels)
        if (std::find(classes_.begin(), classes_.end(), label) == classes_.end()) classes_.push_back(label);
    std::sort(classes_.begin(), classes_.end());
    if (classes_.size() < 2)
        throw std::invalid_argument("GBMClassifier: response must have at least 2 distinct classes, found "
                                    + std::to_string(classes_.size()));
    const std::size_t K = classes_.size();

    feature_min_.assign(p, std::numeric_limits<double>::infinity());
    feature_max_.assign(p, -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            feature_min_[j] = std::min(feature_min_[j], cd.X(i, j));
            feature_max_[j] = std::max(feature_max_[j], cd.X(i, j));
        }

    std::vector<std::size_t> class_index(n);
    std::vector<double>      class_count(K, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        class_index[i] = static_cast<std::size_t>(std::lower_bound(classes_.begin(), classes_.end(), cd.labels[i])
                                                  - classes_.begin());
        class_count[class_index[i]] += 1.0;
    }
    initial_scores_.assign(K, 0.0);
    for (std::size_t k = 0; k < K; ++k) initial_scores_[k] = std::log(class_count[k] / static_cast<double>(n));

    const auto predictor_frame = build_predictor_frame(cd.X);
    const auto tree_formula    = predictor_formula(p);

    DecisionTreeRegressorOptions tree_options;
    tree_options.max_depth         = options_.max_depth;
    tree_options.min_samples_split = options_.min_samples_split;
    tree_options.min_samples_leaf  = options_.min_samples_leaf;

    linalg::DenseMatrix<double> scores(n, K, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < K; ++k) scores(i, k) = initial_scores_[k];

    random::SplitMix64 sampler(options_.seed);
    const auto           sample_size = static_cast<std::size_t>(
        std::max(1.0, std::round(options_.subsample * static_cast<double>(n))));
    const bool use_subsample = options_.subsample < 1.0;

    trees_.clear();
    trees_.reserve(options_.n_trees);
    training_deviance_.reserve(options_.n_trees);

    for (std::size_t m = 0; m < options_.n_trees; ++m) {
        std::vector<std::vector<double>> proba(n);
        for (std::size_t i = 0; i < n; ++i) proba[i] = softmax_row(scores, i, K);

        std::vector<std::size_t> rows;
        if (use_subsample) {
            std::vector<std::size_t> all(n);
            std::iota(all.begin(), all.end(), 0);
            for (std::size_t i = 0; i < sample_size; ++i) {
                const std::size_t j = i + static_cast<std::size_t>(sampler.next_u64() % (n - i));
                std::swap(all[i], all[j]);
            }
            rows.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(sample_size));
            std::sort(rows.begin(), rows.end());
        }

        std::vector<DecisionTreeRegressor> round_trees;
        round_trees.reserve(K);
        for (std::size_t k = 0; k < K; ++k) {
            std::vector<double> residual(n);
            for (std::size_t i = 0; i < n; ++i) residual[i] = (class_index[i] == k ? 1.0 : 0.0) - proba[i][k];

            dstruct::DataFrame round_frame;
            if (use_subsample) {
                round_frame = predictor_frame.take_rows(rows);
                std::vector<double> sub_residual(rows.size());
                for (std::size_t r = 0; r < rows.size(); ++r) sub_residual[r] = residual[rows[r]];
                round_frame.add_column("y", std::move(sub_residual));
            } else {
                round_frame = predictor_frame;
                round_frame.add_column("y", residual);
            }

            round_trees.emplace_back(round_frame, tree_formula, tree_options);
            const auto contribution = round_trees.back().predict(predictor_frame);
            for (std::size_t i = 0; i < n; ++i) scores(i, k) += options_.learning_rate * contribution[i];
        }
        trees_.push_back(std::move(round_trees));

        double deviance = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto p_row = softmax_row(scores, i, K);
            deviance += -std::log(std::max(p_row[class_index[i]], 1e-15));
        }
        training_deviance_.push_back(deviance / static_cast<double>(n));
    }

    fitted_classes_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t best = 0;
        for (std::size_t k = 1; k < K; ++k)
            if (scores(i, k) > scores(i, best)) best = k;
        fitted_classes_[i] = classes_[best];
    }
}

linalg::DenseMatrix<double> GBMClassifier::score_frame(const dstruct::DataFrame& predictor_frame,
                                                        std::size_t n_rows) const {
    const std::size_t K = classes_.size();
    linalg::DenseMatrix<double> scores(n_rows, K, 0.0);
    for (std::size_t i = 0; i < n_rows; ++i)
        for (std::size_t k = 0; k < K; ++k) scores(i, k) = initial_scores_[k];

    for (const auto& round : trees_) {
        for (std::size_t k = 0; k < K; ++k) {
            const auto contribution = round[k].predict(predictor_frame);
            for (std::size_t i = 0; i < n_rows; ++i) scores(i, k) += options_.learning_rate * contribution[i];
        }
    }
    return scores;
}

std::vector<double> GBMClassifier::feature_importance() const {
    const std::size_t   p = predictor_names_.size();
    std::vector<double> importance(p, 0.0);
    std::size_t          count = 0;
    for (const auto& round : trees_) {
        for (const auto& tree : round) {
            const auto imp = tree.feature_importance();
            for (std::size_t j = 0; j < p; ++j) importance[j] += imp[j];
            ++count;
        }
    }
    if (count > 0)
        for (double& v : importance) v /= static_cast<double>(count);
    return importance;
}

double GBMClassifier::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
        if (fitted_classes_[i] == training_labels_[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> GBMClassifier::confusion_matrix() const {
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

std::string GBMClassifier::summary() const {
    std::ostringstream out;
    out << "Call:\ngbm(formula = " << formula_.text() << ", n_trees = " << options_.n_trees << ", learning_rate = "
        << format_stat(options_.learning_rate, 3) << ", max_depth = " << options_.max_depth << ")\n\n";

    out << "Number of Classes: " << classes_.size() << "\n";
    out << "Levels: ";
    for (const auto& c : classes_) out << c << " ";
    out << "\n\n";

    out << "Feature importance:\n";
    const auto importance = feature_importance();
    for (std::size_t j = 0; j < predictor_names_.size(); ++j)
        out << "  " << predictor_names_[j] << ": " << format_stat(importance[j], 4) << "\n";

    out << "\nTraining accuracy: " << format_stat(training_accuracy() * 100.0, 2) << "%  (" << observations_
        << " observations)\n";
    if (!training_deviance_.empty())
        out << "Training deviance: " << format_stat(training_deviance_.front(), 4) << " (round 1) -> "
            << format_stat(training_deviance_.back(), 4) << " (round " << training_deviance_.size() << ")\n";

    return out.str();
}

void GBMClassifier::print_summary(std::ostream& os) const { os << summary(); }

void GBMClassifier::print_summary() const { print_summary(std::cout); }

std::vector<std::string> GBMClassifier::predict(const dstruct::DataFrame& newdata) const {
    return predict_detail(newdata).class_label;
}

GBMClassifierPrediction GBMClassifier::predict_detail(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "GBMClassifier::predict");
    const auto dm               = design_.build_matrix(newdata, /*require_response=*/false);
    const auto predictor_frame  = build_predictor_frame(dm.X);
    const auto scores           = score_frame(predictor_frame, dm.X.rows());
    const std::size_t K         = classes_.size();

    GBMClassifierPrediction result;
    result.class_label.assign(newdata.nrows(), "");
    result.probability.assign(newdata.nrows(), {});
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        const auto p_row = softmax_row(scores, i, K);
        std::size_t best = 0;
        for (std::size_t k = 1; k < K; ++k)
            if (p_row[k] > p_row[best]) best = k;

        const auto original_row              = dm.used_row_indices[i];
        result.class_label[original_row]      = classes_[best];
        result.probability[original_row]      = p_row;
    }
    return result;
}

plot::ScatterPlot GBMClassifier::plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
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

    plot.title("GBM Classification").x_label(x_feature).y_label(y_feature);
    return plot;
}

plot::ScatterPlot GBMClassifier::plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                        std::size_t grid_resolution) const {
    if (predictor_names_.size() != 2)
        throw std::invalid_argument("GBMClassifier::plot_decision_regions: model must have exactly 2 predictors");
    std::size_t xi = 2, yi = 2;
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_names_[j] == x_feature) xi = j;
        if (predictor_names_[j] == y_feature) yi = j;
    }
    if (xi == 2 || yi == 2)
        throw std::invalid_argument("GBMClassifier::plot_decision_regions: x_feature/y_feature must match the "
                                    "model's two predictors (" + predictor_names_[0] + ", " + predictor_names_[1]
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

    plot.title("GBM Decision Regions").x_label(x_feature).y_label(y_feature);
    return plot;
}

plot::ScatterPlot GBMClassifier::plot_training_deviance() const {
    std::vector<double> iteration(training_deviance_.size());
    for (std::size_t i = 0; i < training_deviance_.size(); ++i) iteration[i] = static_cast<double>(i + 1);

    auto plot = plot::ScatterPlot::create();
    plot.line(iteration, training_deviance_, "training deviance");
    plot.title("Training Deviance").x_label("Boosting iteration").y_label("Multinomial deviance");
    return plot;
}

} // namespace datamunge::stats
