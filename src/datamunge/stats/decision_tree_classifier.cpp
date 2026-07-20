#include <datamunge/stats/decision_tree_classifier.hpp>

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

const plot::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

} // namespace

DecisionTreeClassifier::DecisionTreeClassifier(const dstruct::DataFrame& data, const std::string& formula,
                                               DecisionTreeClassifierOptions options)
    : formula_(formula) {
    fit(data, std::move(options));
}

double DecisionTreeClassifier::impurity_of(const std::vector<double>& class_counts, std::size_t n) const {
    if (n == 0) return 0.0;
    double impurity = 0.0;
    if (options_.criterion == SplitCriterion::Gini) {
        double sum_sq = 0.0;
        for (const double count : class_counts) {
            const double p = count / static_cast<double>(n);
            sum_sq += p * p;
        }
        impurity = 1.0 - sum_sq;
    } else {
        for (const double count : class_counts) {
            if (count <= 0.0) continue;
            const double p = count / static_cast<double>(n);
            impurity -= p * std::log2(p);
        }
    }
    return impurity;
}

void DecisionTreeClassifier::fit(const dstruct::DataFrame& data, DecisionTreeClassifierOptions options) {
    options_ = options;
    rng_     = random::SplitMix64(options_.random_seed);
    design_  = formula_.resolve(data, ResponseKind::Categorical);

    auto              cd = design_.build_classification_matrix(data);
    const std::size_t n  = cd.X.rows();

    predictor_names_ = design_.coefficient_names;
    observations_    = n;
    training_labels_ = cd.labels;

    classes_.clear();
    for (const auto& label : cd.labels)
        if (std::find(classes_.begin(), classes_.end(), label) == classes_.end()) classes_.push_back(label);
    std::sort(classes_.begin(), classes_.end());
    if (classes_.size() < 2)
        throw std::invalid_argument("DecisionTreeClassifier: response must have at least 2 distinct classes, found "
                                    + std::to_string(classes_.size()));
    if (options_.max_depth == 0)
        throw std::invalid_argument("DecisionTreeClassifier: max_depth must be at least 1");

    std::vector<std::size_t> class_index(n);
    for (std::size_t i = 0; i < n; ++i)
        class_index[i] = static_cast<std::size_t>(std::lower_bound(classes_.begin(), classes_.end(), cd.labels[i])
                                                  - classes_.begin());

    const std::size_t p = cd.X.cols();
    feature_min_.assign(p, std::numeric_limits<double>::infinity());
    feature_max_.assign(p, -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            feature_min_[j] = std::min(feature_min_[j], cd.X(i, j));
            feature_max_[j] = std::max(feature_max_[j], cd.X(i, j));
        }

    nodes_.clear();
    std::vector<std::size_t> all_rows(n);
    for (std::size_t i = 0; i < n; ++i) all_rows[i] = i;
    build_node(cd.X, class_index, std::move(all_rows), 0);

    std::vector<std::string> fitted(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> row(cd.X.cols());
        for (std::size_t j = 0; j < cd.X.cols(); ++j) row[j] = cd.X(i, j);
        fitted[i] = classes_[nodes_[classify_row(row)].predicted_class_index];
    }
    fitted_classes_ = std::move(fitted);
}

std::vector<std::size_t> DecisionTreeClassifier::feature_candidates(std::size_t p) {
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

std::size_t DecisionTreeClassifier::build_node(const linalg::DenseMatrix<double>& X,
                                               const std::vector<std::size_t>& class_index,
                                               std::vector<std::size_t> rows, std::size_t depth) {
    const std::size_t k = classes_.size();
    const std::size_t p = X.cols();

    Node node;
    node.depth     = depth;
    node.n_samples = rows.size();
    node.class_counts.assign(k, 0.0);
    for (const auto r : rows) node.class_counts[class_index[r]] += 1.0;
    node.predicted_class_index =
        static_cast<std::size_t>(std::max_element(node.class_counts.begin(), node.class_counts.end())
                                 - node.class_counts.begin());
    node.impurity = impurity_of(node.class_counts, rows.size());

    const std::size_t my_index = nodes_.size();
    nodes_.push_back(node);

    const std::size_t num_nonzero_classes =
        static_cast<std::size_t>(std::count_if(node.class_counts.begin(), node.class_counts.end(),
                                               [](const double c) { return c > 0.0; }));
    if (depth >= options_.max_depth || rows.size() < options_.min_samples_split || num_nonzero_classes <= 1)
        return my_index;

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

            std::vector<double> left_counts(k, 0.0), right_counts(k, 0.0);
            std::size_t          left_n = 0, right_n = 0;
            for (const auto r : rows) {
                if (X(r, feature) <= threshold) {
                    left_counts[class_index[r]] += 1.0;
                    ++left_n;
                } else {
                    right_counts[class_index[r]] += 1.0;
                    ++right_n;
                }
            }
            if (left_n < options_.min_samples_leaf || right_n < options_.min_samples_leaf) continue;

            const double weighted = (static_cast<double>(left_n) * impurity_of(left_counts, left_n)
                                    + static_cast<double>(right_n) * impurity_of(right_counts, right_n))
                                   / static_cast<double>(rows.size());
            const double gain = node.impurity - weighted;
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

    const std::size_t left_index  = build_node(X, class_index, std::move(left_rows), depth + 1);
    const std::size_t right_index = build_node(X, class_index, std::move(right_rows), depth + 1);

    nodes_[my_index].is_leaf       = false;
    nodes_[my_index].feature_index = best_feature;
    nodes_[my_index].threshold     = best_threshold;
    nodes_[my_index].left          = left_index;
    nodes_[my_index].right         = right_index;

    return my_index;
}

std::size_t DecisionTreeClassifier::classify_row(const std::vector<double>& x) const {
    std::size_t index = 0;
    while (!nodes_[index].is_leaf)
        index = (x[nodes_[index].feature_index] <= nodes_[index].threshold) ? nodes_[index].left : nodes_[index].right;
    return index;
}

std::size_t DecisionTreeClassifier::leaf_count() const {
    return static_cast<std::size_t>(std::count_if(nodes_.begin(), nodes_.end(), [](const Node& n) { return n.is_leaf; }));
}

std::size_t DecisionTreeClassifier::depth() const {
    std::size_t max_depth = 0;
    for (const auto& node : nodes_)
        if (node.is_leaf) max_depth = std::max(max_depth, node.depth);
    return max_depth;
}

std::vector<double> DecisionTreeClassifier::feature_importance() const {
    std::vector<double> importance(predictor_names_.size(), 0.0);
    const double         total = static_cast<double>(observations_);
    for (const auto& node : nodes_) {
        if (node.is_leaf) continue;
        const auto& left  = nodes_[node.left];
        const auto& right = nodes_[node.right];
        const double weighted_child =
            (static_cast<double>(left.n_samples) * left.impurity + static_cast<double>(right.n_samples) * right.impurity)
            / static_cast<double>(node.n_samples);
        importance[node.feature_index] += (static_cast<double>(node.n_samples) / total) * (node.impurity - weighted_child);
    }
    const double sum = std::accumulate(importance.begin(), importance.end(), 0.0);
    if (sum > 0.0)
        for (double& v : importance) v /= sum;
    return importance;
}

double DecisionTreeClassifier::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
        if (fitted_classes_[i] == training_labels_[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> DecisionTreeClassifier::confusion_matrix() const {
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

void DecisionTreeClassifier::dump_node(std::ostream& os, std::size_t index, std::size_t indent) const {
    const auto& node = nodes_[index];
    os << std::string(indent * 2, ' ');
    if (node.is_leaf) {
        os << "* class=" << classes_[node.predicted_class_index] << "  n=" << node.n_samples
           << "  impurity=" << format_stat(node.impurity, 3) << "\n";
        return;
    }
    os << predictor_names_[node.feature_index] << " <= " << format_stat(node.threshold, 3) << "?  (n=" << node.n_samples
       << ", impurity=" << format_stat(node.impurity, 3) << ")\n";
    dump_node(os, node.left, indent + 1);
    dump_node(os, node.right, indent + 1);
}

std::string DecisionTreeClassifier::summary() const {
    std::ostringstream out;
    out << "Call:\ntree(formula = " << formula_.text() << ", criterion = \""
        << (options_.criterion == SplitCriterion::Gini ? "gini" : "entropy") << "\", max_depth = " << options_.max_depth
        << ")\n\n";

    out << "Number of Classes: " << classes_.size() << "\n";
    out << "Levels: ";
    for (const auto& c : classes_) out << c << " ";
    out << "\n\n";

    out << "Tree: " << node_count() << " nodes, " << leaf_count() << " leaves, depth " << depth() << "\n\n";
    dump_node(out, 0, 0);

    out << "\nFeature importance:\n";
    const auto importance = feature_importance();
    for (std::size_t j = 0; j < predictor_names_.size(); ++j)
        out << "  " << predictor_names_[j] << ": " << format_stat(importance[j], 4) << "\n";

    out << "\nTraining accuracy: " << format_stat(training_accuracy() * 100.0, 2) << "%  (" << observations_
        << " observations)\n";

    return out.str();
}

void DecisionTreeClassifier::print_summary(std::ostream& os) const { os << summary(); }

void DecisionTreeClassifier::print_summary() const { print_summary(std::cout); }

std::vector<std::string> DecisionTreeClassifier::predict(const dstruct::DataFrame& newdata) const {
    return predict_detail(newdata).class_label;
}

DecisionTreeClassifierPrediction DecisionTreeClassifier::predict_detail(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "DecisionTreeClassifier::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);

    DecisionTreeClassifierPrediction result;
    result.class_label.assign(newdata.nrows(), "");
    result.probability.assign(newdata.nrows(), {});
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        std::vector<double> row(dm.X.cols());
        for (std::size_t j = 0; j < dm.X.cols(); ++j) row[j] = dm.X(i, j);
        const auto& leaf = nodes_[classify_row(row)];

        std::vector<double> probability(classes_.size());
        for (std::size_t c = 0; c < classes_.size(); ++c)
            probability[c] = leaf.class_counts[c] / static_cast<double>(leaf.n_samples);

        const auto original_row              = dm.used_row_indices[i];
        result.class_label[original_row]      = classes_[leaf.predicted_class_index];
        result.probability[original_row]      = std::move(probability);
    }
    return result;
}

plot::RPlot DecisionTreeClassifier::plot_classification(const dstruct::DataFrame& data,
                                                               const std::string& x_feature,
                                                               const std::string& y_feature) const {
    const auto predictions = predict(data);
    const auto& response   = design_.response_name;

    auto plot = plot::RPlot::create();
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

    plot.title("Decision Tree Classification").x_label(x_feature).y_label(y_feature);
    return plot;
}

plot::RPlot DecisionTreeClassifier::plot_decision_regions(const std::string& x_feature,
                                                                 const std::string& y_feature,
                                                                 std::size_t grid_resolution) const {
    if (predictor_names_.size() != 2)
        throw std::invalid_argument("DecisionTreeClassifier::plot_decision_regions: model must have exactly 2 "
                                    "predictors");
    std::size_t xi = 2, yi = 2;
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_names_[j] == x_feature) xi = j;
        if (predictor_names_[j] == y_feature) yi = j;
    }
    if (xi == 2 || yi == 2)
        throw std::invalid_argument("DecisionTreeClassifier::plot_decision_regions: x_feature/y_feature must match "
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

    auto plot = plot::RPlot::create();
    std::vector<std::vector<double>> grid_x(classes_.size()), grid_y(classes_.size());
    for (std::size_t gx = 0; gx < grid_resolution; ++gx) {
        for (std::size_t gy = 0; gy < grid_resolution; ++gy) {
            const double x = x_min + (x_max - x_min) * static_cast<double>(gx) / static_cast<double>(grid_resolution - 1);
            const double y = y_min + (y_max - y_min) * static_cast<double>(gy) / static_cast<double>(grid_resolution - 1);
            std::vector<double> point(2);
            point[xi] = x;
            point[yi] = y;
            const auto predicted_class = nodes_[classify_row(point)].predicted_class_index;
            grid_x[predicted_class].push_back(x);
            grid_y[predicted_class].push_back(y);
        }
    }
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        auto color = kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))];
        color.r    = static_cast<std::uint8_t>(std::min(255, color.r + (255 - color.r) * 3 / 4));
        color.g    = static_cast<std::uint8_t>(std::min(255, color.g + (255 - color.g) * 3 / 4));
        color.b    = static_cast<std::uint8_t>(std::min(255, color.b + (255 - color.b) * 3 / 4));
        plot.points(grid_x[c], grid_y[c], classes_[c] + " region", color, 3.0);
    }

    plot.title("Decision Tree Regions").x_label(x_feature).y_label(y_feature);
    return plot;
}

} // namespace datamunge::stats
