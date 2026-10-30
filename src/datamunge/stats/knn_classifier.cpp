/// @file knn_classifier.cpp
/// @brief KNNClassifier: fitting, exact leave-one-out evaluation, prediction and plots, and
///        the large-sample k-NN risk.

#include <datamunge/stats/knn_classifier.hpp>

#include <datamunge/plot/class_regions.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

/// @brief Class colours, cycled.
const graphics::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

} // namespace

KNNClassifier::KNNClassifier(const dstruct::DataFrame& data, const std::string& formula, KNNClassifierOptions options)
    : formula_(formula), options_(options) {
    fit(data);
}

void KNNClassifier::fit(const dstruct::DataFrame& data) {
    if (options_.k == 0) throw std::invalid_argument("KNNClassifier: k must be at least 1");

    design_        = formula_.resolve(data, ResponseKind::Categorical);
    const auto cd  = design_.build_classification_matrix(data);
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
        throw std::invalid_argument("KNNClassifier: response must have at least 2 distinct classes, found "
                                    + std::to_string(classes_.size()));

    feature_min_.assign(p, std::numeric_limits<double>::infinity());
    feature_max_.assign(p, -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            feature_min_[j] = std::min(feature_min_[j], cd.X(i, j));
            feature_max_[j] = std::max(feature_max_[j], cd.X(i, j));
        }

    scaling_        = detail::knn_scaling(cd.X, options_.standardize);
    training_X_   = cd.X;
    multipliers_  = detail::knn_multipliers(scaling_);

    training_class_index_.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        training_class_index_[i] = static_cast<std::size_t>(
            std::lower_bound(classes_.begin(), classes_.end(), cd.labels[i]) - classes_.begin());

    fitted_classes_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto mult   = detail::knn_loo_multipliers(scaling_, training_X_, i, options_.standardize);
        const auto shares = vote_shares(row_of(training_X_, i), i, mult);
        fitted_classes_[i] = classes_[static_cast<std::size_t>(std::max_element(shares.begin(), shares.end())
                                                               - shares.begin())];
    }
}

std::vector<double> KNNClassifier::row_of(const linalg::DenseMatrix<double>& X, std::size_t row) const {
    std::vector<double> v(X.cols());
    for (std::size_t j = 0; j < X.cols(); ++j) v[j] = X(row, j);
    return v;
}

std::vector<double> KNNClassifier::vote_shares(const std::vector<double>& query, std::size_t exclude_row,
                                               const std::vector<double>& mult) const {
    const auto nearest = detail::knn_nearest(training_X_, query, options_.metric, options_.k,
                                             mult.empty() ? multipliers_ : mult, exclude_row);
    std::vector<double> votes(classes_.size(), 0.0);
    for (const auto& [d, row] : nearest) votes[training_class_index_[row]] += detail::knn_weight(d, options_.weighted);
    const double total = std::accumulate(votes.begin(), votes.end(), 0.0);
    if (total > 0.0)
        for (double& v : votes) v /= total;
    return votes;
}

std::vector<double> KNNClassifier::loo_accuracy_path(std::size_t max_k) const {
    if (max_k == 0) throw std::invalid_argument("KNNClassifier::loo_accuracy_path: max_k must be at least 1");
    const std::size_t n = training_X_.rows();
    max_k = std::min(max_k, n - 1);
    std::vector<double> correct(max_k, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const auto nearest = detail::knn_nearest(training_X_, row_of(training_X_, i), options_.metric, max_k,
                                                 detail::knn_loo_multipliers(scaling_, training_X_, i,
                                                                             options_.standardize),
                                                 i);
        std::vector<double> votes(classes_.size(), 0.0);
        for (std::size_t k = 0; k < nearest.size(); ++k) {
            votes[training_class_index_[nearest[k].second]] += detail::knn_weight(nearest[k].first, options_.weighted);
            const auto best = static_cast<std::size_t>(std::max_element(votes.begin(), votes.end()) - votes.begin());
            if (best == training_class_index_[i]) correct[k] += 1.0;
        }
    }
    for (double& c : correct) c /= static_cast<double>(n);
    return correct;
}

KNNNeighbors KNNClassifier::kneighbors(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "KNNClassifier::kneighbors");
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

double KNNClassifier::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
        if (fitted_classes_[i] == training_labels_[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> KNNClassifier::confusion_matrix() const {
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

std::string KNNClassifier::summary() const {
    std::ostringstream out;
    out << "Call:\nknn(formula = " << formula_.text() << ", k = " << options_.k << ", metric = "
        << (options_.metric == DistanceMetric::Euclidean ? "euclidean" : "manhattan")
        << ", weighted = " << (options_.weighted ? "true" : "false") << ")\n\n";

    out << "Number of Classes: " << classes_.size() << "\n";
    out << "Levels: ";
    for (const auto& c : classes_) out << c << " ";
    out << "\n\n";

    out << "Leave-one-out accuracy: " << format_stat(training_accuracy() * 100.0, 2) << "%  (" << observations_
        << " observations)\n";

    return out.str();
}

void KNNClassifier::print_summary(std::ostream& os) const { os << summary(); }

void KNNClassifier::print_summary() const { print_summary(std::cout); }

std::vector<std::string> KNNClassifier::predict(const dstruct::DataFrame& newdata) const {
    return predict_detail(newdata).class_label;
}

KNNClassifierPrediction KNNClassifier::predict_detail(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "KNNClassifier::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);

    KNNClassifierPrediction result;
    result.class_label.assign(newdata.nrows(), "");
    result.vote_share.assign(newdata.nrows(), {});
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        std::vector<double> query(dm.X.cols());
        for (std::size_t j = 0; j < dm.X.cols(); ++j) query[j] = dm.X(i, j);

        const auto shares = vote_shares(query);
        const auto best =
            static_cast<std::size_t>(std::max_element(shares.begin(), shares.end()) - shares.begin());

        const auto original_row          = dm.used_row_indices[i];
        result.class_label[original_row] = classes_[best];
        result.vote_share[original_row]  = shares;
    }
    return result;
}

graphics::RPlot KNNClassifier::plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                      const std::string& y_feature) const {
    const auto  predictions = predict(data);
    const auto& response    = design_.response_name;

    auto plot = graphics::RPlot::create();
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

    plot.title("KNN Classification").x_label(x_feature).y_label(y_feature);
    return plot;
}

graphics::RPlot KNNClassifier::plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                     std::size_t grid_resolution,
                                                     std::optional<std::array<double, 4>> limits) const {
    if (predictor_names_.size() != 2)
        throw std::invalid_argument("KNNClassifier::plot_decision_regions: model must have exactly 2 predictors");
    std::size_t xi = 2, yi = 2;
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_names_[j] == x_feature) xi = j;
        if (predictor_names_[j] == y_feature) yi = j;
    }
    if (xi == 2 || yi == 2 || xi == yi)
        throw std::invalid_argument("KNNClassifier::plot_decision_regions: x_feature/y_feature must match the "
                                    "model's two predictors (" + predictor_names_[0] + ", " + predictor_names_[1]
                                    + ")");
    grid_resolution = std::max<std::size_t>(grid_resolution, 2);

    // Same box as RPlot's default padding: 5% in x, 10% in y.
    const double x_pad = std::max(1e-6, (feature_max_[xi] - feature_min_[xi]) * 0.05);
    const double y_pad = std::max(1e-6, (feature_max_[yi] - feature_min_[yi]) * 0.10);
    const auto box = limits.value_or(std::array<double, 4>{feature_min_[xi] - x_pad, feature_max_[xi] + x_pad,
                                                           feature_min_[yi] - y_pad, feature_max_[yi] + y_pad});
    const double x_lo = box[0], x_hi = box[1], y_lo = box[2], y_hi = box[3];
    if (!(x_hi > x_lo && y_hi > y_lo))
        throw std::invalid_argument("KNNClassifier::plot_decision_regions: limits must be increasing");

    auto plot = graphics::RPlot::create();
    std::vector<graphics::RGB> fills;
    for (std::size_t c = 0; c < classes_.size(); ++c)
        fills.push_back(graphics::tint(kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))], 0.8));
    std::vector<double> query(2);
    graphics::add_class_regions(
        plot,
        [&](double x, double y) {
            query[xi] = x;
            query[yi] = y;
            const auto shares = vote_shares(query);
            return static_cast<std::size_t>(std::max_element(shares.begin(), shares.end()) - shares.begin());
        },
        {x_lo, x_hi, y_lo, y_hi}, grid_resolution, fills);

    for (std::size_t c = 0; c < classes_.size(); ++c) {
        std::vector<double> xs, ys;
        for (std::size_t i = 0; i < training_X_.rows(); ++i) {
            if (training_class_index_[i] != c) continue;
            xs.push_back(training_X_(i, xi));
            ys.push_back(training_X_(i, yi));
        }
        plot.points(xs, ys, classes_[c], kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
    }
    plot.title("k-NN decision regions (k = " + std::to_string(options_.k) + ")").x_label(x_feature).y_label(y_feature);
    return plot;
}

double knn_asymptotic_risk(double eta, std::size_t k) {
    if (!(eta >= 0.0 && eta <= 1.0)) throw std::invalid_argument("knn_asymptotic_risk: eta must be in [0, 1]");
    if (k == 0) throw std::invalid_argument("knn_asymptotic_risk: k must be at least 1");
    double risk = 0.0;
    for (std::size_t b = 0; b <= k; ++b) {
        // P(B = b) for B ~ Bin(k, eta), via logs to stay finite for large k.
        const double log_choose = std::lgamma(static_cast<double>(k) + 1.0) - std::lgamma(static_cast<double>(b) + 1.0)
                                  - std::lgamma(static_cast<double>(k - b) + 1.0);
        double pb = 0.0;
        if (eta == 0.0) pb = b == 0 ? 1.0 : 0.0;
        else if (eta == 1.0) pb = b == k ? 1.0 : 0.0;
        else pb = std::exp(log_choose + static_cast<double>(b) * std::log(eta) + static_cast<double>(k - b) * std::log1p(-eta));
        if (2 * b < k) risk += eta * pb;
        else if (2 * b > k) risk += (1.0 - eta) * pb;
        else risk += 0.5 * pb;
    }
    return risk;
}

} // namespace datamunge::stats
