#include <datamunge/stats/knn_classifier.hpp>

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

KNNClassifier::KNNClassifier(const dstruct::DataFrame& data, const std::string& formula, KNNClassifierOptions options)
    : formula_(formula) {
    options_ = options;
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

    feature_mean_.assign(p, 0.0);
    feature_scale_.assign(p, 1.0);
    if (options_.standardize) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum += cd.X(i, j);
            feature_mean_[j] = sum / static_cast<double>(n);
        }
        for (std::size_t j = 0; j < p; ++j) {
            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double d = cd.X(i, j) - feature_mean_[j];
                ss += d * d;
            }
            const double scale = std::sqrt(ss / static_cast<double>(n));
            feature_scale_[j]  = (scale < 1e-12) ? 1.0 : scale;
        }
    }

    training_X_std_ = linalg::DenseMatrix<double>(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) training_X_std_(i, j) = (cd.X(i, j) - feature_mean_[j]) / feature_scale_[j];

    training_class_index_.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        training_class_index_[i] = static_cast<std::size_t>(std::lower_bound(classes_.begin(), classes_.end(),
                                                                              cd.labels[i])
                                                            - classes_.begin());

    fitted_classes_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> query(p);
        for (std::size_t j = 0; j < p; ++j) query[j] = training_X_std_(i, j);
        const auto shares = vote_shares(query, /*exclude_row=*/i);
        const auto best =
            static_cast<std::size_t>(std::max_element(shares.begin(), shares.end()) - shares.begin());
        fitted_classes_[i] = classes_[best];
    }
}

std::vector<double> KNNClassifier::vote_shares(const std::vector<double>& query_std, std::size_t exclude_row) const {
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

    std::vector<double> votes(classes_.size(), 0.0);
    for (std::size_t idx = 0; idx < kk; ++idx) {
        const double      d   = distances[idx].first;
        const std::size_t row = distances[idx].second;
        const double       w   = options_.weighted ? 1.0 / (d + 1e-12) : 1.0;
        votes[training_class_index_[row]] += w;
    }
    const double total = std::accumulate(votes.begin(), votes.end(), 0.0);
    if (total > 0.0)
        for (double& v : votes) v /= total;
    return votes;
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
        for (std::size_t j = 0; j < dm.X.cols(); ++j) query[j] = (dm.X(i, j) - feature_mean_[j]) / feature_scale_[j];

        const auto shares = vote_shares(query);
        const auto best =
            static_cast<std::size_t>(std::max_element(shares.begin(), shares.end()) - shares.begin());

        const auto original_row          = dm.used_row_indices[i];
        result.class_label[original_row] = classes_[best];
        result.vote_share[original_row]  = shares;
    }
    return result;
}

plot::RPlot KNNClassifier::plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                      const std::string& y_feature) const {
    const auto  predictions = predict(data);
    const auto& response    = design_.response_name;

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

    plot.title("KNN Classification").x_label(x_feature).y_label(y_feature);
    return plot;
}

plot::RPlot KNNClassifier::plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                        std::size_t grid_resolution) const {
    if (predictor_names_.size() != 2)
        throw std::invalid_argument("KNNClassifier::plot_decision_regions: model must have exactly 2 predictors");
    std::size_t xi = 2, yi = 2;
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_names_[j] == x_feature) xi = j;
        if (predictor_names_[j] == y_feature) yi = j;
    }
    if (xi == 2 || yi == 2)
        throw std::invalid_argument("KNNClassifier::plot_decision_regions: x_feature/y_feature must match the "
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

    auto plot = plot::RPlot::create();
    std::vector<std::vector<double>> grid_x(classes_.size()), grid_y(classes_.size());
    for (std::size_t gx = 0; gx < grid_resolution; ++gx) {
        for (std::size_t gy = 0; gy < grid_resolution; ++gy) {
            const double x = x_min + (x_max - x_min) * static_cast<double>(gx) / static_cast<double>(grid_resolution - 1);
            const double y = y_min + (y_max - y_min) * static_cast<double>(gy) / static_cast<double>(grid_resolution - 1);

            std::vector<double> point(2);
            point[xi] = x;
            point[yi] = y;
            std::vector<double> query(2);
            query[0] = (point[0] - feature_mean_[0]) / feature_scale_[0];
            query[1] = (point[1] - feature_mean_[1]) / feature_scale_[1];

            const auto shares = vote_shares(query);
            const auto predicted_class =
                static_cast<std::size_t>(std::max_element(shares.begin(), shares.end()) - shares.begin());
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

    plot.title("KNN Decision Regions (k=" + std::to_string(options_.k) + ")").x_label(x_feature).y_label(y_feature);
    return plot;
}

} // namespace datamunge::stats
