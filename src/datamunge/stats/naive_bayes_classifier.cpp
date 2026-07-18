#include <datamunge/stats/naive_bayes_classifier.hpp>

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

constexpr double kLog2Pi = 1.8378770664093453;

std::string format_stat(double v, int precision = 4) {
    if (std::isnan(v)) return "NaN";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

const plot::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

double log_gaussian_pdf(double x, double mean, double variance) {
    const double d = x - mean;
    return -0.5 * kLog2Pi - 0.5 * std::log(variance) - 0.5 * d * d / variance;
}

} // namespace

NaiveBayesClassifier::NaiveBayesClassifier(const dstruct::DataFrame& data, const std::string& formula,
                                           NaiveBayesClassifierOptions options)
    : formula_(formula) {
    options_ = options;
    fit(data);
}

std::vector<std::size_t> NaiveBayesClassifier::filter_complete_rows(const dstruct::DataFrame& frame,
                                                                     bool require_response) const {
    std::vector<std::size_t> kept;
    for (std::size_t row = 0; row < frame.nrows(); ++row) {
        bool ok = true;
        if (require_response && !frame.optional_string_at(design_.response_name, row)) ok = false;
        if (ok) {
            for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
                if (predictor_is_categorical_[j]) {
                    if (!frame.optional_string_at(predictor_names_[j], row)) {
                        ok = false;
                        break;
                    }
                } else {
                    if (!frame.optional_double_at(predictor_names_[j], row)) {
                        ok = false;
                        break;
                    }
                }
            }
        }
        if (ok) kept.push_back(row);
    }
    return kept;
}

void NaiveBayesClassifier::fit(const dstruct::DataFrame& data) {
    design_ = formula_.resolve(data, ResponseKind::Categorical);

    predictor_names_.clear();
    predictor_is_categorical_.clear();
    std::set<std::string> seen;
    for (const auto& col : design_.columns) {
        if (col.kind != ResolvedColumnKind::Numeric && col.kind != ResolvedColumnKind::CategoricalDummy)
            throw std::invalid_argument("NaiveBayesClassifier: formula terms must be plain column references (no "
                                        "functions, interactions, or transforms)");
        if (!seen.insert(col.source_column).second) continue;
        predictor_names_.push_back(col.source_column);
        predictor_is_categorical_.push_back(col.kind == ResolvedColumnKind::CategoricalDummy);
    }
    if (predictor_names_.empty())
        throw std::invalid_argument("NaiveBayesClassifier: at least one predictor is required");

    const auto kept_rows = filter_complete_rows(data, /*require_response=*/true);
    const std::size_t n  = kept_rows.size();
    if (n == 0)
        throw std::runtime_error("NaiveBayesClassifier: no complete rows remain after dropping missing values");
    observations_ = n;

    training_labels_.resize(n);
    for (std::size_t i = 0; i < n; ++i) training_labels_[i] = *data.optional_string_at(design_.response_name, kept_rows[i]);

    classes_.clear();
    for (const auto& label : training_labels_)
        if (std::find(classes_.begin(), classes_.end(), label) == classes_.end()) classes_.push_back(label);
    std::sort(classes_.begin(), classes_.end());
    if (classes_.size() < 2)
        throw std::invalid_argument("NaiveBayesClassifier: response must have at least 2 distinct classes, found "
                                    + std::to_string(classes_.size()));
    const std::size_t K = classes_.size();

    std::vector<std::size_t> class_index(n);
    std::vector<double>      class_count(K, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        class_index[i] = static_cast<std::size_t>(std::lower_bound(classes_.begin(), classes_.end(),
                                                                    training_labels_[i])
                                                  - classes_.begin());
        class_count[class_index[i]] += 1.0;
    }
    class_priors_.resize(K);
    for (std::size_t k = 0; k < K; ++k) class_priors_[k] = class_count[k] / static_cast<double>(n);

    const std::size_t p = predictor_names_.size();
    numeric_slot_.assign(p, -1);
    categorical_slot_.assign(p, -1);
    numeric_mean_.clear();
    numeric_variance_.clear();
    numeric_min_.clear();
    numeric_max_.clear();
    categorical_levels_.clear();
    categorical_log_prob_.clear();

    for (std::size_t j = 0; j < p; ++j) {
        if (predictor_is_categorical_[j]) {
            categorical_slot_[j] = static_cast<int>(categorical_levels_.size());
            categorical_levels_.push_back(design_.categorical_levels.at(predictor_names_[j]));
        } else {
            numeric_slot_[j] = static_cast<int>(numeric_mean_.size());
            numeric_mean_.emplace_back(K, 0.0);
            numeric_variance_.emplace_back(K, 0.0);
            numeric_min_.push_back(std::numeric_limits<double>::infinity());
            numeric_max_.push_back(-std::numeric_limits<double>::infinity());
        }
    }

    double global_max_var = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        if (predictor_is_categorical_[j]) continue;
        const int slot = numeric_slot_[j];

        std::vector<double> values(n);
        for (std::size_t i = 0; i < n; ++i) {
            values[i]           = *data.optional_double_at(predictor_names_[j], kept_rows[i]);
            numeric_min_[slot]  = std::min(numeric_min_[slot], values[i]);
            numeric_max_[slot]  = std::max(numeric_max_[slot], values[i]);
        }
        const double overall_mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(n);
        double        overall_var  = 0.0;
        for (const double v : values) overall_var += (v - overall_mean) * (v - overall_mean);
        overall_var /= static_cast<double>(n);
        global_max_var = std::max(global_max_var, overall_var);

        for (std::size_t k = 0; k < K; ++k) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                if (class_index[i] == k) sum += values[i];
            const double mean_k = sum / class_count[k];
            double        var_k  = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                if (class_index[i] == k) var_k += (values[i] - mean_k) * (values[i] - mean_k);
            var_k /= class_count[k];
            numeric_mean_[slot][k]     = mean_k;
            numeric_variance_[slot][k] = var_k;
        }
    }
    const double epsilon = options_.var_smoothing * global_max_var;
    for (auto& per_class_var : numeric_variance_)
        for (double& v : per_class_var) v += epsilon;

    for (std::size_t j = 0; j < p; ++j) {
        if (!predictor_is_categorical_[j]) continue;
        const int   slot   = categorical_slot_[j];
        const auto& levels = categorical_levels_[slot];
        const std::size_t L = levels.size();

        std::vector<std::vector<double>> counts(K, std::vector<double>(L, 0.0));
        for (std::size_t i = 0; i < n; ++i) {
            const auto  value     = *data.optional_string_at(predictor_names_[j], kept_rows[i]);
            const auto  it        = std::lower_bound(levels.begin(), levels.end(), value);
            const auto  level_idx = static_cast<std::size_t>(it - levels.begin());
            counts[class_index[i]][level_idx] += 1.0;
        }

        categorical_log_prob_.emplace_back(K, std::vector<double>(L, 0.0));
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t l = 0; l < L; ++l) {
                const double prob =
                    (counts[k][l] + options_.laplace_smoothing)
                    / (class_count[k] + options_.laplace_smoothing * static_cast<double>(L));
                categorical_log_prob_.back()[k][l] = std::log(prob);
            }
    }

    fitted_classes_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto lp = log_posteriors_for_row(data, kept_rows[i]);
        const auto best =
            static_cast<std::size_t>(std::max_element(lp.begin(), lp.end()) - lp.begin());
        fitted_classes_[i] = classes_[best];
    }
}

std::vector<double> NaiveBayesClassifier::log_posteriors_for_row(const dstruct::DataFrame& frame,
                                                                  std::size_t row) const {
    const std::size_t K = classes_.size();
    std::vector<double> lp(K);
    for (std::size_t k = 0; k < K; ++k) lp[k] = std::log(class_priors_[k]);

    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_is_categorical_[j]) {
            const auto  value     = *frame.optional_string_at(predictor_names_[j], row);
            const int   slot      = categorical_slot_[j];
            const auto& levels    = categorical_levels_[slot];
            const auto  it        = std::lower_bound(levels.begin(), levels.end(), value);
            const auto  level_idx = static_cast<std::size_t>(it - levels.begin());
            for (std::size_t k = 0; k < K; ++k) lp[k] += categorical_log_prob_[slot][k][level_idx];
        } else {
            const double value = *frame.optional_double_at(predictor_names_[j], row);
            const int    slot  = numeric_slot_[j];
            for (std::size_t k = 0; k < K; ++k)
                lp[k] += log_gaussian_pdf(value, numeric_mean_[slot][k], numeric_variance_[slot][k]);
        }
    }
    return lp;
}

double NaiveBayesClassifier::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
        if (fitted_classes_[i] == training_labels_[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> NaiveBayesClassifier::confusion_matrix() const {
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

std::string NaiveBayesClassifier::summary() const {
    std::ostringstream out;
    out << "Call:\nnaiveBayes(formula = " << formula_.text() << ")\n\n";

    out << "Number of Classes: " << classes_.size() << "\n";
    out << "Levels: ";
    for (const auto& c : classes_) out << c << " ";
    out << "\n\n";

    out << "Class priors:\n";
    for (std::size_t k = 0; k < classes_.size(); ++k)
        out << "  " << classes_[k] << ": " << format_stat(class_priors_[k], 4) << "\n";

    out << "\nPredictors:\n";
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_is_categorical_[j]) {
            const int   slot   = categorical_slot_[j];
            const auto& levels = categorical_levels_[slot];
            out << "  " << predictor_names_[j] << " (categorical, levels: ";
            for (const auto& l : levels) out << l << " ";
            out << ")\n";
            for (std::size_t k = 0; k < classes_.size(); ++k) {
                out << "    " << classes_[k] << ": ";
                for (std::size_t l = 0; l < levels.size(); ++l)
                    out << levels[l] << "=" << format_stat(std::exp(categorical_log_prob_[slot][k][l]), 3) << " ";
                out << "\n";
            }
        } else {
            const int slot = numeric_slot_[j];
            out << "  " << predictor_names_[j] << " (numeric, mean [sd] per class): ";
            for (std::size_t k = 0; k < classes_.size(); ++k)
                out << classes_[k] << "=" << format_stat(numeric_mean_[slot][k], 3) << " ["
                    << format_stat(std::sqrt(numeric_variance_[slot][k]), 3) << "] ";
            out << "\n";
        }
    }

    out << "\nTraining accuracy: " << format_stat(training_accuracy() * 100.0, 2) << "%  (" << observations_
        << " observations)\n";

    return out.str();
}

void NaiveBayesClassifier::print_summary(std::ostream& os) const { os << summary(); }

void NaiveBayesClassifier::print_summary() const { print_summary(std::cout); }

std::vector<std::string> NaiveBayesClassifier::predict(const dstruct::DataFrame& newdata) const {
    return predict_detail(newdata).class_label;
}

NaiveBayesClassifierPrediction NaiveBayesClassifier::predict_detail(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "NaiveBayesClassifier::predict");
    const auto kept_rows = filter_complete_rows(newdata, /*require_response=*/false);

    NaiveBayesClassifierPrediction result;
    result.class_label.assign(newdata.nrows(), "");
    result.probability.assign(newdata.nrows(), {});
    const std::size_t K = classes_.size();

    for (const auto row : kept_rows) {
        const auto   lp     = log_posteriors_for_row(newdata, row);
        const double max_lp = *std::max_element(lp.begin(), lp.end());

        std::vector<double> prob(K);
        double               sum = 0.0;
        for (std::size_t k = 0; k < K; ++k) {
            prob[k] = std::exp(lp[k] - max_lp);
            sum += prob[k];
        }
        for (double& v : prob) v /= sum;

        const auto best = static_cast<std::size_t>(std::max_element(prob.begin(), prob.end()) - prob.begin());
        result.class_label[row] = classes_[best];
        result.probability[row] = std::move(prob);
    }
    return result;
}

plot::ScatterPlot NaiveBayesClassifier::plot_classification(const dstruct::DataFrame& data,
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

    plot.title("Naive Bayes Classification").x_label(x_feature).y_label(y_feature);
    return plot;
}

plot::ScatterPlot NaiveBayesClassifier::plot_decision_regions(const std::string& x_feature,
                                                               const std::string& y_feature,
                                                               std::size_t grid_resolution) const {
    if (predictor_names_.size() != 2)
        throw std::invalid_argument("NaiveBayesClassifier::plot_decision_regions: model must have exactly 2 "
                                    "predictors");
    std::size_t xi = 2, yi = 2;
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_names_[j] == x_feature) xi = j;
        if (predictor_names_[j] == y_feature) yi = j;
    }
    if (xi == 2 || yi == 2)
        throw std::invalid_argument("NaiveBayesClassifier::plot_decision_regions: x_feature/y_feature must match "
                                    "the model's two predictors (" + predictor_names_[0] + ", " + predictor_names_[1]
                                    + ")");
    if (predictor_is_categorical_[xi] || predictor_is_categorical_[yi])
        throw std::invalid_argument("NaiveBayesClassifier::plot_decision_regions: both predictors must be numeric");

    double x_min = numeric_min_[numeric_slot_[xi]], x_max = numeric_max_[numeric_slot_[xi]];
    double y_min = numeric_min_[numeric_slot_[yi]], y_max = numeric_max_[numeric_slot_[yi]];
    const double x_pad = std::max(1e-6, (x_max - x_min) * 0.1);
    const double y_pad = std::max(1e-6, (y_max - y_min) * 0.1);
    x_min -= x_pad;
    x_max += x_pad;
    y_min -= y_pad;
    y_max += y_pad;

    std::vector<double> col_x, col_y;
    col_x.reserve(grid_resolution * grid_resolution);
    col_y.reserve(grid_resolution * grid_resolution);
    for (std::size_t gx = 0; gx < grid_resolution; ++gx) {
        for (std::size_t gy = 0; gy < grid_resolution; ++gy) {
            col_x.push_back(x_min + (x_max - x_min) * static_cast<double>(gx) / static_cast<double>(grid_resolution - 1));
            col_y.push_back(y_min + (y_max - y_min) * static_cast<double>(gy) / static_cast<double>(grid_resolution - 1));
        }
    }
    dstruct::DataFrame grid;
    grid.add_column(x_feature, col_x);
    grid.add_column(y_feature, col_y);

    const auto detail = predict_detail(grid);

    std::vector<std::vector<double>> grid_x(classes_.size()), grid_y(classes_.size());
    for (std::size_t i = 0; i < detail.class_label.size(); ++i) {
        if (detail.class_label[i].empty()) continue;
        const auto it = std::lower_bound(classes_.begin(), classes_.end(), detail.class_label[i]);
        const auto c  = static_cast<std::size_t>(it - classes_.begin());
        grid_x[c].push_back(col_x[i]);
        grid_y[c].push_back(col_y[i]);
    }

    auto plot = plot::ScatterPlot::create();
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        auto color = kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))];
        color.r    = static_cast<std::uint8_t>(std::min(255, color.r + (255 - color.r) * 3 / 4));
        color.g    = static_cast<std::uint8_t>(std::min(255, color.g + (255 - color.g) * 3 / 4));
        color.b    = static_cast<std::uint8_t>(std::min(255, color.b + (255 - color.b) * 3 / 4));
        plot.points(grid_x[c], grid_y[c], classes_[c] + " region", color, 3.0);
    }

    plot.title("Naive Bayes Decision Regions").x_label(x_feature).y_label(y_feature);
    return plot;
}

} // namespace datamunge::stats
