/// @file naive_bayes_classifier.cpp
/// @brief NaiveBayesClassifier: fitting, exact leave-one-out downdates, prediction, parameter
///        tables and plots.

#include <datamunge/stats/naive_bayes_classifier.hpp>

#include <datamunge/plot/class_regions.hpp>

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

/// @brief Helpers private to this file.
namespace {

/// @brief \f$\log(2\pi)\f$.
constexpr double kLog2Pi = 1.8378770664093453;

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

/// @brief Colour of class @p c.
/// @param c  Class index.
/// @return Its colour.
graphics::RGB class_color(std::size_t c) { return kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]; }

/// @brief Log density of \f$N(\text{mean},\text{variance})\f$ at @p x.
/// @param x         Point.
/// @param mean      Mean.
/// @param variance  Variance (positive).
/// @return \f$\log\varphi\f$.
double log_gaussian_pdf(double x, double mean, double variance) {
    const double d = x - mean;
    return -0.5 * kLog2Pi - 0.5 * std::log(variance) - 0.5 * d * d / variance;
}

/// @brief Index of the largest entry (the first on ties).
/// @param v  Values.
/// @return Index.
std::size_t argmax(const std::vector<double>& v) {
    return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

} // namespace

NaiveBayesClassifier::NaiveBayesClassifier(const dstruct::DataFrame& data, const std::string& formula,
                                           NaiveBayesClassifierOptions options)
    : formula_(formula), options_(std::move(options)) {
    fit(data);
}

std::vector<std::size_t> NaiveBayesClassifier::filter_complete_rows(const dstruct::DataFrame& frame,
                                                                    bool require_response) const {
    std::vector<std::size_t> kept;
    for (std::size_t row = 0; row < frame.nrows(); ++row) {
        bool ok = !require_response || frame.optional_string_at(design_.response_name, row).has_value();
        for (std::size_t j = 0; ok && j < predictor_names_.size(); ++j)
            ok = predictor_is_categorical_[j] ? frame.optional_string_at(predictor_names_[j], row).has_value()
                                              : frame.optional_double_at(predictor_names_[j], row).has_value();
        if (ok) kept.push_back(row);
    }
    return kept;
}

std::vector<double> NaiveBayesClassifier::encode_row(const dstruct::DataFrame& frame, std::size_t row) const {
    std::vector<double> x(predictor_names_.size());
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_is_categorical_[j]) {
            const auto  value  = *frame.optional_string_at(predictor_names_[j], row);
            const auto& levels = categorical_levels_[categorical_slot_[j]];
            const auto  it     = std::lower_bound(levels.begin(), levels.end(), value);
            if (it == levels.end() || *it != value)
                throw std::invalid_argument("NaiveBayesClassifier: level '" + value + "' of " + predictor_names_[j]
                                            + " was not seen in training");
            x[j] = static_cast<double>(it - levels.begin());
        } else {
            x[j] = *frame.optional_double_at(predictor_names_[j], row);
        }
    }
    return x;
}

void NaiveBayesClassifier::fit(const dstruct::DataFrame& data) {
    if (!(options_.laplace_smoothing >= 0.0))
        throw std::invalid_argument("NaiveBayesClassifier: laplace_smoothing must be non-negative");
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
    if (predictor_names_.empty()) throw std::invalid_argument("NaiveBayesClassifier: at least one predictor is required");

    const std::size_t p = predictor_names_.size();
    numeric_slot_.assign(p, -1);
    categorical_slot_.assign(p, -1);
    categorical_levels_.clear();
    std::size_t n_numeric = 0;
    for (std::size_t j = 0; j < p; ++j) {
        if (predictor_is_categorical_[j]) {
            categorical_slot_[j] = static_cast<int>(categorical_levels_.size());
            categorical_levels_.push_back(design_.categorical_levels.at(predictor_names_[j]));
        } else {
            numeric_slot_[j] = static_cast<int>(n_numeric++);
        }
    }

    const auto kept_rows = filter_complete_rows(data, /*require_response=*/true);
    const std::size_t n  = kept_rows.size();
    if (n == 0) throw std::runtime_error("NaiveBayesClassifier: no complete rows remain after dropping missing values");
    observations_ = n;

    training_labels_.resize(n);
    training_x_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        training_labels_[i] = *data.optional_string_at(design_.response_name, kept_rows[i]);
        training_x_[i]      = encode_row(data, kept_rows[i]);
    }
    classes_ = training_labels_;
    std::sort(classes_.begin(), classes_.end());
    classes_.erase(std::unique(classes_.begin(), classes_.end()), classes_.end());
    if (classes_.size() < 2)
        throw std::invalid_argument("NaiveBayesClassifier: response must have at least 2 distinct classes, found "
                                    + std::to_string(classes_.size()));
    const std::size_t K = classes_.size();

    training_class_.resize(n);
    class_count_.assign(K, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        training_class_[i] = static_cast<std::size_t>(
            std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]) - classes_.begin());
        class_count_[training_class_[i]] += 1.0;
    }
    if (options_.priors.empty()) {
        class_priors_.resize(K);
        for (std::size_t k = 0; k < K; ++k) class_priors_[k] = class_count_[k] / static_cast<double>(n);
    } else {
        if (options_.priors.size() != K)
            throw std::invalid_argument("NaiveBayesClassifier: priors must have one entry per class");
        double total = 0.0;
        for (const double v : options_.priors) {
            if (!(v > 0.0)) throw std::invalid_argument("NaiveBayesClassifier: priors must be positive");
            total += v;
        }
        if (std::abs(total - 1.0) > 1e-9) throw std::invalid_argument("NaiveBayesClassifier: priors must sum to 1");
        class_priors_ = options_.priors;
    }

    // Numeric predictors: per-class and overall means and centred sums of squares.
    numeric_mean_.assign(n_numeric, std::vector<double>(K, 0.0));
    numeric_ss_.assign(n_numeric, std::vector<double>(K, 0.0));
    numeric_all_mean_.assign(n_numeric, 0.0);
    numeric_all_ss_.assign(n_numeric, 0.0);
    numeric_min_.assign(n_numeric, std::numeric_limits<double>::infinity());
    numeric_max_.assign(n_numeric, -std::numeric_limits<double>::infinity());
    double max_var = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        if (predictor_is_categorical_[j]) continue;
        const auto s = static_cast<std::size_t>(numeric_slot_[j]);
        for (std::size_t i = 0; i < n; ++i) {
            const double v = training_x_[i][j];
            numeric_mean_[s][training_class_[i]] += v / class_count_[training_class_[i]];
            numeric_all_mean_[s] += v / static_cast<double>(n);
            numeric_min_[s] = std::min(numeric_min_[s], v);
            numeric_max_[s] = std::max(numeric_max_[s], v);
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double v = training_x_[i][j];
            numeric_ss_[s][training_class_[i]] += (v - numeric_mean_[s][training_class_[i]]) * (v - numeric_mean_[s][training_class_[i]]);
            numeric_all_ss_[s] += (v - numeric_all_mean_[s]) * (v - numeric_all_mean_[s]);
        }
        max_var = std::max(max_var, numeric_all_ss_[s] / static_cast<double>(n));
    }
    variance_floor_ = options_.var_smoothing * max_var;
    numeric_variance_.assign(n_numeric, std::vector<double>(K, 0.0));
    for (std::size_t s = 0; s < n_numeric; ++s)
        for (std::size_t k = 0; k < K; ++k) numeric_variance_[s][k] = numeric_ss_[s][k] / class_count_[k] + variance_floor_;

    // Categorical predictors: counts and smoothed log probabilities.
    categorical_count_.assign(categorical_levels_.size(), {});
    categorical_log_prob_.assign(categorical_levels_.size(), {});
    for (std::size_t j = 0; j < p; ++j) {
        if (!predictor_is_categorical_[j]) continue;
        const auto        s = static_cast<std::size_t>(categorical_slot_[j]);
        const std::size_t L = categorical_levels_[s].size();
        categorical_count_[s].assign(K, std::vector<double>(L, 0.0));
        for (std::size_t i = 0; i < n; ++i)
            categorical_count_[s][training_class_[i]][static_cast<std::size_t>(training_x_[i][j])] += 1.0;
        categorical_log_prob_[s].assign(K, std::vector<double>(L, 0.0));
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t l = 0; l < L; ++l)
                categorical_log_prob_[s][k][l] =
                    std::log((categorical_count_[s][k][l] + options_.laplace_smoothing)
                             / (class_count_[k] + options_.laplace_smoothing * static_cast<double>(L)));
    }

    fitted_classes_.resize(n);
    loo_fitted_classes_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        fitted_classes_[i]     = classes_[argmax(log_posteriors(training_x_[i]))];
        loo_fitted_classes_[i] = classes_[argmax(loo_log_posteriors(i))];
    }
}

std::vector<double> NaiveBayesClassifier::log_posteriors(const std::vector<double>& x) const {
    const std::size_t K = classes_.size();
    std::vector<double> lp(K);
    for (std::size_t k = 0; k < K; ++k) lp[k] = std::log(class_priors_[k]);
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_is_categorical_[j]) {
            const auto& table = categorical_log_prob_[categorical_slot_[j]];
            for (std::size_t k = 0; k < K; ++k) lp[k] += table[k][static_cast<std::size_t>(x[j])];
        } else {
            const auto s = static_cast<std::size_t>(numeric_slot_[j]);
            for (std::size_t k = 0; k < K; ++k) lp[k] += log_gaussian_pdf(x[j], numeric_mean_[s][k], numeric_variance_[s][k]);
        }
    }
    return lp;
}

std::vector<double> NaiveBayesClassifier::loo_log_posteriors(std::size_t i) const {
    const std::size_t K = classes_.size(), n = observations_;
    const std::size_t c = training_class_[i];
    const auto&       x = training_x_[i];
    const double      nc = class_count_[c] - 1.0;  // size of the left-out row's class without it
    std::vector<double> lp(K);
    for (std::size_t k = 0; k < K; ++k) {
        if (k == c && nc == 0.0) {
            lp[k] = -std::numeric_limits<double>::infinity();
            continue;
        }
        const double count = k == c ? nc : class_count_[k];
        lp[k] = std::log(options_.priors.empty() ? count / static_cast<double>(n - 1) : class_priors_[k]);
    }

    // Variance floor without row i: downdate each overall sum of squares.
    double max_var = 0.0;
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_is_categorical_[j]) continue;
        const auto   s   = static_cast<std::size_t>(numeric_slot_[j]);
        const double dev = x[j] - numeric_all_mean_[s];
        const double ss  = n > 1 ? std::max(0.0, numeric_all_ss_[s] - static_cast<double>(n) / static_cast<double>(n - 1) * dev * dev) : 0.0;
        max_var          = std::max(max_var, n > 1 ? ss / static_cast<double>(n - 1) : 0.0);
    }
    const double floor = options_.var_smoothing * max_var;

    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_is_categorical_[j]) {
            const auto        s     = static_cast<std::size_t>(categorical_slot_[j]);
            const auto        level = static_cast<std::size_t>(x[j]);
            const std::size_t L     = categorical_levels_[s].size();
            for (std::size_t k = 0; k < K; ++k) {
                if (!std::isfinite(lp[k])) continue;
                const double cnt   = categorical_count_[s][k][level] - (k == c ? 1.0 : 0.0);
                const double total = k == c ? nc : class_count_[k];
                lp[k] += std::log((cnt + options_.laplace_smoothing) / (total + options_.laplace_smoothing * static_cast<double>(L)));
            }
        } else {
            const auto s = static_cast<std::size_t>(numeric_slot_[j]);
            for (std::size_t k = 0; k < K; ++k) {
                if (!std::isfinite(lp[k])) continue;
                double mean = numeric_mean_[s][k], var = numeric_ss_[s][k] / class_count_[k];
                if (k == c) {
                    const double dev = x[j] - numeric_mean_[s][c];
                    mean             = (numeric_mean_[s][c] * class_count_[c] - x[j]) / nc;
                    var = std::max(0.0, numeric_ss_[s][c] - class_count_[c] / nc * dev * dev) / nc;
                }
                lp[k] += log_gaussian_pdf(x[j], mean, var + floor);
            }
        }
    }
    return lp;
}

double NaiveBayesClassifier::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i) correct += fitted_classes_[i] == training_labels_[i];
    return static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

double NaiveBayesClassifier::loo_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < loo_fitted_classes_.size(); ++i) correct += loo_fitted_classes_[i] == training_labels_[i];
    return static_cast<double>(correct) / static_cast<double>(loo_fitted_classes_.size());
}

linalg::DenseMatrix<double> NaiveBayesClassifier::confusion_matrix() const {
    const std::size_t k = classes_.size();
    linalg::DenseMatrix<double> matrix(k, k, 0.0);
    for (std::size_t i = 0; i < training_labels_.size(); ++i) {
        const auto predicted = std::lower_bound(classes_.begin(), classes_.end(), fitted_classes_[i]) - classes_.begin();
        matrix(training_class_[i], static_cast<std::size_t>(predicted)) += 1.0;
    }
    return matrix;
}

dstruct::DataFrame NaiveBayesClassifier::gaussian_frame() const {
    std::vector<std::string> predictor, cls;
    std::vector<double> count, mean, sd;
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_is_categorical_[j]) continue;
        const auto s = static_cast<std::size_t>(numeric_slot_[j]);
        for (std::size_t k = 0; k < classes_.size(); ++k) {
            predictor.push_back(predictor_names_[j]), cls.push_back(classes_[k]), count.push_back(class_count_[k]);
            mean.push_back(numeric_mean_[s][k]), sd.push_back(std::sqrt(numeric_variance_[s][k]));
        }
    }
    dstruct::DataFrame frame;
    frame.add_column("predictor", predictor);
    frame.add_column("class", cls);
    frame.add_column("n", count);
    frame.add_column("mean", mean);
    frame.add_column("sd", sd);
    return frame;
}

dstruct::DataFrame NaiveBayesClassifier::categorical_frame() const {
    std::vector<std::string> predictor, cls, level;
    std::vector<double> count, prob;
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (!predictor_is_categorical_[j]) continue;
        const auto s = static_cast<std::size_t>(categorical_slot_[j]);
        for (std::size_t k = 0; k < classes_.size(); ++k)
            for (std::size_t l = 0; l < categorical_levels_[s].size(); ++l) {
                predictor.push_back(predictor_names_[j]), cls.push_back(classes_[k]), level.push_back(categorical_levels_[s][l]);
                count.push_back(categorical_count_[s][k][l]), prob.push_back(std::exp(categorical_log_prob_[s][k][l]));
            }
    }
    dstruct::DataFrame frame;
    frame.add_column("predictor", predictor);
    frame.add_column("class", cls);
    frame.add_column("level", level);
    frame.add_column("count", count);
    frame.add_column("probability", prob);
    return frame;
}

dstruct::DataFrame NaiveBayesClassifier::posterior_terms(const dstruct::DataFrame& newdata, std::size_t row) const {
    design_.validate_categorical_levels(newdata, "NaiveBayesClassifier::posterior_terms");
    const auto kept = filter_complete_rows(newdata, /*require_response=*/false);
    if (std::find(kept.begin(), kept.end(), row) == kept.end())
        throw std::invalid_argument("NaiveBayesClassifier::posterior_terms: the row has a missing predictor");
    const auto        x = encode_row(newdata, row);
    const std::size_t K = classes_.size();
    std::vector<std::string> terms{"log prior"};
    std::vector<std::vector<double>> values(K);
    for (std::size_t k = 0; k < K; ++k) values[k].push_back(std::log(class_priors_[k]));
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        terms.push_back("log p(" + predictor_names_[j] + " | class)");
        for (std::size_t k = 0; k < K; ++k) {
            if (predictor_is_categorical_[j]) {
                values[k].push_back(categorical_log_prob_[categorical_slot_[j]][k][static_cast<std::size_t>(x[j])]);
            } else {
                const auto s = static_cast<std::size_t>(numeric_slot_[j]);
                values[k].push_back(log_gaussian_pdf(x[j], numeric_mean_[s][k], numeric_variance_[s][k]));
            }
        }
    }
    terms.push_back("log joint");
    terms.push_back("posterior");
    std::vector<double> joint(K);
    for (std::size_t k = 0; k < K; ++k) joint[k] = std::accumulate(values[k].begin(), values[k].end(), 0.0);
    const double top = *std::max_element(joint.begin(), joint.end());
    double total = 0.0;
    for (const double v : joint) total += std::exp(v - top);
    dstruct::DataFrame frame;
    frame.add_column("term", terms);
    for (std::size_t k = 0; k < K; ++k) {
        values[k].push_back(joint[k]);
        values[k].push_back(std::exp(joint[k] - top) / total);
        frame.add_column(classes_[k], values[k]);
    }
    return frame;
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

    out << "\nTraining accuracy: " << format_stat(training_accuracy() * 100.0, 2)
        << "%, leave-one-out accuracy: " << format_stat(loo_accuracy() * 100.0, 2) << "%  (" << observations_
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
    NaiveBayesClassifierPrediction result;
    result.class_label.assign(newdata.nrows(), "");
    result.probability.assign(newdata.nrows(), {});
    for (const auto row : filter_complete_rows(newdata, /*require_response=*/false)) {
        const auto   lp  = log_posteriors(encode_row(newdata, row));
        const double top = *std::max_element(lp.begin(), lp.end());
        std::vector<double> prob(lp.size());
        double total = 0.0;
        for (std::size_t k = 0; k < lp.size(); ++k) total += prob[k] = std::exp(lp[k] - top);
        for (double& v : prob) v /= total;
        result.class_label[row] = classes_[argmax(prob)];
        result.probability[row] = std::move(prob);
    }
    return result;
}

graphics::RPlot NaiveBayesClassifier::plot_classification(const dstruct::DataFrame& data,
                                                             const std::string& x_feature,
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
        plot.points(xs, ys, classes_[c], class_color(c));
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

graphics::RPlot NaiveBayesClassifier::plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                            std::size_t grid_resolution,
                                                            std::optional<std::array<double, 4>> limits) const {
    if (predictor_names_.size() != 2)
        throw std::invalid_argument("NaiveBayesClassifier::plot_decision_regions: model must have exactly 2 predictors");
    std::size_t xi = 2, yi = 2;
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_names_[j] == x_feature) xi = j;
        if (predictor_names_[j] == y_feature) yi = j;
    }
    if (xi == 2 || yi == 2 || xi == yi)
        throw std::invalid_argument("NaiveBayesClassifier::plot_decision_regions: x_feature/y_feature must match the "
                                    "model's two predictors (" + predictor_names_[0] + ", " + predictor_names_[1] + ")");
    if (predictor_is_categorical_[xi] || predictor_is_categorical_[yi])
        throw std::invalid_argument("NaiveBayesClassifier::plot_decision_regions: both predictors must be numeric");
    const auto sx = static_cast<std::size_t>(numeric_slot_[xi]), sy = static_cast<std::size_t>(numeric_slot_[yi]);
    const double x_pad = std::max(1e-6, (numeric_max_[sx] - numeric_min_[sx]) * 0.05);
    const double y_pad = std::max(1e-6, (numeric_max_[sy] - numeric_min_[sy]) * 0.10);
    const auto box = limits.value_or(std::array<double, 4>{numeric_min_[sx] - x_pad, numeric_max_[sx] + x_pad,
                                                           numeric_min_[sy] - y_pad, numeric_max_[sy] + y_pad});

    auto plot = graphics::RPlot::create();
    std::vector<graphics::RGB> fills;
    for (std::size_t c = 0; c < classes_.size(); ++c) fills.push_back(graphics::tint(class_color(c), 0.8));
    std::vector<double> x(2);
    graphics::add_class_regions(
        plot,
        [&](double a, double b) {
            x[xi] = a;
            x[yi] = b;
            return argmax(log_posteriors(x));
        },
        box, grid_resolution, fills);
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        std::vector<double> xs, ys;
        for (std::size_t i = 0; i < observations_; ++i)
            if (training_class_[i] == c) xs.push_back(training_x_[i][xi]), ys.push_back(training_x_[i][yi]);
        plot.points(xs, ys, classes_[c], class_color(c));
    }
    plot.title("Naive Bayes decision regions").x_label(x_feature).y_label(y_feature);
    return plot;
}

} // namespace datamunge::stats
