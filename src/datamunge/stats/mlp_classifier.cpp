#include <datamunge/stats/mlp_classifier.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

namespace {

const plot::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

std::string format_stat(double v, int precision = 4) {
    if (std::isnan(v)) return "NaN";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

double activate(double z, const std::string& act) {
    if (act == "relu") return z > 0.0 ? z : 0.0;
    if (act == "sigmoid") return 1.0 / (1.0 + std::exp(-z));
    return std::tanh(z); // "tanh"
}

// Derivative of the activation, expressed from the pre-activation z.
double activate_deriv(double z, const std::string& act) {
    if (act == "relu") return z > 0.0 ? 1.0 : 0.0;
    if (act == "sigmoid") {
        const double s = 1.0 / (1.0 + std::exp(-z));
        return s * (1.0 - s);
    }
    const double t = std::tanh(z);
    return 1.0 - t * t;
}

void softmax_inplace(std::vector<double>& v) {
    const double mx = *std::max_element(v.begin(), v.end());
    double sum = 0.0;
    for (double& x : v) { x = std::exp(x - mx); sum += x; }
    for (double& x : v) x /= sum;
}

} // namespace

MLPClassifier::MLPClassifier(const dstruct::DataFrame& data, const std::string& formula, MLPClassifierOptions options)
    : formula_(formula) {
    options_ = options;
    fit(data);
}

std::vector<std::size_t> MLPClassifier::architecture() const {
    std::vector<std::size_t> arch;
    arch.push_back(predictor_names_.size());
    for (auto h : options_.hidden_layer_sizes) arch.push_back(h);
    arch.push_back(classes_.size());
    return arch;
}

std::size_t MLPClassifier::n_parameters() const {
    std::size_t total = 0;
    for (const auto& W : weights_) total += W.rows() * W.cols() + W.rows();
    return total;
}

std::vector<double> MLPClassifier::forward_probs(const std::vector<double>& x_std) const {
    std::vector<double> a = x_std;
    const std::size_t L = weights_.size();
    for (std::size_t l = 0; l < L; ++l) {
        const std::size_t out = weights_[l].rows(), in = weights_[l].cols();
        std::vector<double> z(out, 0.0);
        for (std::size_t o = 0; o < out; ++o) {
            double s = biases_[l][o];
            for (std::size_t i = 0; i < in; ++i) s += weights_[l](o, i) * a[i];
            z[o] = s;
        }
        if (l + 1 < L)
            for (double& v : z) v = activate(v, options_.activation);
        else
            softmax_inplace(z);
        a = std::move(z);
    }
    return a;
}

void MLPClassifier::fit(const dstruct::DataFrame& data) {
    if (options_.activation != "relu" && options_.activation != "tanh" && options_.activation != "sigmoid")
        throw std::invalid_argument("MLPClassifier: activation must be 'relu', 'tanh', or 'sigmoid'");
    if (options_.max_epochs == 0) throw std::invalid_argument("MLPClassifier: max_epochs must be at least 1");
    if (options_.learning_rate <= 0.0) throw std::invalid_argument("MLPClassifier: learning_rate must be positive");

    design_       = formula_.resolve(data, ResponseKind::Categorical);
    const auto cd = design_.build_classification_matrix(data);
    const std::size_t n = cd.X.rows();
    const std::size_t p = cd.X.cols();
    if (p == 0) throw std::invalid_argument("MLPClassifier: at least one predictor is required");

    predictor_names_ = design_.coefficient_names;
    observations_    = n;
    training_labels_ = cd.labels;

    classes_.clear();
    for (const auto& label : cd.labels)
        if (std::find(classes_.begin(), classes_.end(), label) == classes_.end()) classes_.push_back(label);
    std::sort(classes_.begin(), classes_.end());
    if (classes_.size() < 2)
        throw std::invalid_argument("MLPClassifier: response must have at least 2 distinct classes");
    const std::size_t K = classes_.size();

    // Standardize predictors; record range for decision-region plots.
    feature_min_.assign(p, cd.X(0, 0));
    feature_max_.assign(p, cd.X(0, 0));
    for (std::size_t j = 0; j < p; ++j) {
        feature_min_[j] = cd.X(0, j);
        feature_max_[j] = cd.X(0, j);
        for (std::size_t i = 0; i < n; ++i) {
            feature_min_[j] = std::min(feature_min_[j], cd.X(i, j));
            feature_max_[j] = std::max(feature_max_[j], cd.X(i, j));
        }
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
            for (std::size_t i = 0; i < n; ++i) { const double d = cd.X(i, j) - feature_mean_[j]; ss += d * d; }
            const double sd = std::sqrt(ss / static_cast<double>(n));
            feature_scale_[j] = (sd < 1e-12) ? 1.0 : sd;
        }
    }
    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) X(i, j) = (cd.X(i, j) - feature_mean_[j]) / feature_scale_[j];

    std::vector<std::size_t> target(n);
    for (std::size_t i = 0; i < n; ++i)
        target[i] = static_cast<std::size_t>(
            std::lower_bound(classes_.begin(), classes_.end(), cd.labels[i]) - classes_.begin());

    // Build layer widths and initialize parameters (He for relu, Xavier otherwise).
    std::vector<std::size_t> units;
    units.push_back(p);
    for (auto h : options_.hidden_layer_sizes) units.push_back(h);
    units.push_back(K);
    const std::size_t L = units.size() - 1;

    std::mt19937_64 rng(options_.seed);
    weights_.clear();
    biases_.clear();
    for (std::size_t l = 0; l < L; ++l) {
        const std::size_t in = units[l], out = units[l + 1];
        const double gain = (options_.activation == "relu") ? 2.0 : 1.0;
        const double sd = std::sqrt(gain / static_cast<double>(in));
        std::normal_distribution<double> init(0.0, sd);
        linalg::DenseMatrix<double> W(out, in, 0.0);
        for (std::size_t o = 0; o < out; ++o)
            for (std::size_t i = 0; i < in; ++i) W(o, i) = init(rng);
        weights_.push_back(std::move(W));
        biases_.emplace_back(out, 0.0);
    }

    const std::size_t batch = (options_.batch_size == 0) ? n : std::min(options_.batch_size, n);
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);

    loss_curve_.assign(options_.max_epochs, 0.0);
    for (std::size_t epoch = 0; epoch < options_.max_epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        for (std::size_t start = 0; start < n; start += batch) {
            const std::size_t end = std::min(start + batch, n);
            const double inv = 1.0 / static_cast<double>(end - start);

            // Gradient accumulators.
            std::vector<linalg::DenseMatrix<double>> gW;
            std::vector<std::vector<double>> gb;
            for (std::size_t l = 0; l < L; ++l) {
                gW.emplace_back(weights_[l].rows(), weights_[l].cols(), 0.0);
                gb.emplace_back(weights_[l].rows(), 0.0);
            }

            for (std::size_t bi = start; bi < end; ++bi) {
                const std::size_t idx = order[bi];
                // Forward with caching.
                std::vector<std::vector<double>> a(L + 1), z(L);
                a[0].resize(p);
                for (std::size_t j = 0; j < p; ++j) a[0][j] = X(idx, j);
                for (std::size_t l = 0; l < L; ++l) {
                    const std::size_t out = units[l + 1], in = units[l];
                    z[l].assign(out, 0.0);
                    for (std::size_t o = 0; o < out; ++o) {
                        double s = biases_[l][o];
                        for (std::size_t i = 0; i < in; ++i) s += weights_[l](o, i) * a[l][i];
                        z[l][o] = s;
                    }
                    a[l + 1] = z[l];
                    if (l + 1 < L)
                        for (double& v : a[l + 1]) v = activate(v, options_.activation);
                    else
                        softmax_inplace(a[l + 1]);
                }

                // Backward: output delta = softmax - onehot.
                std::vector<double> delta = a[L];
                delta[target[idx]] -= 1.0;
                for (std::size_t l = L; l-- > 0;) {
                    const std::size_t out = units[l + 1], in = units[l];
                    for (std::size_t o = 0; o < out; ++o) {
                        gb[l][o] += delta[o];
                        for (std::size_t i = 0; i < in; ++i) gW[l](o, i) += delta[o] * a[l][i];
                    }
                    if (l > 0) {
                        std::vector<double> new_delta(in, 0.0);
                        for (std::size_t i = 0; i < in; ++i) {
                            double s = 0.0;
                            for (std::size_t o = 0; o < out; ++o) s += weights_[l](o, i) * delta[o];
                            new_delta[i] = s * activate_deriv(z[l - 1][i], options_.activation);
                        }
                        delta = std::move(new_delta);
                    }
                }
            }

            // Parameter update (batch-averaged gradient + L2 on weights).
            for (std::size_t l = 0; l < L; ++l) {
                for (std::size_t o = 0; o < weights_[l].rows(); ++o) {
                    for (std::size_t i = 0; i < weights_[l].cols(); ++i)
                        weights_[l](o, i) -= options_.learning_rate *
                                             (gW[l](o, i) * inv + options_.l2 * weights_[l](o, i));
                    biases_[l][o] -= options_.learning_rate * gb[l][o] * inv;
                }
            }
        }

        // Epoch training loss (mean cross-entropy).
        double loss = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<double> xi(p);
            for (std::size_t j = 0; j < p; ++j) xi[j] = X(i, j);
            const auto probs = forward_probs(xi);
            loss -= std::log(std::max(probs[target[i]], 1e-12));
        }
        loss_curve_[epoch] = loss / static_cast<double>(n);
    }
    converged_ = options_.max_epochs < 2 ||
                 std::abs(loss_curve_.back() - loss_curve_[loss_curve_.size() - 2]) < 1e-5;

    // Resubstitution fitted classes.
    fitted_classes_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> xi(p);
        for (std::size_t j = 0; j < p; ++j) xi[j] = X(i, j);
        const auto probs = forward_probs(xi);
        fitted_classes_[i] =
            classes_[static_cast<std::size_t>(std::max_element(probs.begin(), probs.end()) - probs.begin())];
    }
}

double MLPClassifier::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
        if (fitted_classes_[i] == training_labels_[i]) ++correct;
    return fitted_classes_.empty() ? 0.0 : static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> MLPClassifier::confusion_matrix() const {
    const std::size_t k = classes_.size();
    linalg::DenseMatrix<double> matrix(k, k, 0.0);
    for (std::size_t i = 0; i < training_labels_.size(); ++i) {
        const auto a = std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]);
        const auto pidx = std::lower_bound(classes_.begin(), classes_.end(), fitted_classes_[i]);
        matrix(static_cast<std::size_t>(a - classes_.begin()), static_cast<std::size_t>(pidx - classes_.begin())) +=
            1.0;
    }
    return matrix;
}

MLPClassifierPrediction MLPClassifier::predict_detail(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "MLPClassifier::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);
    MLPClassifierPrediction result;
    result.class_label.assign(newdata.nrows(), "");
    result.probabilities.assign(newdata.nrows(), {});
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        std::vector<double> query(dm.X.cols());
        for (std::size_t j = 0; j < dm.X.cols(); ++j) query[j] = (dm.X(i, j) - feature_mean_[j]) / feature_scale_[j];
        const auto probs = forward_probs(query);
        const auto best = static_cast<std::size_t>(std::max_element(probs.begin(), probs.end()) - probs.begin());
        result.class_label[dm.used_row_indices[i]] = classes_[best];
        result.probabilities[dm.used_row_indices[i]] = probs;
    }
    return result;
}

std::vector<std::string> MLPClassifier::predict(const dstruct::DataFrame& newdata) const {
    return predict_detail(newdata).class_label;
}

double MLPClassifier::accuracy(const dstruct::DataFrame& newdata) const {
    const auto cd = design_.build_classification_matrix(newdata);
    std::size_t correct = 0, total = 0;
    for (std::size_t i = 0; i < cd.X.rows(); ++i) {
        std::vector<double> query(cd.X.cols());
        for (std::size_t j = 0; j < cd.X.cols(); ++j) query[j] = (cd.X(i, j) - feature_mean_[j]) / feature_scale_[j];
        const auto probs = forward_probs(query);
        const auto best = static_cast<std::size_t>(std::max_element(probs.begin(), probs.end()) - probs.begin());
        if (classes_[best] == cd.labels[i]) ++correct;
        ++total;
    }
    return total == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(total);
}

std::string MLPClassifier::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void MLPClassifier::print_summary() const { print_summary(std::cout); }

void MLPClassifier::print_summary(std::ostream& os) const {
    os << "Multi-Layer Perceptron Classifier\n";
    os << "Formula: " << formula_.text() << "\n";
    os << "Architecture: ";
    const auto arch = architecture();
    for (std::size_t i = 0; i < arch.size(); ++i) os << (i ? " -> " : "") << arch[i];
    os << " (" << options_.activation << " hidden, softmax output)\n";
    os << "Trainable parameters: " << n_parameters() << ", observations: " << observations_ << "\n";
    os << "Epochs: " << options_.max_epochs << ", learning rate: " << format_stat(options_.learning_rate);
    if (options_.l2 > 0.0) os << ", L2: " << format_stat(options_.l2);
    os << "\n";
    if (!loss_curve_.empty())
        os << "Final training cross-entropy: " << format_stat(loss_curve_.back()) << "\n";
    os << "Training accuracy: " << format_stat(training_accuracy()) << "\n";
}

plot::RPlot MLPClassifier::plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                 std::size_t grid_resolution) const {
    if (predictor_names_.size() != 2)
        throw std::invalid_argument("MLPClassifier::plot_decision_regions: model must have exactly 2 predictors");
    std::size_t xi = 2, yi = 2;
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        if (predictor_names_[j] == x_feature) xi = j;
        if (predictor_names_[j] == y_feature) yi = j;
    }
    if (xi == 2 || yi == 2)
        throw std::invalid_argument("MLPClassifier::plot_decision_regions: x_feature/y_feature must match the "
                                    "model's two predictors (" + predictor_names_[0] + ", " + predictor_names_[1] + ")");

    double x_min = feature_min_[xi], x_max = feature_max_[xi];
    double y_min = feature_min_[yi], y_max = feature_max_[yi];
    const double x_pad = std::max(1e-6, (x_max - x_min) * 0.1);
    const double y_pad = std::max(1e-6, (y_max - y_min) * 0.1);
    x_min -= x_pad; x_max += x_pad;
    y_min -= y_pad; y_max += y_pad;

    auto plot = plot::RPlot::create();
    std::vector<std::vector<double>> grid_x(classes_.size()), grid_y(classes_.size());
    for (std::size_t gx = 0; gx < grid_resolution; ++gx) {
        for (std::size_t gy = 0; gy < grid_resolution; ++gy) {
            const double x = x_min + (x_max - x_min) * static_cast<double>(gx) / static_cast<double>(grid_resolution - 1);
            const double y = y_min + (y_max - y_min) * static_cast<double>(gy) / static_cast<double>(grid_resolution - 1);
            std::vector<double> query(2);
            query[xi] = (x - feature_mean_[xi]) / feature_scale_[xi];
            query[yi] = (y - feature_mean_[yi]) / feature_scale_[yi];
            const auto probs = forward_probs(query);
            const auto pc = static_cast<std::size_t>(std::max_element(probs.begin(), probs.end()) - probs.begin());
            grid_x[pc].push_back(x);
            grid_y[pc].push_back(y);
        }
    }
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        auto color = kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))];
        color.r = static_cast<std::uint8_t>(std::min(255, color.r + (255 - color.r) * 3 / 4));
        color.g = static_cast<std::uint8_t>(std::min(255, color.g + (255 - color.g) * 3 / 4));
        color.b = static_cast<std::uint8_t>(std::min(255, color.b + (255 - color.b) * 3 / 4));
        plot.points(grid_x[c], grid_y[c], classes_[c] + " region", color, 3.0);
    }
    plot.title("MLP Decision Regions").x_label(x_feature).y_label(y_feature);
    return plot;
}

plot::RPlot MLPClassifier::plot_loss_curve() const {
    std::vector<double> epochs(loss_curve_.size());
    for (std::size_t i = 0; i < loss_curve_.size(); ++i) epochs[i] = static_cast<double>(i + 1);
    auto plot = plot::RPlot::create();
    plot.line(epochs, loss_curve_, "training cross-entropy", {37, 99, 235}, 2.0);
    plot.title("Training Loss Curve").x_label("epoch").y_label("mean cross-entropy");
    return plot;
}

} // namespace datamunge::stats
