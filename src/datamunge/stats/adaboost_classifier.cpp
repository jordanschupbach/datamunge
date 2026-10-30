/// @file adaboost_classifier.cpp
/// @brief AdaBoostClassifier: formula resolution, prediction, margins, history and region plots.

#include <datamunge/stats/adaboost_classifier.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

AdaBoostClassifier::AdaBoostClassifier(const dstruct::DataFrame& data, const std::string& formula,
                                       AdaBoostClassifierOptions options)
    : formula_(formula), options_(options) {
    if (options_.n_rounds == 0) throw std::invalid_argument("AdaBoostClassifier: n_rounds must be at least 1");
    design_ = formula_.resolve(data, ResponseKind::Categorical);
    const auto cd = design_.build_classification_matrix(data);
    predictor_names_ = design_.coefficient_names;
    observations_ = cd.X.rows();

    classes_ = cd.labels;
    std::sort(classes_.begin(), classes_.end());
    classes_.erase(std::unique(classes_.begin(), classes_.end()), classes_.end());
    if (classes_.size() != 2)
        throw std::invalid_argument("AdaBoostClassifier: response must have exactly 2 classes, found "
                                    + std::to_string(classes_.size()));

    const std::size_t n = cd.X.rows(), p = cd.X.cols();
    std::vector<std::vector<double>> X(n, std::vector<double>(p));
    std::vector<int> y(n);
    feature_min_.assign(p, std::numeric_limits<double>::infinity());
    feature_max_.assign(p, -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            X[i][j] = cd.X(i, j);
            feature_min_[j] = std::min(feature_min_[j], X[i][j]);
            feature_max_[j] = std::max(feature_max_[j], X[i][j]);
        }
        y[i] = cd.labels[i] == classes_[1] ? 1 : -1;
    }
    model_ = algorithms::adaboost_train(X, y, options_.n_rounds);
}

void AdaBoostClassifier::design(const dstruct::DataFrame& data, bool require_response,
                                std::vector<std::vector<double>>& X, std::vector<int>& y,
                                std::vector<std::size_t>& rows) const {
    design_.validate_categorical_levels(data, "AdaBoostClassifier");
    X.clear();
    y.clear();
    if (require_response) {
        if (!data.has_column(design_.response_name))
            throw std::invalid_argument("AdaBoostClassifier: data has no response column '" + design_.response_name + "'");
        const auto cd = design_.build_classification_matrix(data);
        rows = cd.used_row_indices;
        for (std::size_t i = 0; i < cd.X.rows(); ++i) {
            std::vector<double> x(cd.X.cols());
            for (std::size_t j = 0; j < x.size(); ++j) x[j] = cd.X(i, j);
            X.push_back(std::move(x));
            y.push_back(cd.labels[i] == classes_[1] ? 1 : -1);
        }
    } else {
        const auto dmx = design_.build_matrix(data, /*require_response=*/false);
        rows = dmx.used_row_indices;
        for (std::size_t i = 0; i < dmx.X.rows(); ++i) {
            std::vector<double> x(dmx.X.cols());
            for (std::size_t j = 0; j < x.size(); ++j) x[j] = dmx.X(i, j);
            X.push_back(std::move(x));
            y.push_back(0);
        }
    }
}

double AdaBoostClassifier::vote(const std::vector<double>& x, std::size_t rounds) const {
    const std::size_t T = rounds == 0 ? model_.stumps.size() : std::min(rounds, model_.stumps.size());
    double s = 0.0;
    for (std::size_t t = 0; t < T; ++t)
        s += model_.stumps[t].alpha * static_cast<double>(model_.stumps[t].classify(x));
    return s;
}

dstruct::DataFrame AdaBoostClassifier::history() const {
    std::vector<double> round, threshold, polarity, eps, alpha, z, bound, train;
    std::vector<std::string> feature;
    double product = 1.0;
    for (std::size_t t = 0; t < model_.stumps.size(); ++t) {
        const auto& s = model_.stumps[t];
        const double e = model_.weak_error[t];
        round.push_back(static_cast<double>(t + 1));
        feature.push_back(predictor_names_[s.feature]);
        threshold.push_back(s.threshold);
        polarity.push_back(s.polarity);
        eps.push_back(e);
        alpha.push_back(s.alpha);
        z.push_back(2.0 * std::sqrt(e * (1.0 - e)));
        product *= z.back();
        bound.push_back(product);
        train.push_back(model_.training_error[t]);
    }
    using DF = dstruct::DataFrame;
    return DF{{"round", DF::numeric_column_type(round)},        {"feature", DF::string_column_type(feature)},
              {"threshold", DF::numeric_column_type(threshold)}, {"polarity", DF::numeric_column_type(polarity)},
              {"weak_error", DF::numeric_column_type(eps)},      {"alpha", DF::numeric_column_type(alpha)},
              {"Z", DF::numeric_column_type(z)},                 {"bound", DF::numeric_column_type(bound)},
              {"train_error", DF::numeric_column_type(train)}};
}

std::vector<double> AdaBoostClassifier::decision_function(const dstruct::DataFrame& newdata, std::size_t rounds) const {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::vector<std::size_t> rows;
    design(newdata, false, X, y, rows);
    std::vector<double> out(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < X.size(); ++i) out[rows[i]] = vote(X[i], rounds);
    return out;
}

std::vector<std::string> AdaBoostClassifier::predict(const dstruct::DataFrame& newdata) const {
    const auto f = decision_function(newdata);
    std::vector<std::string> out(f.size());
    for (std::size_t i = 0; i < f.size(); ++i)
        if (!std::isnan(f[i])) out[i] = f[i] >= 0.0 ? classes_[1] : classes_[0];
    return out;
}

double AdaBoostClassifier::error(const dstruct::DataFrame& data, std::size_t rounds) const {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::vector<std::size_t> rows;
    design(data, true, X, y, rows);
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < X.size(); ++i) wrong += (vote(X[i], rounds) >= 0.0 ? 1 : -1) != y[i];
    return X.empty() ? 0.0 : static_cast<double>(wrong) / static_cast<double>(X.size());
}

std::vector<double> AdaBoostClassifier::staged_error(const dstruct::DataFrame& data) const {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::vector<std::size_t> rows;
    design(data, true, X, y, rows);
    std::vector<double> score(X.size(), 0.0), out;
    for (const auto& s : model_.stumps) {
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < X.size(); ++i) {
            score[i] += s.alpha * static_cast<double>(s.classify(X[i]));
            wrong += (score[i] >= 0.0 ? 1 : -1) != y[i];
        }
        out.push_back(X.empty() ? 0.0 : static_cast<double>(wrong) / static_cast<double>(X.size()));
    }
    return out;
}

std::vector<double> AdaBoostClassifier::margins(const dstruct::DataFrame& data, std::size_t rounds) const {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::vector<std::size_t> rows;
    design(data, true, X, y, rows);
    const std::size_t T = rounds == 0 ? model_.stumps.size() : std::min(rounds, model_.stumps.size());
    double total = 0.0;
    for (std::size_t t = 0; t < T; ++t) total += model_.stumps[t].alpha;
    std::vector<double> out(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) out[i] = total > 0.0 ? y[i] * vote(X[i], T) / total : 0.0;
    return out;
}

dstruct::DataFrame AdaBoostClassifier::feature_importance() const {
    std::vector<double> weight(predictor_names_.size(), 0.0);
    double total = 0.0;
    for (const auto& s : model_.stumps) {
        weight[s.feature] += s.alpha;
        total += s.alpha;
    }
    if (total > 0.0)
        for (auto& w : weight) w /= total;
    using DF = dstruct::DataFrame;
    return DF{{"feature", DF::string_column_type(predictor_names_)}, {"importance", DF::numeric_column_type(weight)}}
        .arrange({"importance"}, {false});
}

graphics::RPlot AdaBoostClassifier::plot_decision_regions(const dstruct::DataFrame& data) const {
    if (predictor_names_.size() != 2)
        throw std::logic_error("AdaBoostClassifier::plot_decision_regions: the model must have exactly 2 predictors");
    const graphics::RGB colors[2] = {{37, 99, 235}, {220, 38, 38}};
    double x_lo = feature_min_[0], x_hi = feature_max_[0], y_lo = feature_min_[1], y_hi = feature_max_[1];
    const double x_pad = 0.05 * std::max(x_hi - x_lo, 1e-9), y_pad = 0.05 * std::max(y_hi - y_lo, 1e-9);
    x_lo -= x_pad, x_hi += x_pad, y_lo -= y_pad, y_hi += y_pad;

    // Cell edges: the plot limits plus every stump threshold inside them.
    std::vector<double> xs{x_lo, x_hi}, ys{y_lo, y_hi};
    for (const auto& s : model_.stumps) {
        auto& edges = s.feature == 0 ? xs : ys;
        const double lo = s.feature == 0 ? x_lo : y_lo, hi = s.feature == 0 ? x_hi : y_hi;
        if (s.threshold > lo && s.threshold < hi) edges.push_back(s.threshold);
    }
    for (auto* edges : {&xs, &ys}) {
        std::sort(edges->begin(), edges->end());
        edges->erase(std::unique(edges->begin(), edges->end()), edges->end());
    }

    auto plot = graphics::RPlot::create();
    plot.x_limits(x_lo, x_hi).y_limits(y_lo, y_hi).show_grid(false).x_label(predictor_names_[0]).y_label(predictor_names_[1]);
    // Opaque pastel versions of the class colours (16% over white), so adjacent cells leave no
    // visible seams; within each vertical strip, runs of same-class cells become one rectangle.
    const auto pastel = [](graphics::RGB c) {
        const auto mix = [](int v) { return static_cast<int>(std::lround(255.0 - 0.16 * (255.0 - v))); };
        return graphics::RGB{mix(c.r), mix(c.g), mix(c.b)};
    };
    for (std::size_t a = 0; a + 1 < xs.size(); ++a) {
        const double cx = 0.5 * (xs[a] + xs[a + 1]);
        std::size_t b = 0;
        while (b + 1 < ys.size()) {
            const int cls = vote({cx, 0.5 * (ys[b] + ys[b + 1])}, 0) >= 0.0 ? 1 : 0;
            std::size_t e = b + 1;
            while (e + 1 < ys.size() && (vote({cx, 0.5 * (ys[e] + ys[e + 1])}, 0) >= 0.0 ? 1 : 0) == cls) ++e;
            plot.polygon({xs[a], xs[a + 1], xs[a + 1], xs[a]}, {ys[b], ys[b], ys[e], ys[e]}, pastel(colors[cls]), true);
            b = e;
        }
    }

    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::vector<std::size_t> rows;
    design(data, true, X, y, rows);
    for (int k = 0; k < 2; ++k) {
        std::vector<double> px, py;
        for (std::size_t i = 0; i < X.size(); ++i)
            if ((y[i] > 0) == (k == 1)) {
                px.push_back(X[i][0]);
                py.push_back(X[i][1]);
            }
        plot.points(px, py, classes_[static_cast<std::size_t>(k)], colors[k], 3.5);
    }
    return plot;
}

} // namespace datamunge::stats
