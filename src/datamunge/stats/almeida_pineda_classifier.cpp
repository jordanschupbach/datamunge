/// @file almeida_pineda_classifier.cpp
/// @brief AlmeidaPinedaClassifier: standardisation, momentum training, outputs and confusion.

#include <datamunge/stats/almeida_pineda_classifier.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::stats {

AlmeidaPinedaClassifier::AlmeidaPinedaClassifier(const dstruct::DataFrame& data, const std::string& formula,
                                                 AlmeidaPinedaClassifierOptions options)
    : formula_(formula), options_(std::move(options)) {
    if (options_.epochs == 0) throw std::invalid_argument("AlmeidaPinedaClassifier: epochs must be at least 1");
    design_ = formula_.resolve(data, ResponseKind::Categorical);
    predictor_names_ = design_.coefficient_names;
    std::vector<std::string> labels;
    const auto raw = raw_rows(data, true, &labels);
    observations_ = raw.size();
    classes_ = labels;
    std::sort(classes_.begin(), classes_.end());
    classes_.erase(std::unique(classes_.begin(), classes_.end()), classes_.end());
    if (classes_.size() < 2) throw std::invalid_argument("AlmeidaPinedaClassifier: the response needs at least 2 classes");

    const std::size_t p = predictor_names_.size(), n = raw.size();
    mean_.assign(p, 0.0);
    sd_.assign(p, 0.0);
    for (const auto& row : raw)
        for (std::size_t j = 0; j < p; ++j) mean_[j] += row[j] / static_cast<double>(n);
    for (const auto& row : raw)
        for (std::size_t j = 0; j < p; ++j) sd_[j] += (row[j] - mean_[j]) * (row[j] - mean_[j]);
    for (auto& s : sd_) s = s > 0.0 ? std::sqrt(s / static_cast<double>(n > 1 ? n - 1 : 1)) : 1.0;

    const std::size_t m = 1 + p, K = classes_.size(), units = m + options_.hidden + K;
    std::vector<std::size_t> outputs(K);
    for (std::size_t k = 0; k < K; ++k) outputs[k] = m + options_.hidden + k;
    net_ = algorithms::almeida_pineda_network(units, m, outputs, options_.init_sd, options_.seed);

    const auto X = inputs(data);
    const auto T = targets(data);
    std::vector<std::vector<double>> velocity(units, std::vector<double>(units, 0.0));
    for (std::size_t epoch = 1; epoch <= options_.epochs; ++epoch) {
        std::vector<std::vector<double>> grad(units, std::vector<double>(units, 0.0));
        double loss = 0.0, iterations = 0.0, worst = 0.0, unconverged = 0.0, wrong = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            const auto g = algorithms::almeida_pineda_gradient(net_, X[r], T[r], options_.settle);
            loss += g.error;
            iterations += static_cast<double>(g.forward.iterations);
            worst = std::max(worst, static_cast<double>(g.forward.iterations));
            unconverged += g.converged ? 0.0 : 1.0;
            std::size_t best = 0;
            for (std::size_t k = 1; k < K; ++k)
                if (g.forward.x[outputs[k]] > g.forward.x[outputs[best]]) best = k;
            wrong += classes_[best] == labels[r] ? 0.0 : 1.0;
            for (std::size_t i = m; i < units; ++i)
                for (std::size_t j = 0; j < units; ++j) grad[i][j] += g.dW[i][j] / static_cast<double>(n);
        }
        epoch_.push_back(static_cast<double>(epoch));
        loss_.push_back(loss / static_cast<double>(n));
        train_error_.push_back(wrong / static_cast<double>(n));
        mean_iterations_.push_back(iterations / static_cast<double>(n));
        max_iterations_.push_back(worst);
        unconverged_.push_back(unconverged);
        for (std::size_t i = m; i < units; ++i)
            for (std::size_t j = 0; j < units; ++j) {
                velocity[i][j] = options_.momentum * velocity[i][j] - options_.learning_rate * grad[i][j];
                net_.W[i][j] += velocity[i][j];
            }
    }
}

std::vector<std::vector<double>> AlmeidaPinedaClassifier::raw_rows(const dstruct::DataFrame& data, bool response,
                                                                   std::vector<std::string>* labels) const {
    design_.validate_categorical_levels(data, "AlmeidaPinedaClassifier");
    std::vector<std::vector<double>> rows;
    if (response) {
        if (!data.has_column(design_.response_name))
            throw std::invalid_argument("AlmeidaPinedaClassifier: data has no response column '" + design_.response_name + "'");
        const auto cd = design_.build_classification_matrix(data);
        for (std::size_t i = 0; i < cd.X.rows(); ++i) {
            std::vector<double> x(cd.X.cols());
            for (std::size_t j = 0; j < x.size(); ++j) x[j] = cd.X(i, j);
            rows.push_back(std::move(x));
        }
        if (labels) *labels = cd.labels;
    } else {
        const auto dmx = design_.build_matrix(data, /*require_response=*/false);
        for (std::size_t i = 0; i < dmx.X.rows(); ++i) {
            std::vector<double> x(dmx.X.cols());
            for (std::size_t j = 0; j < x.size(); ++j) x[j] = dmx.X(i, j);
            rows.push_back(std::move(x));
        }
    }
    return rows;
}

std::vector<std::vector<double>> AlmeidaPinedaClassifier::inputs(const dstruct::DataFrame& data) const {
    const bool has_response = data.has_column(design_.response_name);
    auto rows = raw_rows(data, has_response, nullptr);
    std::vector<std::vector<double>> out;
    for (const auto& row : rows) {
        std::vector<double> x{1.0};
        for (std::size_t j = 0; j < mean_.size(); ++j) x.push_back((row[j] - mean_[j]) / sd_[j]);
        out.push_back(std::move(x));
    }
    return out;
}

std::vector<std::vector<double>> AlmeidaPinedaClassifier::targets(const dstruct::DataFrame& data) const {
    std::vector<std::string> labels;
    (void)raw_rows(data, true, &labels);
    std::vector<std::vector<double>> out;
    for (const auto& label : labels) {
        std::vector<double> t(classes_.size(), -options_.target);
        for (std::size_t k = 0; k < classes_.size(); ++k)
            if (classes_[k] == label) t[k] = options_.target;
        out.push_back(std::move(t));
    }
    return out;
}

dstruct::DataFrame AlmeidaPinedaClassifier::outputs(const dstruct::DataFrame& data) const {
    const auto X = inputs(data);
    std::vector<std::vector<double>> columns(classes_.size());
    std::vector<double> iterations;
    for (const auto& x : X) {
        const auto s = algorithms::almeida_pineda_settle(net_, x, options_.settle);
        for (std::size_t k = 0; k < classes_.size(); ++k) columns[k].push_back(s.x[net_.output_units[k]]);
        iterations.push_back(static_cast<double>(s.iterations));
    }
    dstruct::DataFrame frame;
    for (std::size_t k = 0; k < classes_.size(); ++k) frame.add_column(classes_[k], columns[k]);
    frame.add_column("iterations", iterations);
    return frame;
}

std::vector<std::string> AlmeidaPinedaClassifier::predict(const dstruct::DataFrame& data) const {
    const auto X = inputs(data);
    std::vector<std::string> out;
    for (const auto& x : X) {
        const auto s = algorithms::almeida_pineda_settle(net_, x, options_.settle);
        std::size_t best = 0;
        for (std::size_t k = 1; k < classes_.size(); ++k)
            if (s.x[net_.output_units[k]] > s.x[net_.output_units[best]]) best = k;
        out.push_back(classes_[best]);
    }
    return out;
}

double AlmeidaPinedaClassifier::error(const dstruct::DataFrame& data) const {
    std::vector<std::string> labels;
    (void)raw_rows(data, true, &labels);
    const auto predicted = predict(data);
    if (labels.empty()) return 0.0;
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < labels.size(); ++i) wrong += labels[i] != predicted[i];
    return static_cast<double>(wrong) / static_cast<double>(labels.size());
}

dstruct::DataFrame AlmeidaPinedaClassifier::confusion(const dstruct::DataFrame& data) const {
    std::vector<std::string> labels;
    (void)raw_rows(data, true, &labels);
    const auto predicted = predict(data);
    const std::size_t K = classes_.size();
    std::vector<std::vector<double>> counts(K, std::vector<double>(K, 0.0));
    auto index = [&](const std::string& c) {
        return static_cast<std::size_t>(std::find(classes_.begin(), classes_.end(), c) - classes_.begin());
    };
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const std::size_t a = index(labels[i]);
        if (a < K) counts[a][index(predicted[i])] += 1.0;
    }
    dstruct::DataFrame frame;
    frame.add_column("actual", classes_);
    for (std::size_t k = 0; k < K; ++k) {
        std::vector<double> column(K);
        for (std::size_t a = 0; a < K; ++a) column[a] = counts[a][k];
        frame.add_column(classes_[k], column);
    }
    return frame;
}

dstruct::DataFrame AlmeidaPinedaClassifier::history() const {
    dstruct::DataFrame frame;
    frame.add_column("epoch", epoch_);
    frame.add_column("loss", loss_);
    frame.add_column("train_error", train_error_);
    frame.add_column("mean_iterations", mean_iterations_);
    frame.add_column("max_iterations", max_iterations_);
    frame.add_column("unconverged", unconverged_);
    return frame;
}

} // namespace datamunge::stats
