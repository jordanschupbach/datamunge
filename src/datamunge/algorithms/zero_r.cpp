/// @file zero_r.cpp
/// @brief Integer-label adapters and evaluation without refitting for ZeroR.
#include <datamunge/algorithms/zero_r.hpp>

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private validated label conversion and counting.
namespace {
/// @brief Copy numeric categorical labels into integers without truncation.
/// @param frame Borrowed dataset.
/// @param response Existing numeric column with finite, non-null integer labels.
/// @return Owned labels, with order preserved.
/// @throws std::invalid_argument If type, nulls or values are invalid.
/// @throws std::out_of_range If response is missing.
std::vector<int> zero_r_labels(const dstruct::DataFrame& frame, const std::string& response) {
    const auto& column = frame.get_column(response);
    std::vector<int> labels;
    labels.reserve(frame.nrows());
    for (std::size_t r = 0; r < frame.nrows(); ++r) {
        const double value = column.at(r);
        if (!std::isfinite(value) || value != std::floor(value) ||
            value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
            throw std::invalid_argument("ZeroR: expected finite integer labels in int range");
        labels.push_back(static_cast<int>(value));
    }
    return labels;
}
/// @brief Count validated labels in increasing integer order.
/// @param labels Any integer labels.
/// @return Owned class-to-count map.
std::map<int, std::size_t> zero_r_tally(const std::vector<int>& labels) {
    std::map<int, std::size_t> counts;
    for (int label : labels) ++counts[label];
    return counts;
}
}
ZeroRModel zero_r_train(const dstruct::DataFrame& frame, const std::string& response) {
    return zero_r_train(zero_r_labels(frame, response));
}
dstruct::DataFrame zero_r_counts(const dstruct::DataFrame& frame, const std::string& response) {
    const auto labels = zero_r_labels(frame, response);
    const auto counts = zero_r_tally(labels);
    std::vector<double> classes, n, p;
    for (auto [label, count] : counts) {
        classes.push_back(label); n.push_back(static_cast<double>(count));
        p.push_back(static_cast<double>(count) / labels.size());
    }
    dstruct::DataFrame out;
    out.add_column("class", classes); out.add_column("count", n); out.add_column("proportion", p);
    return out;
}
dstruct::DataFrame zero_r_confusion(const ZeroRModel& model, const dstruct::DataFrame& frame, const std::string& response) {
    if (model.total == 0 || frame.nrows() == 0) throw std::invalid_argument("zero_r_confusion: empty model or evaluation data");
    const auto labels = zero_r_labels(frame, response);
    auto counts = zero_r_tally(labels);
    counts.emplace(model.prediction, 0);
    std::vector<double> classes, support, predicted, true_positive, recall, precision;
    for (auto [label, count] : counts) {
        const bool selected = label == model.prediction;
        classes.push_back(label); support.push_back(static_cast<double>(count));
        predicted.push_back(selected ? static_cast<double>(labels.size()) : 0);
        true_positive.push_back(selected ? static_cast<double>(count) : 0);
        recall.push_back(selected && count > 0 ? 1 : 0);
        precision.push_back(selected ? static_cast<double>(count) / labels.size() : 0);
    }
    dstruct::DataFrame out;
    out.add_column("class", classes); out.add_column("support", support); out.add_column("predicted", predicted);
    out.add_column("true_positive", true_positive); out.add_column("recall", recall); out.add_column("precision", precision);
    return out;
}
dstruct::DataFrame zero_r_evaluate(const ZeroRModel& model, const dstruct::DataFrame& frame, const std::string& response) {
    const auto confusion = zero_r_confusion(model, frame, response);
    double correct = 0, classes = 0, recalls = 0;
    for (std::size_t r = 0; r < confusion.nrows(); ++r) {
        correct += confusion.double_at("true_positive", r);
        if (confusion.double_at("support", r) > 0) { ++classes; recalls += confusion.double_at("recall", r); }
    }
    dstruct::DataFrame out;
    out.add_column("prediction", std::vector<double>{static_cast<double>(model.prediction)});
    out.add_column("training_rows", std::vector<double>{static_cast<double>(model.total)});
    out.add_column("training_accuracy", std::vector<double>{model.accuracy()});
    out.add_column("evaluated_rows", std::vector<double>{static_cast<double>(frame.nrows())});
    out.add_column("observed_classes", std::vector<double>{classes});
    out.add_column("accuracy", std::vector<double>{correct / frame.nrows()});
    out.add_column("balanced_accuracy", std::vector<double>{recalls / classes});
    return out;
}
} // namespace datamunge::algorithms
