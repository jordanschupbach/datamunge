#include <datamunge/stats/conformal.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

namespace {

// The finite-sample split-conformal quantile of a set of nonconformity scores: the
// ceil((n+1)(1-alpha))-th smallest score, or +infinity when that rank exceeds n (so the guarantee
// forces the trivial all-inclusive prediction). Returns {quantile, is_infinite}.
std::pair<double, bool> conformal_quantile(std::vector<double> scores, double alpha) {
    const std::size_t n = scores.size();
    if (n == 0) throw std::invalid_argument("conformal: empty calibration set");
    const double rank_real = std::ceil((static_cast<double>(n) + 1.0) * (1.0 - alpha));
    if (rank_real > static_cast<double>(n)) return {std::numeric_limits<double>::infinity(), true};
    const std::size_t rank = static_cast<std::size_t>(std::max(1.0, rank_real));
    std::sort(scores.begin(), scores.end());
    return {scores[rank - 1], false};
}

} // namespace

double SplitConformalClassifier::nonconformity(const std::vector<double>& probabilities, std::size_t k) const {
    if (options_.score == "aps") {
        // Cumulative probability of every class at least as likely as class k (inclusive of ties).
        double sum = 0.0;
        const double pk = probabilities[k];
        for (double p : probabilities)
            if (p >= pk) sum += p;
        return sum;
    }
    // "lac": one minus the predicted probability of class k.
    return 1.0 - probabilities[k];
}

SplitConformalClassifier::SplitConformalClassifier(const linalg::DenseMatrix<double>& calibration_probabilities,
                                                   const std::vector<std::size_t>& calibration_labels,
                                                   std::vector<std::string> classes,
                                                   ConformalClassifierOptions options)
    : options_(options), classes_(std::move(classes)) {
    if (options_.alpha <= 0.0 || options_.alpha >= 1.0)
        throw std::invalid_argument("SplitConformalClassifier: alpha must be in (0, 1)");
    if (options_.score != "lac" && options_.score != "aps")
        throw std::invalid_argument("SplitConformalClassifier: score must be 'lac' or 'aps'");
    n_cal_ = calibration_probabilities.rows();
    const std::size_t K = classes_.size();
    if (K < 2) throw std::invalid_argument("SplitConformalClassifier: need at least 2 classes");
    if (calibration_probabilities.cols() != K)
        throw std::invalid_argument("SplitConformalClassifier: probability columns must match the number of classes");
    if (calibration_labels.size() != n_cal_)
        throw std::invalid_argument("SplitConformalClassifier: labels must match the number of calibration rows");

    std::vector<double> scores(n_cal_, 0.0);
    for (std::size_t i = 0; i < n_cal_; ++i) {
        if (calibration_labels[i] >= K) throw std::invalid_argument("SplitConformalClassifier: label out of range");
        std::vector<double> row(K);
        for (std::size_t k = 0; k < K; ++k) row[k] = calibration_probabilities(i, k);
        scores[i] = nonconformity(row, calibration_labels[i]);
    }
    const auto [q, inf] = conformal_quantile(std::move(scores), options_.alpha);
    quantile_ = q;
    include_all_ = inf;
}

std::vector<char> SplitConformalClassifier::membership(const std::vector<double>& probabilities) const {
    const std::size_t K = classes_.size();
    std::vector<char> in(K, 0);
    for (std::size_t k = 0; k < K; ++k)
        in[k] = (include_all_ || nonconformity(probabilities, k) <= quantile_) ? 1 : 0;
    return in;
}

std::vector<std::string> SplitConformalClassifier::predict_set(const std::vector<double>& probabilities) const {
    const auto in = membership(probabilities);
    std::vector<std::string> set;
    for (std::size_t k = 0; k < classes_.size(); ++k)
        if (in[k]) set.push_back(classes_[k]);
    return set;
}

SplitConformalClassifier::Evaluation SplitConformalClassifier::evaluate(
    const linalg::DenseMatrix<double>& test_probabilities, const std::vector<std::size_t>& test_labels) const {
    const std::size_t m = test_probabilities.rows(), K = classes_.size();
    if (test_labels.size() != m)
        throw std::invalid_argument("SplitConformalClassifier::evaluate: labels must match test rows");
    Evaluation ev;
    ev.set_sizes.resize(m, 0);
    std::size_t covered = 0, total_size = 0, empty = 0;
    for (std::size_t i = 0; i < m; ++i) {
        std::vector<double> row(K);
        for (std::size_t k = 0; k < K; ++k) row[k] = test_probabilities(i, k);
        const auto in = membership(row);
        std::size_t sz = 0;
        for (std::size_t k = 0; k < K; ++k) sz += in[k];
        ev.set_sizes[i] = sz;
        total_size += sz;
        if (sz == 0) ++empty;
        if (test_labels[i] < K && in[test_labels[i]]) ++covered;
    }
    ev.coverage = m ? static_cast<double>(covered) / static_cast<double>(m) : 0.0;
    ev.average_set_size = m ? static_cast<double>(total_size) / static_cast<double>(m) : 0.0;
    ev.empty_rate = m ? static_cast<double>(empty) / static_cast<double>(m) : 0.0;
    return ev;
}

SplitConformalRegressor::SplitConformalRegressor(const std::vector<double>& calibration_predictions,
                                                 const std::vector<double>& calibration_truth,
                                                 ConformalRegressorOptions options)
    : options_(options) {
    if (options_.alpha <= 0.0 || options_.alpha >= 1.0)
        throw std::invalid_argument("SplitConformalRegressor: alpha must be in (0, 1)");
    n_cal_ = calibration_predictions.size();
    if (calibration_truth.size() != n_cal_)
        throw std::invalid_argument("SplitConformalRegressor: predictions and truth must have equal length");
    std::vector<double> residuals(n_cal_);
    for (std::size_t i = 0; i < n_cal_; ++i) residuals[i] = std::abs(calibration_truth[i] - calibration_predictions[i]);
    const auto [q, inf] = conformal_quantile(std::move(residuals), options_.alpha);
    quantile_ = q;
    infinite_ = inf;
}

std::pair<double, double> SplitConformalRegressor::predict_interval(double prediction) const {
    if (infinite_)
        return {-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    return {prediction - quantile_, prediction + quantile_};
}

SplitConformalRegressor::Evaluation SplitConformalRegressor::evaluate(const std::vector<double>& test_predictions,
                                                                      const std::vector<double>& test_truth) const {
    const std::size_t m = test_predictions.size();
    if (test_truth.size() != m)
        throw std::invalid_argument("SplitConformalRegressor::evaluate: predictions and truth must have equal length");
    Evaluation ev;
    std::size_t covered = 0;
    for (std::size_t i = 0; i < m; ++i) {
        const auto [lo, hi] = predict_interval(test_predictions[i]);
        if (test_truth[i] >= lo && test_truth[i] <= hi) ++covered;
    }
    ev.coverage = m ? static_cast<double>(covered) / static_cast<double>(m) : 0.0;
    ev.average_width = infinite_ ? std::numeric_limits<double>::infinity() : 2.0 * quantile_;
    return ev;
}

} // namespace datamunge::stats
