/// @file conformal.cpp
/// @brief Split conformal quantile and coverage law, LAC/APS classification sets (marginal or
///        class-conditional) with p-values, and band-and-scale regression intervals.

#include <datamunge/stats/conformal.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Input checks shared by the conformal classes.
namespace {

/// @brief Throws unless @p alpha is in (0, 1).
/// @param alpha  Level.
/// @param who    Caller name for the message.
void require_level(double alpha, const char* who) {
    if (!(alpha > 0.0 && alpha < 1.0)) throw std::invalid_argument(std::string(who) + ": alpha must be in (0, 1)");
}

/// @brief The conformal rank \f$\lceil(n+1)(1-\alpha)\rceil\f$, guarded against rounding just above an integer.
/// @param n      Calibration size.
/// @param alpha  Level.
/// @return The rank.
std::size_t conformal_rank(std::size_t n, double alpha) {
    const double r = (static_cast<double>(n) + 1.0) * (1.0 - alpha);
    return static_cast<std::size_t>(std::max(1.0, std::ceil(r - 1e-9)));
}

} // namespace

double conformal_quantile(std::vector<double> scores, double alpha) {
    require_level(alpha, "conformal_quantile");
    const std::size_t n = scores.size();
    if (n == 0) throw std::invalid_argument("conformal: empty calibration set");
    const std::size_t k = conformal_rank(n, alpha);
    if (k > n) return std::numeric_limits<double>::infinity();
    std::nth_element(scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(k - 1), scores.end());
    return scores[k - 1];
}

ConformalCoverage conformal_coverage(std::size_t n, double alpha) {
    require_level(alpha, "conformal_coverage");
    if (n == 0) throw std::invalid_argument("conformal_coverage: n must be at least 1");
    ConformalCoverage c;
    c.rank = conformal_rank(n, alpha);
    if (c.rank > n) return c;
    const double a = static_cast<double>(c.rank), b = static_cast<double>(n + 1 - c.rank);
    c.mean = a / (a + b), c.beta_a = a, c.beta_b = b;
    c.sd   = std::sqrt(a * b / ((a + b) * (a + b) * (a + b + 1.0)));
    return c;
}

// --- classification -------------------------------------------------------------------------

double SplitConformalClassifier::nonconformity(const std::vector<double>& probabilities, std::size_t k) const {
    if (k >= probabilities.size()) throw std::invalid_argument("SplitConformalClassifier: class index out of range");
    if (options_.score == "aps") {
        // Total probability of every class at least as likely as class k (ties included).
        double       sum = 0.0;
        const double pk  = probabilities[k];
        for (const double p : probabilities)
            if (p >= pk) sum += p;
        return sum;
    }
    return 1.0 - probabilities[k];
}

void SplitConformalClassifier::calibrate(const std::vector<std::vector<double>>& rows, const std::vector<std::size_t>& labels) {
    require_level(options_.alpha, "SplitConformalClassifier");
    if (options_.score != "lac" && options_.score != "aps")
        throw std::invalid_argument("SplitConformalClassifier: score must be 'lac' or 'aps'");
    const std::size_t K = classes_.size();
    if (K < 2) throw std::invalid_argument("SplitConformalClassifier: need at least 2 classes");
    if (labels.size() != rows.size())
        throw std::invalid_argument("SplitConformalClassifier: labels must match the number of calibration rows");
    n_cal_ = rows.size();
    scores_.assign(n_cal_, 0.0);
    labels_ = labels;
    for (std::size_t i = 0; i < n_cal_; ++i) {
        if (rows[i].size() != K)
            throw std::invalid_argument("SplitConformalClassifier: probability columns must match the number of classes");
        if (labels[i] >= K) throw std::invalid_argument("SplitConformalClassifier: label out of range");
        scores_[i] = nonconformity(rows[i], labels[i]);
    }
    quantile_ = conformal_quantile(scores_, options_.alpha);
    class_quantiles_.assign(K, quantile_);
    if (options_.class_conditional) {
        for (std::size_t k = 0; k < K; ++k) {
            std::vector<double> sk;
            for (std::size_t i = 0; i < n_cal_; ++i)
                if (labels_[i] == k) sk.push_back(scores_[i]);
            if (sk.empty())
                throw std::invalid_argument("SplitConformalClassifier: class '" + classes_[k] + "' has no calibration rows");
            class_quantiles_[k] = conformal_quantile(std::move(sk), options_.alpha);
        }
    }
}

SplitConformalClassifier::SplitConformalClassifier(const linalg::DenseMatrix<double>& calibration_probabilities,
                                                   const std::vector<std::size_t>& calibration_labels,
                                                   std::vector<std::string> classes, ConformalClassifierOptions options)
    : options_(std::move(options)), classes_(std::move(classes)) {
    std::vector<std::vector<double>> rows(calibration_probabilities.rows(), std::vector<double>(calibration_probabilities.cols()));
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t k = 0; k < rows[i].size(); ++k) rows[i][k] = calibration_probabilities(i, k);
    calibrate(rows, calibration_labels);
}

SplitConformalClassifier::SplitConformalClassifier(const std::vector<std::vector<double>>& calibration_probabilities,
                                                   const std::vector<std::string>& calibration_labels,
                                                   std::vector<std::string> classes, ConformalClassifierOptions options)
    : options_(std::move(options)), classes_(std::move(classes)) {
    std::vector<std::size_t> idx;
    idx.reserve(calibration_labels.size());
    for (const auto& l : calibration_labels) idx.push_back(class_index(l));
    calibrate(calibration_probabilities, idx);
}

std::size_t SplitConformalClassifier::class_index(const std::string& label) const {
    const auto it = std::find(classes_.begin(), classes_.end(), label);
    if (it == classes_.end()) throw std::invalid_argument("SplitConformalClassifier: unknown class '" + label + "'");
    return static_cast<std::size_t>(it - classes_.begin());
}

ConformalCoverage SplitConformalClassifier::coverage() const { return conformal_coverage(n_cal_, options_.alpha); }

std::vector<char> SplitConformalClassifier::membership(const std::vector<double>& probabilities) const {
    const std::size_t K = classes_.size();
    if (probabilities.size() != K) throw std::invalid_argument("SplitConformalClassifier: wrong number of probabilities");
    std::vector<char> in(K, 0);
    bool any = false;
    for (std::size_t k = 0; k < K; ++k) {
        in[k] = nonconformity(probabilities, k) <= class_quantiles_[k] ? 1 : 0;
        any   = any || in[k];
    }
    if (!any && options_.nonempty)
        in[static_cast<std::size_t>(std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin())] = 1;
    return in;
}

std::vector<std::string> SplitConformalClassifier::predict_set(const std::vector<double>& probabilities) const {
    const auto               in = membership(probabilities);
    std::vector<std::string> set;
    for (std::size_t k = 0; k < classes_.size(); ++k)
        if (in[k]) set.push_back(classes_[k]);
    return set;
}

std::vector<std::vector<std::string>> SplitConformalClassifier::predict_sets(const std::vector<std::vector<double>>& probabilities) const {
    std::vector<std::vector<std::string>> out;
    out.reserve(probabilities.size());
    for (const auto& row : probabilities) out.push_back(predict_set(row));
    return out;
}

std::vector<double> SplitConformalClassifier::p_values(const std::vector<double>& probabilities) const {
    const std::size_t K = classes_.size();
    if (probabilities.size() != K) throw std::invalid_argument("SplitConformalClassifier: wrong number of probabilities");
    std::vector<double> p(K);
    for (std::size_t k = 0; k < K; ++k) {
        const double s   = nonconformity(probabilities, k);
        double       ge  = 0.0, n = 0.0;
        for (std::size_t i = 0; i < n_cal_; ++i) {
            if (options_.class_conditional && labels_[i] != k) continue;
            n += 1.0;
            ge += scores_[i] >= s ? 1.0 : 0.0;
        }
        p[k] = (1.0 + ge) / (n + 1.0);
    }
    return p;
}

SplitConformalClassifier::Evaluation SplitConformalClassifier::evaluate(const linalg::DenseMatrix<double>& test_probabilities,
                                                                        const std::vector<std::size_t>& test_labels) const {
    const std::size_t m = test_probabilities.rows(), K = classes_.size();
    if (test_labels.size() != m) throw std::invalid_argument("SplitConformalClassifier::evaluate: labels must match test rows");
    Evaluation ev;
    ev.set_sizes.resize(m, 0);
    ev.size_counts.assign(K + 1, 0);
    std::vector<double> class_n(K, 0.0), class_hit(K, 0.0);
    std::size_t         covered = 0, total = 0, empty = 0;
    for (std::size_t i = 0; i < m; ++i) {
        std::vector<double> row(K);
        for (std::size_t k = 0; k < K; ++k) row[k] = test_probabilities(i, k);
        const auto  in = membership(row);
        std::size_t sz = 0;
        for (std::size_t k = 0; k < K; ++k) sz += static_cast<std::size_t>(in[k]);
        ev.set_sizes[i] = sz;
        ++ev.size_counts[sz];
        total += sz;
        if (sz == 0) ++empty;
        if (test_labels[i] < K) {
            class_n[test_labels[i]] += 1.0;
            if (in[test_labels[i]]) ++covered, class_hit[test_labels[i]] += 1.0;
        }
    }
    const double md     = static_cast<double>(m);
    ev.coverage         = m ? static_cast<double>(covered) / md : 0.0;
    ev.average_set_size = m ? static_cast<double>(total) / md : 0.0;
    ev.empty_rate       = m ? static_cast<double>(empty) / md : 0.0;
    ev.class_coverage.resize(K);
    for (std::size_t k = 0; k < K; ++k)
        ev.class_coverage[k] = class_n[k] > 0.0 ? class_hit[k] / class_n[k] : std::numeric_limits<double>::quiet_NaN();
    return ev;
}

SplitConformalClassifier::Evaluation SplitConformalClassifier::evaluate(const std::vector<std::vector<double>>& test_probabilities,
                                                                        const std::vector<std::string>& test_labels) const {
    if (test_labels.size() != test_probabilities.size())
        throw std::invalid_argument("SplitConformalClassifier::evaluate: labels must match test rows");
    const std::size_t           K = classes_.size();
    linalg::DenseMatrix<double> P(test_probabilities.size(), K);
    std::vector<std::size_t>    idx;
    for (std::size_t i = 0; i < test_probabilities.size(); ++i) {
        if (test_probabilities[i].size() != K) throw std::invalid_argument("SplitConformalClassifier: wrong number of probabilities");
        for (std::size_t k = 0; k < K; ++k) P(i, k) = test_probabilities[i][k];
        idx.push_back(class_index(test_labels[i]));
    }
    return evaluate(P, idx);
}

// --- regression -----------------------------------------------------------------------------

SplitConformalRegressor::SplitConformalRegressor(const std::vector<double>& calibration_predictions,
                                                 const std::vector<double>& calibration_truth, ConformalRegressorOptions options)
    : SplitConformalRegressor(calibration_predictions, calibration_predictions,
                              std::vector<double>(calibration_predictions.size(), 1.0), calibration_truth, options) {}

SplitConformalRegressor::SplitConformalRegressor(const std::vector<double>& lower, const std::vector<double>& upper,
                                                 const std::vector<double>& scale, const std::vector<double>& truth,
                                                 ConformalRegressorOptions options)
    : options_(options) {
    require_level(options_.alpha, "SplitConformalRegressor");
    n_cal_ = truth.size();
    if (lower.size() != n_cal_ || upper.size() != n_cal_ || scale.size() != n_cal_)
        throw std::invalid_argument("SplitConformalRegressor: predictions and truth must have equal length");
    scores_.resize(n_cal_);
    for (std::size_t i = 0; i < n_cal_; ++i) {
        if (!(scale[i] > 0.0)) throw std::invalid_argument("SplitConformalRegressor: scales must be positive");
        scores_[i] = std::max(lower[i] - truth[i], truth[i] - upper[i]) / scale[i];
    }
    quantile_ = conformal_quantile(scores_, options_.alpha);
}

SplitConformalRegressor SplitConformalRegressor::normalized(const std::vector<double>& predictions, const std::vector<double>& scale,
                                                            const std::vector<double>& truth, ConformalRegressorOptions options) {
    return SplitConformalRegressor(predictions, predictions, scale, truth, options);
}

SplitConformalRegressor SplitConformalRegressor::quantile_band(const std::vector<double>& lower, const std::vector<double>& upper,
                                                               const std::vector<double>& truth, ConformalRegressorOptions options) {
    return SplitConformalRegressor(lower, upper, std::vector<double>(lower.size(), 1.0), truth, options);
}

ConformalCoverage SplitConformalRegressor::coverage() const { return conformal_coverage(n_cal_, options_.alpha); }

std::pair<double, double> SplitConformalRegressor::predict_interval(double prediction) const {
    return predict_interval(prediction, prediction, 1.0);
}

std::pair<double, double> SplitConformalRegressor::predict_interval(double lower, double upper, double scale) const {
    if (std::isinf(quantile_))
        return {-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    return {lower - quantile_ * scale, upper + quantile_ * scale};
}

SplitConformalRegressor::Evaluation SplitConformalRegressor::evaluate(const std::vector<double>& test_predictions,
                                                                      const std::vector<double>& test_truth) const {
    return evaluate(test_predictions, test_predictions, std::vector<double>(test_predictions.size(), 1.0), test_truth);
}

SplitConformalRegressor::Evaluation SplitConformalRegressor::evaluate(const std::vector<double>& lower, const std::vector<double>& upper,
                                                                      const std::vector<double>& scale,
                                                                      const std::vector<double>& truth) const {
    const std::size_t m = truth.size();
    if (lower.size() != m || upper.size() != m || scale.size() != m)
        throw std::invalid_argument("SplitConformalRegressor::evaluate: predictions and truth must have equal length");
    Evaluation ev;
    ev.covered.resize(m), ev.widths.resize(m);
    double hits = 0.0, width = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        const auto [lo, hi] = predict_interval(lower[i], upper[i], scale[i]);
        ev.covered[i]       = truth[i] >= lo && truth[i] <= hi ? 1 : 0;
        ev.widths[i]        = hi - lo;
        hits += ev.covered[i], width += hi - lo;
    }
    ev.coverage      = m ? hits / static_cast<double>(m) : 0.0;
    ev.average_width = m ? width / static_cast<double>(m) : 0.0;
    return ev;
}

} // namespace datamunge::stats
