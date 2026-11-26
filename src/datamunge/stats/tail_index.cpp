/// @file tail_index.cpp
/// @brief Order-statistic tail-index estimators and their asymptotic variances.
#include <datamunge/stats/tail_index.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for tail-index estimation.
namespace {
/// @brief Largest admissible k for a method and sample size.
/// @param method Estimator.
/// @param n Sample size.
/// @return Maximum k.
std::size_t max_k(TailIndexMethod method, std::size_t n) { return method == TailIndexMethod::pickands ? n / 4 : n - 1; }

/// @brief Asymptotic variance factor \f$k\,\mathrm{Var}(\hat\xi)\f$ at \f$\xi\f$.
/// @param method Estimator.
/// @param xi Index.
/// @return Variance factor.
double variance_factor(TailIndexMethod method, double xi) {
    switch (method) {
    case TailIndexMethod::hill:
        return xi * xi;
    case TailIndexMethod::pickands: {
        // xi^2 (2^(2xi+1) + 1) / (2 (2^xi - 1) log 2)^2, with limit 3 / (4 log^4 2) at xi = 0.
        const double l2 = std::numbers::ln2;
        if (std::abs(xi) < 1e-8)
            return 3 / (4 * std::pow(l2, 4));
        const double d = 2 * std::expm1(xi * l2) * l2;
        return xi * xi * (std::exp2(2 * xi + 1) + 1) / (d * d);
    }
    case TailIndexMethod::moment:
        if (xi >= 0)
            return 1 + xi * xi;
        return (1 - xi) * (1 - xi) * (1 - 2 * xi) *
               (4 - 8 * (1 - 2 * xi) / (1 - 3 * xi) + (5 - 11 * xi) * (1 - 2 * xi) / ((1 - 3 * xi) * (1 - 4 * xi)));
    }
    return 0.0;
}

/// @brief Estimate at k from data sorted ascending.
/// @param x Sorted data.
/// @param k Order statistics used.
/// @param method Estimator.
/// @param zq Normal quantile for the interval.
/// @return Estimate.
TailIndexEstimate estimate_sorted(const std::vector<double>& x, std::size_t k, TailIndexMethod method, double zq) {
    const std::size_t n = x.size();
    // x[n - i] is X_{n-i+1:n}; the threshold X_{n-k:n} is x[n - k - 1].
    TailIndexEstimate e;
    e.k = k;
    e.threshold = x[n - k - 1];
    if (method == TailIndexMethod::pickands) {
        const double a = x[n - k], b = x[n - 2 * k], c = x[n - 4 * k];
        e.estimate = std::log((a - b) / (b - c)) / std::numbers::ln2;
    } else {
        const double lt = std::log(e.threshold);
        double m1 = 0, m2 = 0;
        for (std::size_t i = 1; i <= k; ++i) {
            const double d = std::log(x[n - i]) - lt;
            m1 += d, m2 += d * d;
        }
        m1 /= static_cast<double>(k), m2 /= static_cast<double>(k);
        e.estimate = method == TailIndexMethod::hill ? m1 : m1 + 1 - 0.5 / (1 - m1 * m1 / m2);
    }
    e.standard_error = std::sqrt(variance_factor(method, e.estimate) / static_cast<double>(k));
    e.lower = e.estimate - zq * e.standard_error, e.upper = e.estimate + zq * e.standard_error;
    return e;
}

/// @brief Validates and sorts the data.
/// @param data Observations.
/// @param confidence Interval level.
/// @return Sorted copy.
std::vector<double> sorted_data(const std::vector<double>& data, double confidence) {
    if (!std::ranges::all_of(data, [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("tail_index: data must be finite");
    if (!(confidence > 0 && confidence < 1))
        throw std::invalid_argument("tail_index: confidence must be in (0, 1)");
    auto x = data;
    std::ranges::sort(x);
    return x;
}
} // namespace

TailIndexEstimate tail_index(const std::vector<double>& data, std::size_t k, TailIndexMethod method,
                             double confidence) {
    const auto x = sorted_data(data, confidence);
    const std::size_t n = x.size();
    const std::size_t kmin = method == TailIndexMethod::moment ? 2 : 1;
    if (n < 2 || k < kmin || k > max_k(method, n))
        throw std::invalid_argument("tail_index: k out of range for this method and sample size");
    if (method != TailIndexMethod::pickands && !(x[n - k - 1] > 0))
        throw std::invalid_argument("tail_index: Hill and moment estimators need a positive threshold X_{n-k}");
    return estimate_sorted(x, k, method, random::normal_quantile(1 - (1 - confidence) / 2));
}

std::vector<TailIndexEstimate> tail_index_path(const std::vector<double>& data, TailIndexMethod method,
                                               std::size_t k_min, std::size_t k_max, double confidence) {
    const auto x = sorted_data(data, confidence);
    const std::size_t lowest = method == TailIndexMethod::moment ? 2 : 1;
    if (k_min < lowest || k_min > k_max)
        throw std::invalid_argument("tail_index_path: need lowest admissible k <= k_min <= k_max");
    const double zq = random::normal_quantile(1 - (1 - confidence) / 2);
    std::vector<TailIndexEstimate> out;
    if (x.size() < 2)
        return out;
    const std::size_t top = std::min(k_max, max_k(method, x.size()));
    for (std::size_t k = k_min; k <= top; ++k) {
        if (method != TailIndexMethod::pickands && !(x[x.size() - k - 1] > 0))
            break;
        out.push_back(estimate_sorted(x, k, method, zq));
    }
    return out;
}

double weissman_quantile(const std::vector<double>& data, std::size_t k, double p) {
    if (!(p > 0 && p < 1))
        throw std::invalid_argument("weissman_quantile: p must be in (0, 1)");
    const auto e = tail_index(data, k, TailIndexMethod::hill);
    const double n = static_cast<double>(data.size()), kk = static_cast<double>(k);
    return e.threshold * std::pow((kk + 1) / ((n + 1) * p), e.estimate);
}

} // namespace datamunge::stats
