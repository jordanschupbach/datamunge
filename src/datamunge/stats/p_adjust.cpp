#include <datamunge/stats/p_adjust.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace datamunge::stats {

namespace {

std::vector<std::size_t> order_ascending(const std::vector<double>& p) {
    std::vector<std::size_t> o(p.size());
    std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&p](std::size_t a, std::size_t b) { return p[a] < p[b]; });
    return o;
}

std::vector<std::size_t> order_descending(const std::vector<double>& p) {
    std::vector<std::size_t> o(p.size());
    std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&p](std::size_t a, std::size_t b) { return p[a] > p[b]; });
    return o;
}

std::vector<double> bonferroni_adjust(const std::vector<double>& p) {
    const double n = static_cast<double>(p.size());
    std::vector<double> result(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) result[i] = std::min(1.0, n * p[i]);
    return result;
}

// Step-down: pmin(1, cummax((n - i + 1) * p[o])) walked in ascending-p order.
std::vector<double> holm_adjust(const std::vector<double>& p) {
    const std::size_t n = p.size();
    const auto o = order_ascending(p);
    std::vector<double> result(n);
    double running_max = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double factor = static_cast<double>(n - k);
        running_max = std::max(running_max, factor * p[o[k]]);
        result[o[k]] = std::min(1.0, running_max);
    }
    return result;
}

// Step-up: pmin(1, cummin((n - i + 1) * p[o])) walked in descending-p order.
std::vector<double> hochberg_adjust(const std::vector<double>& p) {
    const std::size_t n = p.size();
    const auto o = order_descending(p);
    std::vector<double> result(n);
    double running_min = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < n; ++k) {
        const double factor = static_cast<double>(k + 1);
        running_min = std::min(running_min, factor * p[o[k]]);
        result[o[k]] = std::min(1.0, running_min);
    }
    return result;
}

// Benjamini-Hochberg: pmin(1, cummin(n / i * p[o])) walked in descending-p order.
std::vector<double> bh_adjust(const std::vector<double>& p) {
    const std::size_t n = p.size();
    const auto o = order_descending(p);
    std::vector<double> result(n);
    double running_min = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < n; ++k) {
        const double i = static_cast<double>(n - k);
        running_min = std::min(running_min, (static_cast<double>(n) / i) * p[o[k]]);
        result[o[k]] = std::min(1.0, running_min);
    }
    return result;
}

// Benjamini-Yekutieli: same as BH, scaled by the harmonic number sum(1/j, j=1..n).
std::vector<double> by_adjust(const std::vector<double>& p) {
    const std::size_t n = p.size();
    double q = 0.0;
    for (std::size_t j = 1; j <= n; ++j) q += 1.0 / static_cast<double>(j);

    const auto o = order_descending(p);
    std::vector<double> result(n);
    double running_min = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < n; ++k) {
        const double i = static_cast<double>(n - k);
        running_min = std::min(running_min, (q * static_cast<double>(n) / i) * p[o[k]]);
        result[o[k]] = std::min(1.0, running_min);
    }
    return result;
}

// Hommel: step-up procedure, uniformly more powerful than Hochberg. Translated directly
// from R's stats::p.adjust.default (the `hommel` branch), which has no closed form and is
// inherently O(n^2). See that source for the derivation; variable names here (q, pa, j, i1/i2)
// mirror it as closely as 0-indexing allows.
std::vector<double> hommel_adjust(const std::vector<double>& p) {
    const std::size_t n = p.size();
    const auto o = order_ascending(p);
    std::vector<double> p_sorted(n);
    for (std::size_t k = 0; k < n; ++k) p_sorted[k] = p[o[k]];

    double min_np_over_i = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < n; ++k) {
        min_np_over_i = std::min(min_np_over_i, static_cast<double>(n) * p_sorted[k] / static_cast<double>(k + 1));
    }
    std::vector<double> q(n, min_np_over_i);
    std::vector<double> pa(n, min_np_over_i);

    for (long j = static_cast<long>(n) - 1; j >= 2; --j) {
        const std::size_t len_i1 = n - static_cast<std::size_t>(j) + 1; // |{1..n-j+1}|
        double q1 = std::numeric_limits<double>::infinity();
        for (std::size_t t = 0; t < static_cast<std::size_t>(j) - 1; ++t) {
            const std::size_t idx = len_i1 + t;   // 0-indexed position within i2 = (n-j+2)..n
            const double divisor = static_cast<double>(t + 2); // R's 2..j
            q1 = std::min(q1, static_cast<double>(j) * p_sorted[idx] / divisor);
        }
        for (std::size_t k = 0; k < len_i1; ++k) q[k] = std::min(static_cast<double>(j) * p_sorted[k], q1);
        for (std::size_t k = len_i1; k < n; ++k) q[k] = q[len_i1 - 1];
        for (std::size_t k = 0; k < n; ++k) pa[k] = std::max(pa[k], q[k]);
    }

    std::vector<double> result(n);
    for (std::size_t k = 0; k < n; ++k) result[o[k]] = std::max(pa[k], p_sorted[k]);
    return result;
}

} // namespace

std::vector<double> p_adjust(const std::vector<double>& p, PAdjustMethod method) {
    if (p.size() <= 1) return p;

    // Matches R: hommel isn't defined for n == 2, so it falls back to hochberg.
    if (method == PAdjustMethod::Hommel && p.size() == 2) method = PAdjustMethod::Hochberg;

    switch (method) {
        case PAdjustMethod::Bonferroni: return bonferroni_adjust(p);
        case PAdjustMethod::Holm: return holm_adjust(p);
        case PAdjustMethod::Hochberg: return hochberg_adjust(p);
        case PAdjustMethod::Hommel: return hommel_adjust(p);
        case PAdjustMethod::BH: return bh_adjust(p);
        case PAdjustMethod::BY: return by_adjust(p);
        case PAdjustMethod::None:
        default: return p;
    }
}

namespace {

void validate_resample_matrix(const std::vector<std::vector<double>>& resampled_p, std::size_t m,
                               const char* caller) {
    if (resampled_p.empty()) throw std::invalid_argument(std::string(caller) + ": at least one resample is required");
    for (const auto& row : resampled_p) {
        if (row.size() != m)
            throw std::invalid_argument(std::string(caller) + ": every resample row must have length equal to sorted_p");
    }
}

} // namespace

std::vector<double> westfall_young_adjust(const std::vector<std::vector<double>>& resampled_p,
                                           const std::vector<double>& sorted_p) {
    const std::size_t m = sorted_p.size();
    if (m == 0) return {};
    validate_resample_matrix(resampled_p, m, "westfall_young_adjust");
    const std::size_t B = resampled_p.size();

    std::vector<double> resample_min(B);
    for (std::size_t b = 0; b < B; ++b) resample_min[b] = *std::min_element(resampled_p[b].begin(), resampled_p[b].end());

    std::vector<double> adj(m);
    for (std::size_t k = 0; k < m; ++k) {
        std::size_t count = 0;
        for (std::size_t b = 0; b < B; ++b)
            if (resample_min[b] <= sorted_p[k]) ++count;
        adj[k] = static_cast<double>(count) / static_cast<double>(B);
    }
    // Already monotonic by construction (same reference distribution for every k, evaluated at
    // increasing thresholds), but enforce defensively for float-comparison robustness.
    for (std::size_t k = 1; k < m; ++k) adj[k] = std::max(adj[k], adj[k - 1]);
    return adj;
}

std::vector<double> romano_wolf_adjust(const std::vector<std::vector<double>>& resampled_p,
                                        const std::vector<double>& sorted_p) {
    const std::size_t m = sorted_p.size();
    if (m == 0) return {};
    validate_resample_matrix(resampled_p, m, "romano_wolf_adjust");
    const std::size_t B = resampled_p.size();

    // qmin[b] accumulates, via a single backward pass over k = m-1 .. 0, the running minimum of
    // resampled_p[b][k..m-1] -- i.e. the per-resample minimum over hypotheses not yet stepped
    // through, narrowing by one column per step.
    std::vector<double> qmin(B, std::numeric_limits<double>::infinity());
    std::vector<double> adj(m);
    for (long k = static_cast<long>(m) - 1; k >= 0; --k) {
        const auto uk = static_cast<std::size_t>(k);
        std::size_t count = 0;
        for (std::size_t b = 0; b < B; ++b) {
            qmin[b] = std::min(qmin[b], resampled_p[b][uk]);
            if (qmin[b] <= sorted_p[uk]) ++count;
        }
        adj[uk] = static_cast<double>(count) / static_cast<double>(B);
    }
    // Narrowing the comparison set at each step means monotonicity isn't automatic -- enforce it.
    for (std::size_t k = 1; k < m; ++k) adj[k] = std::max(adj[k], adj[k - 1]);
    return adj;
}

} // namespace datamunge::stats
