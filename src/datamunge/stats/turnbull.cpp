/// @file turnbull.cpp
/// @brief Innermost intervals from keyed endpoints and self-consistency iteration for their masses.

#include <datamunge/stats/turnbull.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Internal helpers for the Turnbull estimator.
namespace {

/// @brief An ordered endpoint key: at equal values, closed left ends come first, then right ends, then open left ends.
struct Key {
    double value{}; ///< Endpoint value.
    int rank{};     ///< 0 closed left, 1 right, 2 open left.

    /// @brief Lexicographic order on (value, rank).
    /// @param other Other key.
    /// @return Ordering.
    auto operator<=>(const Key& other) const = default;
};

} // namespace

double TurnbullEstimate::survival(double time) const {
    double s = 1;
    for (const auto& i : intervals)
        if (i.upper < time || (i.upper == time && (i.lower_closed || i.lower < time)))
            s -= i.mass;
    return std::max(s,0.0);
}

TurnbullEstimate turnbull(const std::vector<CensoredTime>& times, const TurnbullOptions& options) {
    const std::size_t n = times.size();
    if (n == 0)
        throw std::invalid_argument("turnbull: no observations");
    std::vector<std::pair<Key,Key>> obs;
    std::vector<std::pair<Key,bool>> ends; // key, is_left
    for (const auto& t : times) {
        if (std::isnan(t.lower) || std::isnan(t.upper) || !(t.lower >= 0) || std::isinf(t.lower) || t.upper < t.lower)
            throw std::invalid_argument("turnbull: invalid censored time");
        const Key left{t.lower,t.lower == t.upper ? 0 : 2}, right{t.upper,1};
        obs.emplace_back(left,right);
        ends.emplace_back(left,true);
        ends.emplace_back(right,false);
    }
    std::ranges::sort(ends);
    TurnbullEstimate e;
    std::vector<std::pair<Key,Key>> inner;
    for (std::size_t k = 0; k+1 < ends.size(); ++k)
        if (ends[k].second && !ends[k+1].second)
            inner.emplace_back(ends[k].first,ends[k+1].first);
    const std::size_t m = inner.size();
    // Membership: interval j lies within observation i.
    std::vector<std::vector<std::size_t>> members(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            if (obs[i].first <= inner[j].first && inner[j].second <= obs[i].second)
                members[i].push_back(j);
    std::vector<double> p(m,1.0/static_cast<double>(m)), d(m);
    const double nd = static_cast<double>(n);
    const auto gradient = [&] {
        std::ranges::fill(d,0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double total = 0;
            for (const std::size_t j : members[i])
                total += p[j];
            for (const std::size_t j : members[i])
                d[j] += 1/total;
        }
    };
    // Self-consistency iterations; after each convergence, drop masses whose d_j is clearly
    // below n (they are converging to zero, which EM approaches only geometrically) and revive
    // any dropped interval whose d_j exceeds n, then iterate again.
    for (e.iterations = 0; e.iterations < options.max_iterations; ++e.iterations) {
        gradient();
        e.kkt_violation = -1;
        for (std::size_t j = 0; j < m; ++j)
            e.kkt_violation = std::max(e.kkt_violation,d[j]/nd-1);
        if (e.kkt_violation <= options.tolerance) {
            bool changed = false;
            for (std::size_t j = 0; j < m; ++j)
                if (p[j] > 0 && d[j]/nd < 1-1e-6) {
                    p[j] = 0;
                    changed = true;
                }
            if (!changed) {
                e.converged = true;
                break;
            }
            double total = 0;
            for (const double v : p)
                total += v;
            for (double& v : p)
                v /= total;
            continue;
        }
        for (std::size_t j = 0; j < m; ++j) {
            if (p[j] == 0 && d[j]/nd > 1+options.tolerance)
                p[j] = 1e-6;
            p[j] *= d[j]/nd;
        }
        double total = 0;
        for (const double v : p)
            total += v;
        for (double& v : p)
            v /= total;
    }
    double total = 0;
    for (const double v : p)
        total += v;
    e.candidate_intervals = m;
    for (std::size_t j = 0; j < m; ++j)
        if (p[j] > 0)
            e.intervals.push_back({inner[j].first.value,inner[j].second.value,inner[j].first.rank == 0,p[j]/total});
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0;
        for (const std::size_t j : members[i])
            s += p[j]/total;
        e.log_likelihood += std::log(s);
    }
    return e;
}

} // namespace datamunge::stats
