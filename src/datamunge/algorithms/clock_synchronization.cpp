/// @file clock_synchronization.cpp
/// @brief Fault-tolerant Berkeley averaging, the quartz drift law, and a Berkeley network simulator.

#include <datamunge/algorithms/clock_synchronization.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::algorithms {

BerkeleyResult berkeley_sync_window(const std::vector<double>& readings, double window) {
    BerkeleyResult r;
    const std::size_t n = readings.size();
    if (n == 0) return r;
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, {}, [&](std::size_t i) { return readings[i]; });
    // Slide a window over the sorted readings: the largest group that fits, smallest spread on ties.
    std::size_t best_lo = 0, best_hi = n;  // [lo, hi) in sorted order
    if (window > 0.0) {
        std::size_t best_count = 0;
        double      best_spread = std::numeric_limits<double>::infinity();
        for (std::size_t lo = 0, hi = 0; lo < n; ++lo) {
            hi = std::max(hi, lo + 1);
            while (hi < n && readings[order[hi]] - readings[order[lo]] <= window) ++hi;
            const std::size_t count  = hi - lo;
            const double      spread = readings[order[hi - 1]] - readings[order[lo]];
            if (count > best_count || (count == best_count && spread < best_spread)) {
                best_count  = count;
                best_spread = spread;
                best_lo     = lo;
                best_hi     = hi;
            }
        }
    }
    r.included.assign(n, 0);
    double sum = 0.0;
    for (std::size_t k = best_lo; k < best_hi; ++k) {
        r.included[order[k]] = 1;
        sum += readings[order[k]];
    }
    r.synchronized_time = sum / static_cast<double>(best_hi - best_lo);
    r.adjustments.resize(n);
    for (std::size_t i = 0; i < n; ++i) r.adjustments[i] = r.synchronized_time - readings[i];
    return r;
}

double crystal_drift_ppm(double temperature_c, double turnover_c, double coefficient) {
    const double d = temperature_c - turnover_c;
    return -coefficient * d * d;
}

BerkeleyNetworkTrace simulate_berkeley_network(const std::vector<std::vector<double>>& drift_ppm,
                                               const BerkeleyNetworkOptions&            o) {
    const std::size_t n = drift_ppm.size();
    if (n == 0 || drift_ppm[0].empty()) throw std::invalid_argument("simulate_berkeley_network: no clocks or no days");
    const std::size_t days = drift_ppm[0].size();
    for (const auto& row : drift_ppm)
        if (row.size() != days) throw std::invalid_argument("simulate_berkeley_network: ragged drift table");
    if (o.master >= n) throw std::invalid_argument("simulate_berkeley_network: master out of range");
    if (!(o.round_interval > 0.0) || o.min_delay < 0.0 || o.max_delay < o.min_delay)
        throw std::invalid_argument("simulate_berkeley_network: bad interval or delays");

    std::mt19937_64                        rng(o.seed);
    std::uniform_real_distribution<double> delay(o.min_delay, o.max_delay);
    const std::size_t rounds = static_cast<std::size_t>(std::floor(days * 86400.0 / o.round_interval));
    BerkeleyNetworkTrace t;
    t.before.assign(n, {});
    t.after.assign(n, {});
    t.included.assign(n, {});
    std::vector<double> offset(n, 0.0), estimate(n);
    for (std::size_t k = 1; k <= rounds; ++k) {
        const double      now = static_cast<double>(k) * o.round_interval;
        const std::size_t day = std::min(days - 1, static_cast<std::size_t>((now - 0.5 * o.round_interval) / 86400.0));
        for (std::size_t i = 0; i < n; ++i) offset[i] += drift_ppm[i][day] * 1e-6 * o.round_interval;
        t.time_days.push_back(now / 86400.0);
        for (std::size_t i = 0; i < n; ++i) t.before[i].push_back(offset[i]);

        std::vector<int> included(n, 1);
        if (o.synchronize) {
            // The master reads each clock relative to its own; the estimate is off by (a - b) / 2.
            for (std::size_t i = 0; i < n; ++i) {
                const double a = delay(rng), b = delay(rng);
                estimate[i]    = i == o.master ? 0.0 : offset[i] - offset[o.master] + (a - b) / 2.0;
            }
            const auto round = berkeley_sync_window(estimate, o.window);
            included         = round.included;
            for (std::size_t i = 0; i < n; ++i) offset[i] += round.adjustments[i];
        }
        const auto spread = [&](const std::vector<double>& v) {
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            for (std::size_t i = 0; i < n; ++i)
                if (included[i]) {
                    lo = std::min(lo, v[i]);
                    hi = std::max(hi, v[i]);
                }
            return hi - lo;
        };
        std::vector<double> before_now(n);
        for (std::size_t i = 0; i < n; ++i) before_now[i] = t.before[i].back();
        t.skew_before.push_back(spread(before_now));
        t.skew_after.push_back(spread(offset));
        t.rejected.push_back(n - static_cast<std::size_t>(std::count(included.begin(), included.end(), 1)));
        for (std::size_t i = 0; i < n; ++i) {
            t.after[i].push_back(offset[i]);
            t.included[i].push_back(included[i]);
        }
    }
    return t;
}

}  // namespace datamunge::algorithms
