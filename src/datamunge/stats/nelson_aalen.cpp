/// @file nelson_aalen.cpp
/// @brief Risk-set counting with delayed entry and the cumulative hazard recursion.

#include <datamunge/stats/nelson_aalen.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::stats {

std::vector<NelsonAalenPoint> nelson_aalen(const std::vector<SurvivalObservation>& observations,
                                           const NelsonAalenOptions& options) {
    const std::size_t n = observations.size();
    if (n == 0)
        throw std::invalid_argument("nelson_aalen: no observations");
    if (!(options.confidence > 0 && options.confidence < 1))
        throw std::invalid_argument("nelson_aalen: confidence must be in (0, 1)");
    if (!options.entry.empty() && options.entry.size() != n)
        throw std::invalid_argument("nelson_aalen: need one entry time per observation");
    for (std::size_t i = 0; i < n; ++i) {
        const double t = observations[i].time;
        if (!std::isfinite(t) || t < 0)
            throw std::invalid_argument("nelson_aalen: times must be finite and nonnegative");
        if (!options.entry.empty() && (!std::isfinite(options.entry[i]) || !(options.entry[i] < t)))
            throw std::invalid_argument("nelson_aalen: entry times must be finite and below exit times");
    }
    // Sorted exits (with event flags) and entries; at risk at t = #{exit >= t} - #{entry >= t}.
    std::vector<std::pair<double,bool>> exits;
    std::vector<double> entries;
    for (std::size_t i = 0; i < n; ++i) {
        exits.emplace_back(observations[i].time,observations[i].event);
        if (!options.entry.empty())
            entries.push_back(options.entry[i]);
    }
    std::ranges::sort(exits);
    std::ranges::sort(entries);
    const double z = random::normal_quantile(.5+options.confidence/2);
    std::vector<NelsonAalenPoint> out;
    double h = 0, var = 0;
    for (std::size_t i = 0; i < n;) {
        NelsonAalenPoint p;
        p.time = exits[i].first;
        const auto entered_later = static_cast<std::size_t>(
            entries.end()-std::ranges::lower_bound(entries,p.time));
        p.at_risk = (n-i)-entered_later;
        for (; i < n && exits[i].first == p.time; ++i)
            (exits[i].second ? p.events : p.censored)++;
        const double r = static_cast<double>(p.at_risk), d = static_cast<double>(p.events);
        if (p.events > 0) {
            if (options.ties == HazardTies::nelson_aalen) {
                p.increment = d/r;
                var += d/(r*r);
            } else {
                for (std::size_t j = 0; j < p.events; ++j) {
                    const double m = r-static_cast<double>(j);
                    p.increment += 1/m;
                    var += 1/(m*m);
                }
            }
        }
        h += p.increment;
        p.cumulative_hazard = h;
        p.standard_error = std::sqrt(var);
        p.survival = std::exp(-h);
        if (h > 0 && options.interval == HazardInterval::log) {
            const double f = std::exp(z*p.standard_error/h);
            p.lower = h/f;
            p.upper = h*f;
        } else {
            p.lower = std::max(0.0,h-z*p.standard_error);
            p.upper = h+z*p.standard_error;
        }
        out.push_back(p);
    }
    return out;
}

} // namespace datamunge::stats
