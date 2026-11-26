/// @file staggered_did.cpp
/// @brief Callaway-Sant'Anna group-time effects and aggregations.
#include <datamunge/causal/staggered_did.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Influence values of one estimate, one per unit.
using Influence = std::vector<double>;

/// @brief Standard error from an influence function scaled to the full sample: sqrt(sum IF^2) / n.
double influence_se(const Influence& f) {
    double ss = 0;
    for (const double v : f)
        ss += v * v;
    return std::sqrt(ss) / static_cast<double>(f.size());
}

/// @brief Fills the confidence limits of an aggregated effect.
AggregatedEffect make_effect(double key, double att, const Influence& f, double z) {
    const double se = influence_se(f);
    return {key, att, se, att - z * se, att + z * se};
}

/// @brief Weighted combination of group-time effects, with the influence of estimated cohort shares.
///
/// Implements R did's aggregation: weights pg_k / sum(pg_keep), and the weight influence function
/// (wif) added to the combination of the group-time influence functions.
std::pair<double, Influence> combine(const std::vector<std::size_t>& keep, const std::vector<double>& att,
                                     const std::vector<double>& pg, const std::vector<double>& group,
                                     const std::vector<Influence>& inf, const std::vector<double>& unit_group) {
    const std::size_t n = unit_group.size();
    double total = 0;
    for (const std::size_t k : keep)
        total += pg[k];
    double estimate = 0;
    Influence f(n, 0.0);
    for (const std::size_t k : keep) {
        const double w = pg[k] / total;
        estimate += w * att[k];
        for (std::size_t i = 0; i < n; ++i)
            f[i] += w * inf[k][i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        double row_sum = 0; // sum_k (1{G_i = g_k} - pg_k)
        for (const std::size_t k : keep)
            row_sum += (unit_group[i] == group[k] ? 1.0 : 0.0) - pg[k];
        for (const std::size_t k : keep) {
            const double if1 = ((unit_group[i] == group[k] ? 1.0 : 0.0) - pg[k]) / total;
            const double if2 = row_sum * pg[k] / (total * total);
            f[i] += (if1 - if2) * att[k];
        }
    }
    return {estimate, f};
}
} // namespace

StaggeredDid staggered_did(const std::vector<double>& unit, const std::vector<double>& period,
                           const std::vector<double>& cohort, const std::vector<double>& outcome,
                           const StaggeredDidOptions& options) {
    const std::size_t rows = unit.size();
    if (rows == 0 || period.size() != rows || cohort.size() != rows || outcome.size() != rows)
        throw std::invalid_argument("staggered_did: unit, period, cohort and outcome must have equal length");
    if (!(options.confidence > 0 && options.confidence < 1))
        throw std::invalid_argument("staggered_did: confidence must lie in (0, 1)");
    for (std::size_t r = 0; r < rows; ++r)
        if (!std::isfinite(unit[r]) || !std::isfinite(period[r]) || !std::isfinite(cohort[r]) ||
            !std::isfinite(outcome[r]) || cohort[r] < 0)
            throw std::invalid_argument("staggered_did: values must be finite and cohorts nonnegative");
    const std::set<double> period_set(period.begin(), period.end());
    const std::vector<double> periods(period_set.begin(), period_set.end());
    std::map<double, std::size_t> unit_index, period_index;
    for (std::size_t t = 0; t < periods.size(); ++t)
        period_index[periods[t]] = t;
    for (const double u : unit)
        unit_index.emplace(u, unit_index.size());
    const std::size_t all_units = unit_index.size(), T = periods.size();
    if (rows != all_units * T)
        throw std::invalid_argument("staggered_did: the panel must be balanced (every unit in every period once)");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::vector<double>> y(all_units, std::vector<double>(T, nan));
    std::vector<double> g_of(all_units, nan);
    for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t i = unit_index[unit[r]], t = period_index[period[r]];
        if (!std::isnan(y[i][t]))
            throw std::invalid_argument("staggered_did: duplicate unit-period observation");
        y[i][t] = outcome[r];
        // Cohorts after the last period are never treated within the panel.
        const double g = cohort[r] > periods.back() ? 0.0 : cohort[r];
        if (!std::isnan(g_of[i]) && g_of[i] != g)
            throw std::invalid_argument("staggered_did: cohort must be constant within a unit");
        g_of[i] = g;
    }
    // Drop units already treated in the first period.
    std::vector<std::size_t> kept;
    for (std::size_t i = 0; i < all_units; ++i)
        if (g_of[i] == 0 || g_of[i] > periods.front())
            kept.push_back(i);
    const std::size_t n = kept.size();
    std::vector<double> unit_group(n);
    std::map<double, std::size_t> cohort_size;
    for (std::size_t k = 0; k < n; ++k) {
        unit_group[k] = g_of[kept[k]];
        if (unit_group[k] > 0)
            ++cohort_size[unit_group[k]];
    }
    if (cohort_size.empty())
        throw std::invalid_argument("staggered_did: no cohort is treated within the panel");
    const double a = static_cast<double>(options.anticipation);
    const double z = random::normal_quantile(0.5 + options.confidence / 2);
    StaggeredDid out;
    out.units = n;
    std::vector<Influence> inf;
    std::vector<double> att, pg, group, time;
    for (const auto& [g, size] : cohort_size) {
        // Base period of post-treatment effects: the last period with t + anticipation < g.
        std::ptrdiff_t post_base = -1;
        for (std::size_t t = 0; t < T; ++t)
            if (periods[t] + a < g)
                post_base = static_cast<std::ptrdiff_t>(t);
        if (post_base < 0)
            continue;
        for (std::size_t t = 0; t < T; ++t) {
            const bool post = periods[t] + a >= g;
            std::ptrdiff_t b = post_base;
            if (!post && options.base_period == DidBasePeriod::varying)
                b = static_cast<std::ptrdiff_t>(t) - 1;
            if (b < 0)
                continue;
            const double base = periods[static_cast<std::size_t>(b)];
            GroupTimeEffect e{.group = g, .time = periods[t], .base = base};
            Influence f(n, 0.0);
            std::vector<std::size_t> treated, controls;
            for (std::size_t k = 0; k < n; ++k) {
                const double gk = unit_group[k];
                if (gk == g)
                    treated.push_back(k);
                else if (gk == 0 || (options.control_group == DidControlGroup::not_yet_treated &&
                                     gk > std::max(periods[t], base) + a))
                    controls.push_back(k);
            }
            e.treated = treated.size();
            e.controls = controls.size();
            if (static_cast<std::size_t>(b) == t) { // universal base period: zero by construction
                e.att = 0;
                e.standard_error = e.lower = e.upper = nan;
            } else {
                if (controls.empty())
                    throw std::invalid_argument("staggered_did: no comparison units for a group-time effect");
                const auto change = [&](std::size_t k) { return y[kept[k]][t] - y[kept[k]][static_cast<std::size_t>(b)]; };
                double mt = 0, mc = 0;
                for (const std::size_t k : treated)
                    mt += change(k) / static_cast<double>(treated.size());
                for (const std::size_t k : controls)
                    mc += change(k) / static_cast<double>(controls.size());
                e.att = mt - mc;
                const double nn = static_cast<double>(n);
                for (const std::size_t k : treated)
                    f[k] = nn * (change(k) - mt) / static_cast<double>(treated.size());
                for (const std::size_t k : controls)
                    f[k] = -nn * (change(k) - mc) / static_cast<double>(controls.size());
                e.standard_error = influence_se(f);
                e.lower = e.att - z * e.standard_error;
                e.upper = e.att + z * e.standard_error;
            }
            out.group_time.push_back(e);
            inf.push_back(std::move(f));
            att.push_back(e.att);
            pg.push_back(static_cast<double>(size) / static_cast<double>(n));
            group.push_back(g);
            time.push_back(periods[t]);
        }
    }
    const std::size_t K = att.size();
    // Event study: all event times.
    std::map<double, std::vector<std::size_t>> by_event;
    for (std::size_t k = 0; k < K; ++k)
        by_event[time[k] - group[k]].push_back(k);
    Influence dynamic_f(n, 0.0);
    double dynamic_sum = 0;
    std::size_t dynamic_count = 0;
    for (const auto& [e, keep] : by_event) {
        auto [estimate, f] = combine(keep, att, pg, group, inf, unit_group);
        auto effect = make_effect(e, estimate, f, z);
        // The universal base period is zero by construction and has no sampling variability.
        if (std::ranges::any_of(keep, [&](std::size_t k) { return std::isnan(out.group_time[k].standard_error); }))
            effect.standard_error = effect.lower = effect.upper = std::numeric_limits<double>::quiet_NaN();
        out.event_study.push_back(effect);
        if (e >= 0) {
            dynamic_sum += estimate;
            ++dynamic_count;
            for (std::size_t i = 0; i < n; ++i)
                dynamic_f[i] += f[i];
        }
    }
    if (dynamic_count > 0) {
        for (double& v : dynamic_f)
            v /= static_cast<double>(dynamic_count);
        out.dynamic_overall = make_effect(0, dynamic_sum / static_cast<double>(dynamic_count), dynamic_f, z);
    }
    // Cohort averages over post-treatment periods, then weighted by cohort share.
    std::vector<double> cohort_att, cohort_pg, cohort_key;
    std::vector<Influence> cohort_inf;
    for (const auto& [g, size] : cohort_size) {
        std::vector<std::size_t> keep;
        for (std::size_t k = 0; k < K; ++k)
            if (group[k] == g && time[k] >= g)
                keep.push_back(k);
        if (keep.empty())
            continue;
        double estimate = 0;
        Influence f(n, 0.0);
        for (const std::size_t k : keep) {
            estimate += att[k] / static_cast<double>(keep.size());
            for (std::size_t i = 0; i < n; ++i)
                f[i] += inf[k][i] / static_cast<double>(keep.size());
        }
        out.by_group.push_back(make_effect(g, estimate, f, z));
        cohort_att.push_back(estimate);
        cohort_pg.push_back(static_cast<double>(size) / static_cast<double>(n));
        cohort_key.push_back(g);
        cohort_inf.push_back(std::move(f));
    }
    if (!cohort_att.empty()) {
        std::vector<std::size_t> all(cohort_att.size());
        for (std::size_t c = 0; c < all.size(); ++c)
            all[c] = c;
        auto [estimate, f] = combine(all, cohort_att, cohort_pg, cohort_key, cohort_inf, unit_group);
        out.group_overall = make_effect(0, estimate, f, z);
    }
    std::vector<std::size_t> post;
    for (std::size_t k = 0; k < K; ++k)
        if (time[k] >= group[k])
            post.push_back(k);
    if (!post.empty()) {
        auto [estimate, f] = combine(post, att, pg, group, inf, unit_group);
        out.simple = make_effect(0, estimate, f, z);
    }
    return out;
}

} // namespace datamunge::causal
