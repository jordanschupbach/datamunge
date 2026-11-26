/// @file survival_tests.cpp
/// @brief Per-stratum risk-set sweep for weighted observed-minus-expected events and their covariance.

#include <datamunge/stats/survival_tests.hpp>

#include <datamunge/linalg/lu.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

SurvivalGroupTest survival_group_test(const std::vector<SurvivalObservation>& observations, const std::vector<int>& groups,
                                      const SurvivalGroupTestOptions& options) {
    const std::size_t n = observations.size();
    if (groups.size() != n || (!options.strata.empty() && options.strata.size() != n))
        throw std::invalid_argument("survival_group_test: need one group (and stratum) per observation");
    if (!(options.rho >= 0) || !(options.gamma >= 0) || !std::isfinite(options.rho) || !std::isfinite(options.gamma))
        throw std::invalid_argument("survival_group_test: weight exponents must be finite and nonnegative");
    for (const auto& o : observations)
        if (!std::isfinite(o.time) || o.time < 0)
            throw std::invalid_argument("survival_group_test: times must be finite and nonnegative");
    SurvivalGroupTest r;
    r.groups = groups;
    std::ranges::sort(r.groups);
    r.groups.erase(std::unique(r.groups.begin(),r.groups.end()),r.groups.end());
    const std::size_t g = r.groups.size();
    if (g < 2)
        throw std::invalid_argument("survival_group_test: need at least two groups");
    std::vector<std::size_t> index(n);
    for (std::size_t i = 0; i < n; ++i)
        index[i] = static_cast<std::size_t>(std::ranges::lower_bound(r.groups,groups[i])-r.groups.begin());
    r.observed.assign(g,0.0);
    r.expected.assign(g,0.0);
    r.variance = linalg::DenseMatrix<double>(g,g);

    // Order by stratum, then time, events before censorings (as survdiff).
    std::vector<std::size_t> order(n);
    std::iota(order.begin(),order.end(),0);
    const auto stratum = [&](std::size_t i) { return options.strata.empty() ? 0 : options.strata[i]; };
    std::ranges::sort(order,[&](std::size_t a, std::size_t b) {
        if (stratum(a) != stratum(b))
            return stratum(a) < stratum(b);
        if (observations[a].time != observations[b].time)
            return observations[a].time < observations[b].time;
        return observations[a].event > observations[b].event;
    });
    for (std::size_t begin = 0; begin < n;) {
        std::size_t end = begin;
        while (end < n && stratum(order[end]) == stratum(order[begin]))
            ++end;
        // Left-continuous pooled Kaplan–Meier within the stratum.
        std::vector<double> km(end-begin);
        double s = 1;
        for (std::size_t i = begin; i < end;) {
            std::size_t j = i;
            double deaths = 0;
            while (j < end && observations[order[j]].time == observations[order[i]].time) {
                km[j-begin] = s;
                deaths += observations[order[j]].event ? 1 : 0;
                ++j;
            }
            const double risk = static_cast<double>(end-i);
            s *= (risk-deaths)/risk;
            i = j;
        }
        // Backward sweep accumulates the risk set by group.
        std::vector<double> risk(g,0.0);
        for (std::size_t i = end; i > begin;) {
            const double t = observations[order[i-1]].time;
            const double surv = km[i-1-begin];
            double w = 1;
            if (options.rho != 0)
                w *= std::pow(surv,options.rho);
            if (options.gamma != 0)
                w *= std::pow(1-surv,options.gamma);
            double deaths = 0;
            std::size_t j = i;
            while (j > begin && observations[order[j-1]].time == t) {
                const auto& o = observations[order[j-1]];
                const std::size_t k = index[order[j-1]];
                risk[k] += 1;
                if (o.event) {
                    deaths += 1;
                    r.observed[k] += w;
                }
                --j;
            }
            i = j;
            const double total = static_cast<double>(end-i);
            if (deaths > 0) {
                for (std::size_t k = 0; k < g; ++k)
                    r.expected[k] += w*deaths*risk[k]/total;
                if (total > 1) {
                    const double w2 = w*w;
                    for (std::size_t a = 0; a < g; ++a) {
                        const double tmp = w2*deaths*risk[a]*(total-deaths)/(total*(total-1));
                        r.variance(a,a) += tmp;
                        for (std::size_t b = 0; b < g; ++b)
                            r.variance(a,b) -= tmp*risk[b]/total;
                    }
                }
            }
        }
        begin = end;
    }
    // Quadratic form over groups with positive expectation, dropping the first of them.
    std::vector<std::size_t> keep;
    for (std::size_t k = 0; k < g; ++k)
        if (r.expected[k] > 0)
            keep.push_back(k);
    if (keep.size() < 2) {
        r.chi_square = 0;
        r.degrees_of_freedom = 0;
        r.p_value = 1;
        return r;
    }
    const std::size_t m = keep.size()-1;
    linalg::DenseMatrix<double> v(m,m);
    std::vector<double> diff(m);
    for (std::size_t a = 0; a < m; ++a) {
        diff[a] = r.observed[keep[a+1]]-r.expected[keep[a+1]];
        for (std::size_t b = 0; b < m; ++b)
            v(a,b) = r.variance(keep[a+1],keep[b+1]);
    }
    std::vector<double> sol;
    try {
        sol = linalg::lu(v).solve(diff);
    } catch (const std::exception&) {
        throw std::invalid_argument("survival_group_test: covariance of observed minus expected is singular");
    }
    for (std::size_t a = 0; a < m; ++a)
        r.chi_square += sol[a]*diff[a];
    r.degrees_of_freedom = m;
    r.p_value = detail::chi_squared_test_survival(r.chi_square,static_cast<double>(m));
    return r;
}

} // namespace datamunge::stats
