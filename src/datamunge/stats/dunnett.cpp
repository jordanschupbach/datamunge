/// @file dunnett.cpp
/// @brief Pooled statistics, many-to-one correlations and multivariate t tails for Dunnett's procedure.

#include <datamunge/stats/dunnett.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

DunnettResult dunnett_test(const std::vector<std::vector<double>>& groups, const DunnettOptions& options) {
    const std::size_t k = groups.size();
    if (k < 2 || options.control >= k)
        throw std::invalid_argument("dunnett_test: need at least two groups and a valid control index");
    if (!(options.conf_level > 0 && options.conf_level < 1))
        throw std::invalid_argument("dunnett_test: conf_level must be in (0, 1)");
    const Alternative alt = options.alternative;
    if (alt != Alternative::TwoSided && alt != Alternative::Less && alt != Alternative::Greater)
        throw std::invalid_argument("dunnett_test: invalid alternative");
    std::vector<double> mean(k), size(k);
    double ss = 0, total = 0;
    for (std::size_t g = 0; g < k; ++g) {
        if (groups[g].empty() || !std::ranges::all_of(groups[g],[](double v) { return std::isfinite(v); }))
            throw std::invalid_argument("dunnett_test: every group must be nonempty and finite");
        size[g] = static_cast<double>(groups[g].size());
        for (const double v : groups[g])
            mean[g] += v/size[g];
        for (const double v : groups[g])
            ss += (v-mean[g])*(v-mean[g]);
        total += size[g];
    }
    DunnettResult result;
    result.df = total-static_cast<double>(k);
    if (result.df < 1)
        throw std::invalid_argument("dunnett_test: need at least one error degree of freedom");
    result.pooled_sd = std::sqrt(ss/result.df);
    if (!(result.pooled_sd > 0))
        throw std::invalid_argument("dunnett_test: pooled variance is zero");

    const std::size_t c = options.control;
    std::vector<std::size_t> treatments;
    for (std::size_t g = 0; g < k; ++g)
        if (g != c)
            treatments.push_back(g);
    const std::size_t m = treatments.size();
    linalg::DenseMatrix<double> correlation(m,m,1.0);
    for (std::size_t a = 0; a < m; ++a)
        for (std::size_t b = 0; b < m; ++b)
            if (a != b) {
                const double na = size[treatments[a]], nb = size[treatments[b]], n0 = size[c];
                correlation(a,b) = std::sqrt(na*nb/((na+n0)*(nb+n0)));
            }
    constexpr double infinity = std::numeric_limits<double>::infinity();
    const std::vector<double> zero(m,0.0);
    // P(all statistics inside the acceptance region bounded by x).
    const auto inside = [&](double x) {
        const std::vector<double> lo(m,alt == Alternative::TwoSided ? -x : -infinity), hi(m,x);
        const auto p = mvt_rectangle_probability(lo,hi,zero,correlation,result.df,options.integration);
        result.integration_error = std::max(result.integration_error,p.error);
        return p.value;
    };
    for (const std::size_t g : treatments) {
        DunnettComparison comparison;
        comparison.treatment = g;
        comparison.estimate = mean[g]-mean[c];
        comparison.std_error = result.pooled_sd*std::sqrt(1/size[g]+1/size[c]);
        comparison.statistic = comparison.estimate/comparison.std_error;
        const double bound = alt == Alternative::TwoSided ? std::abs(comparison.statistic)
                             : alt == Alternative::Greater ? comparison.statistic : -comparison.statistic;
        comparison.p_value = std::clamp(1-inside(bound),0.0,1.0);
        result.comparisons.push_back(comparison);
    }
    // Illinois false position on inside(x) - conf_level; the fixed seed makes it a smooth function of x.
    double lo = 0, hi = 2;
    double f_lo = inside(lo)-options.conf_level, f_hi = inside(hi)-options.conf_level;
    while (f_hi < 0) {
        lo = hi;
        f_lo = f_hi;
        hi *= 2;
        f_hi = inside(hi)-options.conf_level;
    }
    double x = hi;
    int side = 0;
    for (int it = 0; it < 100 && hi-lo > 1e-9; ++it) {
        x = (lo*f_hi-hi*f_lo)/(f_hi-f_lo);
        const double fx = inside(x)-options.conf_level;
        if (std::abs(fx) < 1e-12)
            break;
        if ((fx < 0) == (f_lo < 0)) {
            lo = x;
            f_lo = fx;
            if (side == -1)
                f_hi /= 2;
            side = -1;
        } else {
            hi = x;
            f_hi = fx;
            if (side == 1)
                f_lo /= 2;
            side = 1;
        }
    }
    result.critical_value = x;
    for (auto& comparison : result.comparisons) {
        const double half = result.critical_value*comparison.std_error;
        comparison.conf_low = alt == Alternative::Less ? -infinity : comparison.estimate-half;
        comparison.conf_high = alt == Alternative::Greater ? infinity : comparison.estimate+half;
    }
    return result;
}

} // namespace datamunge::stats
