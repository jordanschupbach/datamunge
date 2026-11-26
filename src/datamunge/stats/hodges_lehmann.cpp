/// @file hodges_lehmann.cpp
/// @brief Walsh-average and pairwise-difference medians with signed-rank and Mann–Whitney intervals.

#include <datamunge/stats/hodges_lehmann.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace datamunge::stats {
/// @brief Null distributions and order-statistic interval selection.
namespace {

/// @brief Largest sample for which the exact signed-rank distribution is tabulated.
constexpr std::size_t max_exact_signed_rank = 1000;

/// @brief Largest number of pairwise differences for which the exact Mann–Whitney distribution is tabulated.
constexpr std::size_t max_exact_rank_sum = 40000;

/// @brief Throws unless every value is finite.
/// @param values Values to check.
/// @param name Argument name for the message.
/// @throws std::invalid_argument On a nonfinite value.
void check_finite(const std::vector<double>& values, const char* name) {
    if (!std::ranges::all_of(values,[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument(std::string("hodges_lehmann: ")+name+" must be finite");
}

/// @brief Whether a sample repeats a value.
/// @param values Sample.
/// @return True when two entries compare equal.
bool has_ties(const std::vector<double>& values) {
    std::unordered_set<double> seen;
    for (const double v : values)
        if (!seen.insert(v).second)
            return true;
    return false;
}

/// @brief Null probabilities \f$P(V=v)\f$ of the signed-rank statistic for n untied values.
/// @param n Sample size.
/// @return Probabilities indexed by \f$v=0,\dots,n(n+1)/2\f$.
std::vector<double> signed_rank_pmf(std::size_t n) {
    std::vector<double> pmf(n*(n+1)/2+1,0.0);
    pmf[0] = 1;
    std::size_t top = 0;
    for (std::size_t k = 1; k <= n; ++k) {
        top += k;
        for (std::size_t v = top; v >= k; --v)
            pmf[v] = .5*(pmf[v]+pmf[v-k]);
        for (std::size_t v = 0; v < k && v <= top; ++v)
            pmf[v] *= .5;
    }
    return pmf;
}

/// @brief Null probabilities \f$P(U=u)\f$ of the Mann–Whitney statistic for untied samples.
///
/// Uses \f$p_{i,j}(u)=\tfrac{i}{i+j}p_{i-1,j}(u-j)+\tfrac{j}{i+j}p_{i,j-1}(u)\f$, keeping one row of j.
/// @param m First sample size.
/// @param n Second sample size.
/// @return Probabilities indexed by \f$u=0,\dots,mn\f$.
std::vector<double> rank_sum_pmf(std::size_t m, std::size_t n) {
    // row[j] holds p_{i,j}; p_{0,j} and p_{i,0} are point masses at zero.
    std::vector<std::vector<double>> row(n+1,std::vector<double>{1.0});
    for (std::size_t i = 1; i <= m; ++i) {
        std::vector<std::vector<double>> next(n+1);
        next[0] = {1.0};
        for (std::size_t j = 1; j <= n; ++j) {
            std::vector<double> p(i*j+1,0.0);
            const double a = static_cast<double>(i)/static_cast<double>(i+j);
            for (std::size_t u = 0; u < row[j].size(); ++u)
                p[u+j] += a*row[j][u];
            for (std::size_t u = 0; u < next[j-1].size(); ++u)
                p[u] += (1-a)*next[j-1][u];
            next[j] = std::move(p);
        }
        row = std::move(next);
    }
    return row[n];
}

/// @brief Smallest q with \f$P(S\le q)\ge p\f$ under a tabulated null.
/// @param pmf Null probabilities from zero.
/// @param p Lower-tail probability.
/// @return The quantile.
std::size_t exact_quantile(const std::vector<double>& pmf, double p) {
    double cdf = 0;
    for (std::size_t q = 0; q < pmf.size(); ++q) {
        cdf += pmf[q];
        if (cdf >= p*(1-64*std::numeric_limits<double>::epsilon()))
            return q;
    }
    return pmf.size()-1;
}

/// @brief \f$P(S\le q)\f$ under a tabulated null, zero for negative q.
/// @param pmf Null probabilities from zero.
/// @param q Bound, possibly negative.
/// @return Lower-tail probability.
double exact_cdf(const std::vector<double>& pmf, long long q) {
    double cdf = 0;
    for (long long v = 0; v <= q && v < static_cast<long long>(pmf.size()); ++v)
        cdf += pmf[static_cast<std::size_t>(v)];
    return cdf;
}

/// @brief Builds the estimate and order-statistic interval from sorted values and a null distribution.
/// @param sorted Sorted Walsh averages or differences.
/// @param options Coverage and sidedness.
/// @param exact Whether to use the tabulated null.
/// @param pmf Tabulated null probabilities (unused when not exact).
/// @param sd Null standard deviation of the rank statistic (normal method).
/// @return The completed result.
HodgesLehmannResult interval_from_order_statistics(const std::vector<double>& sorted,
                                                   const HodgesLehmannOptions& options, bool exact,
                                                   const std::vector<double>& pmf, double sd) {
    const std::size_t total = sorted.size();
    HodgesLehmannResult result;
    result.pair_count = total;
    result.exact = exact;
    result.estimate = total%2 == 1 ? sorted[total/2] : .5*(sorted[total/2-1]+sorted[total/2]);
    const bool two_sided = options.alternative == Alternative::TwoSided;
    const double tail = (1-options.conf_level)/(two_sided ? 2.0 : 1.0);
    const double centre = static_cast<double>(total)/2;
    std::size_t q;
    double tail_below;
    if (exact) {
        q = std::max<std::size_t>(exact_quantile(pmf,tail),1);
        tail_below = exact_cdf(pmf,static_cast<long long>(q)-1);
    } else {
        const double z = random::normal_quantile(tail);
        q = static_cast<std::size_t>(std::max(1.0,std::ceil(centre-.5+z*sd-1e-9)));
        tail_below = random::normal_cdf((static_cast<double>(q)-.5-centre)/sd);
    }
    q = std::min(q,(total+1)/2);
    constexpr double infinity = std::numeric_limits<double>::infinity();
    result.lower_rank = options.alternative == Alternative::Less ? 0 : q;
    result.upper_rank = options.alternative == Alternative::Greater ? 0 : total-q+1;
    result.conf_int_lower = result.lower_rank == 0 ? -infinity : sorted[result.lower_rank-1];
    result.conf_int_upper = result.upper_rank == 0 ? infinity : sorted[result.upper_rank-1];
    result.achieved_conf_level = 1-(two_sided ? 2.0 : 1.0)*tail_below;
    return result;
}

/// @brief Validates the options shared by both estimators.
/// @param options Options to check.
/// @throws std::invalid_argument For a coverage outside (0, 1) or an unknown alternative.
void check_options(const HodgesLehmannOptions& options) {
    if (!(options.conf_level > 0 && options.conf_level < 1))
        throw std::invalid_argument("hodges_lehmann: conf_level must be in (0, 1)");
    if (options.alternative != Alternative::TwoSided && options.alternative != Alternative::Less &&
        options.alternative != Alternative::Greater)
        throw std::invalid_argument("hodges_lehmann: invalid alternative");
}

} // namespace

HodgesLehmannResult hodges_lehmann_location(const std::vector<double>& x, const HodgesLehmannOptions& options) {
    check_options(options);
    if (x.size() < 2)
        throw std::invalid_argument("hodges_lehmann_location: need at least two values");
    check_finite(x,"x");
    const std::size_t n = x.size();
    const bool exact = options.method == HodgesLehmannInterval::exact ||
                       (options.method == HodgesLehmannInterval::automatic && n < 50 && !has_ties(x));
    if (exact && n > max_exact_signed_rank)
        throw std::invalid_argument("hodges_lehmann_location: exact interval supports at most 1000 values");
    std::vector<double> walsh;
    walsh.reserve(n*(n+1)/2);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j)
            walsh.push_back(.5*(x[i]+x[j]));
    std::ranges::sort(walsh);
    const double nd = static_cast<double>(n);
    return interval_from_order_statistics(walsh,options,exact,exact ? signed_rank_pmf(n) : std::vector<double>{},
                                          std::sqrt(nd*(nd+1)*(2*nd+1)/24));
}

HodgesLehmannResult hodges_lehmann_shift(const std::vector<double>& x, const std::vector<double>& y,
                                         const HodgesLehmannOptions& options) {
    check_options(options);
    if (x.empty() || y.empty())
        throw std::invalid_argument("hodges_lehmann_shift: both samples must be nonempty");
    check_finite(x,"x");
    check_finite(y,"y");
    const std::size_t m = x.size(), n = y.size();
    std::vector<double> pooled(x);
    pooled.insert(pooled.end(),y.begin(),y.end());
    const bool exact = options.method == HodgesLehmannInterval::exact ||
                       (options.method == HodgesLehmannInterval::automatic && m < 50 && n < 50 && !has_ties(pooled));
    if (exact && m*n > max_exact_rank_sum)
        throw std::invalid_argument("hodges_lehmann_shift: exact interval supports at most 40000 differences");
    std::vector<double> differences;
    differences.reserve(m*n);
    for (const double a : x)
        for (const double b : y)
            differences.push_back(a-b);
    std::ranges::sort(differences);
    const double md = static_cast<double>(m), nd = static_cast<double>(n);
    return interval_from_order_statistics(differences,options,exact,exact ? rank_sum_pmf(m,n) : std::vector<double>{},
                                          std::sqrt(md*nd*(md+nd+1)/12));
}

} // namespace datamunge::stats
