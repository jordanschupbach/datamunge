/// @file tolerance_limits.cpp
/// @brief Exact and approximate normal tolerance factors and Wilks order-statistic limits.

#include <datamunge/stats/tolerance_limits.hpp>

#include <datamunge/random/distributions.hpp>
#include <datamunge/random/noncentral_distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Coverage integrals, binomial tails and argument checks.
namespace {

/// @brief Positive infinity.
constexpr double infinity = std::numeric_limits<double>::infinity();

/// @brief Throws unless a level lies strictly inside (0, 1).
/// @param value Level to check.
/// @param name Argument name.
/// @throws std::invalid_argument When outside.
void check_level(double value, const char* name) {
    if (!(value > 0 && value < 1))
        throw std::invalid_argument(std::string("tolerance: ")+name+" must be in (0, 1)");
}

/// @brief Standard normal distribution function.
/// @param x Argument.
/// @return \f$\Phi(x)\f$.
double phi(double x) {
    return .5*std::erfc(-x/std::numbers::sqrt2);
}

/// @brief Half-width r with \f$\Phi(z+r)-\Phi(z-r)=P\f$.
/// @param z Center offset.
/// @param coverage Fraction P.
/// @return r.
double half_width(double z, double coverage) {
    double lo = 0, hi = random::normal_quantile((1+coverage)/2)+std::abs(z)+1;
    for (int it = 0; it < 200 && hi-lo > 1e-15*hi; ++it) {
        const double mid = .5*(lo+hi);
        (phi(z+mid)-phi(z-mid) < coverage ? lo : hi) = mid;
    }
    return .5*(lo+hi);
}

/// @brief Exact two-sided normal tolerance factor by Simpson integration over z and bisection in k.
/// @param n Sample size.
/// @param coverage Fraction P.
/// @param confidence Level.
/// @return The factor.
double exact_two_sided_factor(std::size_t n, double coverage, double confidence) {
    const double nd = static_cast<double>(n), df = nd-1;
    constexpr int panels = 4000; // even
    const double top = 10/std::sqrt(nd), h = top/panels;
    std::vector<double> weight(panels+1), r2(panels+1);
    for (int i = 0; i <= panels; ++i) {
        const double z = i*h;
        const double simpson = (i == 0 || i == panels) ? 1 : (i%2 ? 4 : 2);
        weight[i] = simpson*h/3*std::sqrt(2*nd/std::numbers::pi)*std::exp(-nd*z*z/2);
        const double r = half_width(z,coverage);
        r2[i] = r*r;
    }
    const auto confidence_at = [&](double k) {
        double total = 0;
        for (int i = 0; i <= panels; ++i)
            total += weight[i]*(1-random::chi_squared_cdf(df*r2[i]/(k*k),df));
        return total;
    };
    double lo = random::normal_quantile((1+coverage)/2)*.5, hi = 2*lo+1;
    while (confidence_at(hi) < confidence)
        hi *= 2;
    for (int it = 0; it < 200 && hi-lo > 1e-12*hi; ++it) {
        const double mid = .5*(lo+hi);
        (confidence_at(mid) < confidence ? lo : hi) = mid;
    }
    return .5*(lo+hi);
}

/// @brief \f$P\{\mathrm{Bin}(n,p)\le k\}\f$ by summing the probability mass in log space.
/// @param k Upper count (negative gives zero).
/// @param n Trials.
/// @param p Success probability.
/// @return Lower tail.
double binomial_lower_tail(long long k, std::size_t n, double p) {
    if (k < 0)
        return 0;
    double total = 0;
    const double nd = static_cast<double>(n);
    for (long long j = 0; j <= k && j <= static_cast<long long>(n); ++j) {
        const double jd = static_cast<double>(j);
        total += std::exp(std::lgamma(nd+1)-std::lgamma(jd+1)-std::lgamma(nd-jd+1)+jd*std::log(p)+
                          (nd-jd)*std::log1p(-p));
    }
    return std::min(total,1.0);
}

} // namespace

double normal_tolerance_factor(std::size_t n, double coverage, double confidence, ToleranceSide side,
                               NormalToleranceMethod method) {
    if (n < 2)
        throw std::invalid_argument("normal_tolerance_factor: need at least two observations");
    check_level(coverage,"coverage");
    check_level(confidence,"confidence");
    const double nd = static_cast<double>(n);
    if (side != ToleranceSide::two_sided)
        return random::noncentral_t_quantile(confidence,nd-1,random::normal_quantile(coverage)*std::sqrt(nd))/
               std::sqrt(nd);
    if (method == NormalToleranceMethod::howe) {
        const double z = random::normal_quantile((1+coverage)/2);
        return std::sqrt((nd-1)*(1+1/nd)*z*z/random::chi_squared_quantile(1-confidence,nd-1));
    }
    return exact_two_sided_factor(n,coverage,confidence);
}

ToleranceInterval normal_tolerance_interval(const std::vector<double>& x, double coverage, double confidence,
                                            ToleranceSide side, NormalToleranceMethod method) {
    if (x.size() < 2 || !std::ranges::all_of(x,[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("normal_tolerance_interval: need at least two finite values");
    const double n = static_cast<double>(x.size());
    double mean = 0, ss = 0;
    for (const double v : x)
        mean += v/n;
    for (const double v : x)
        ss += (v-mean)*(v-mean);
    const double sd = std::sqrt(ss/(n-1));
    if (!(sd > 0))
        throw std::invalid_argument("normal_tolerance_interval: sample standard deviation must be positive");
    ToleranceInterval result;
    result.factor = normal_tolerance_factor(x.size(),coverage,confidence,side,method);
    result.lower = side == ToleranceSide::upper ? -infinity : mean-result.factor*sd;
    result.upper = side == ToleranceSide::lower ? infinity : mean+result.factor*sd;
    result.achieved_confidence = confidence;
    return result;
}

ToleranceInterval nonparametric_tolerance_interval(const std::vector<double>& x, double coverage, double confidence,
                                                   ToleranceSide side) {
    if (x.empty() || !std::ranges::all_of(x,[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("nonparametric_tolerance_interval: need nonempty finite values");
    check_level(coverage,"coverage");
    check_level(confidence,"confidence");
    std::vector<double> sorted(x);
    std::ranges::sort(sorted);
    const std::size_t n = sorted.size();
    const long long nl = static_cast<long long>(n);
    ToleranceInterval result;
    result.lower = -infinity;
    result.upper = infinity;
    if (side == ToleranceSide::two_sided) {
        // Largest r with P(Bin(n, P) <= n - 2r) >= confidence.
        std::size_t best = 0;
        for (std::size_t r = 1; 2*r <= n; ++r) {
            if (binomial_lower_tail(nl-2*static_cast<long long>(r),n,coverage) >= confidence)
                best = r;
            else
                break;
        }
        const std::size_t r = std::max<std::size_t>(best,1);
        result.achieved_confidence = binomial_lower_tail(nl-2*static_cast<long long>(r),n,coverage);
        result.feasible = best > 0;
        if (result.feasible) {
            result.lower_rank = r;
            result.upper_rank = n-r+1;
            result.lower = sorted[r-1];
            result.upper = sorted[n-r];
        }
        return result;
    }
    // Smallest m with P(Bin(n, P) <= m - 1) >= confidence; an upper bound uses X_(m), a lower X_(n-m+1).
    std::size_t m = n+1;
    for (std::size_t k = 1; k <= n; ++k)
        if (binomial_lower_tail(static_cast<long long>(k)-1,n,coverage) >= confidence) {
            m = k;
            break;
        }
    result.feasible = m <= n;
    result.achieved_confidence = binomial_lower_tail(static_cast<long long>(std::min(m,n))-1,n,coverage);
    if (result.feasible) {
        if (side == ToleranceSide::upper) {
            result.upper_rank = m;
            result.upper = sorted[m-1];
        } else {
            result.lower_rank = n-m+1;
            result.lower = sorted[n-m];
        }
    }
    return result;
}

} // namespace datamunge::stats
