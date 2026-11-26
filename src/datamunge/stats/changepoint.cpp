/// @file changepoint.cpp
/// @brief Cumulative-sum segment costs, penalty rules and the pruned optimal-partitioning recursion.

#include <datamunge/stats/changepoint.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Segment cost evaluation from cumulative sums.
namespace {

/// @brief Cumulative sums \f$\sum x\f$, \f$\sum x^2\f$ and \f$\sum(x-\mu)^2\f$ with a leading zero.
struct Sums {
    std::vector<double> s1; ///< Sums of x.
    std::vector<double> s2; ///< Sums of x squared.
    std::vector<double> s3; ///< Sums of squared deviations from the fixed mean.
};

/// @brief Cost of the segment \f$(a,b]\f$ (R's mll_* and mbic_* functions).
/// @param sums Cumulative sums.
/// @param a Start (exclusive).
/// @param b End (inclusive).
/// @param options Cost type and known variance.
/// @param mbic Whether to add \f$\ln m\f$.
/// @return Segment cost.
double cost(const Sums& sums, std::size_t a, std::size_t b, const PeltOptions& options, bool mbic) {
    const double x = sums.s1[b]-sums.s1[a], x2 = sums.s2[b]-sums.s2[a], m = static_cast<double>(b-a);
    double c = 0;
    switch (options.cost) {
    case ChangeCost::normal_mean:
        c = (x2-x*x/m)/options.noise_variance;
        break;
    case ChangeCost::normal_variance: {
        double x3 = sums.s3[b]-sums.s3[a];
        if (x3 <= 0)
            x3 = 1e-11;
        c = m*(std::log(2*std::numbers::pi)+std::log(x3/m)+1);
        break;
    }
    case ChangeCost::normal_mean_variance: {
        double sigsq = (x2-x*x/m)/m;
        if (sigsq <= 0)
            sigsq = 1e-11;
        c = m*(std::log(2*std::numbers::pi)+std::log(sigsq)+1);
        break;
    }
    case ChangeCost::poisson:
        // R returns 0 for an all-zero segment, without the MBIC length term.
        if (x == 0)
            return 0;
        c = 2*x*(std::log(m)-std::log(x));
        break;
    case ChangeCost::exponential:
        c = 2*m*(std::log(x)-std::log(m));
        break;
    }
    return mbic ? c+std::log(m) : c;
}

} // namespace

ChangepointResult pelt_changepoints(const std::vector<double>& x, const PeltOptions& options) {
    const std::size_t n = x.size();
    if (n == 0)
        throw std::invalid_argument("pelt_changepoints: series is empty");
    for (const double v : x) {
        if (!std::isfinite(v))
            throw std::invalid_argument("pelt_changepoints: values must be finite");
        if (options.cost == ChangeCost::poisson && (v < 0 || v != std::floor(v)))
            throw std::invalid_argument("pelt_changepoints: Poisson cost needs nonnegative integer counts");
        if (options.cost == ChangeCost::exponential && !(v > 0))
            throw std::invalid_argument("pelt_changepoints: exponential cost needs positive values");
    }
    if (!(options.noise_variance > 0) || !std::isfinite(options.noise_variance))
        throw std::invalid_argument("pelt_changepoints: noise variance must be positive");
    const bool variance_cost = options.cost == ChangeCost::normal_variance || options.cost == ChangeCost::normal_mean_variance;
    std::size_t minseg = options.min_segment_length;
    if (minseg == 0)
        minseg = options.cost == ChangeCost::normal_mean ? 1 : 2;
    if (variance_cost && minseg < 2)
        throw std::invalid_argument("pelt_changepoints: variance costs need segments of at least 2");

    const double d = options.cost == ChangeCost::normal_mean_variance ? 2.0 : 1.0, logn = std::log(static_cast<double>(n));
    double beta = 0;
    switch (options.penalty) {
    case ChangePenalty::mbic: beta = (d+2)*logn; break;
    case ChangePenalty::bic: beta = (d+1)*logn; break;
    case ChangePenalty::aic: beta = 2*(d+1); break;
    case ChangePenalty::hannan_quinn: beta = 2*(d+1)*std::log(logn); break;
    case ChangePenalty::none: beta = 0; break;
    case ChangePenalty::manual:
        if (!(options.penalty_value >= 0) || !std::isfinite(options.penalty_value))
            throw std::invalid_argument("pelt_changepoints: manual penalty must be finite and nonnegative");
        beta = options.penalty_value;
        break;
    }
    const bool mbic = options.penalty == ChangePenalty::mbic;

    double mean = 0;
    for (const double v : x)
        mean += v;
    mean /= static_cast<double>(n);
    const double mu = options.known_mean.value_or(mean);
    Sums sums;
    sums.s1.assign(n+1,0.0);
    sums.s2.assign(n+1,0.0);
    sums.s3.assign(n+1,0.0);
    for (std::size_t i = 0; i < n; ++i) {
        sums.s1[i+1] = sums.s1[i]+x[i];
        sums.s2[i+1] = sums.s2[i]+x[i]*x[i];
        sums.s3[i+1] = sums.s3[i]+(x[i]-mu)*(x[i]-mu);
    }

    ChangepointResult r;
    r.penalty = beta;
    std::vector<std::size_t> last(n+1,0);
    if (n >= 2*minseg) {
        std::vector<double> f(n+1,0.0);
        f[0] = -beta;
        for (std::size_t j = minseg; j < 2*minseg; ++j)
            f[j] = cost(sums,0,j,options,mbic);
        std::vector<std::size_t> candidates{0,minseg};
        std::vector<double> values;
        for (std::size_t t = 2*minseg; t <= n; ++t) {
            values.resize(candidates.size());
            std::size_t best = 0;
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                values[i] = f[candidates[i]]+cost(sums,candidates[i],t,options,mbic)+beta;
                if (values[i] < values[best])
                    best = i;
            }
            f[t] = values[best];
            last[t] = candidates[best];
            std::size_t kept = 0;
            for (std::size_t i = 0; i < candidates.size(); ++i)
                if (values[i] <= f[t]+beta)
                    candidates[kept++] = candidates[i];
            candidates.resize(kept);
            candidates.push_back(t-(minseg-1));
        }
        r.cost = f[n];
        for (std::size_t t = last[n]; t != 0; t = last[t])
            r.changepoints.insert(r.changepoints.begin(),t);
    } else {
        r.cost = cost(sums,0,n,options,mbic);
    }
    std::size_t start = 0;
    for (std::size_t j = 0; j <= r.changepoints.size(); ++j) {
        const std::size_t end = j < r.changepoints.size() ? r.changepoints[j] : n;
        const double m = static_cast<double>(end-start);
        const double s = sums.s1[end]-sums.s1[start];
        double ss = 0;
        for (std::size_t i = start; i < end; ++i)
            ss += (x[i]-s/m)*(x[i]-s/m);
        r.means.push_back(s/m);
        r.variances.push_back(ss/m);
        start = end;
    }
    return r;
}

} // namespace datamunge::stats
