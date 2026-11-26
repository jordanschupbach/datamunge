/// @file equivalence.cpp
/// @brief TOST for means and Wald or Farrington–Manning noninferiority tests for risk differences.

#include <datamunge/stats/equivalence.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Sample summaries, the TOST core and restricted binomial estimates.
namespace {

/// @brief Mean and unbiased variance of a sample with at least two finite values.
/// @param x Sample.
/// @param mean Set to the mean.
/// @param variance Set to the variance with divisor n-1.
/// @throws std::invalid_argument For fewer than two or nonfinite values.
void summarize(const std::vector<double>& x, double& mean, double& variance) {
    if (x.size() < 2 || !std::ranges::all_of(x,[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("tost: each sample needs at least two finite values");
    const double n = static_cast<double>(x.size());
    mean = 0;
    for (const double v : x)
        mean += v/n;
    variance = 0;
    for (const double v : x)
        variance += (v-mean)*(v-mean)/(n-1);
}

/// @brief Completes a TOST result from the estimate, its standard error and df.
/// @param estimate Mean or mean difference.
/// @param se Standard error.
/// @param df Degrees of freedom.
/// @param options Margins and alpha.
/// @return The result.
/// @throws std::invalid_argument For invalid margins or alpha, or a zero standard error.
TostResult finish_tost(double estimate, double se, double df, const TostOptions& options) {
    if (!std::isfinite(options.lower_margin) || !std::isfinite(options.upper_margin) ||
        !(options.lower_margin < options.upper_margin))
        throw std::invalid_argument("tost: need finite margins with lower < upper");
    if (!(options.alpha > 0 && options.alpha < .5))
        throw std::invalid_argument("tost: alpha must be in (0, 0.5)");
    if (!(se > 0))
        throw std::invalid_argument("tost: standard error must be positive");
    TostResult r;
    r.estimate = estimate;
    r.std_error = se;
    r.df = df;
    r.t_lower = (estimate-options.lower_margin)/se;
    r.t_upper = (estimate-options.upper_margin)/se;
    r.p_lower = 1-random::student_t_cdf(r.t_lower,df);
    r.p_upper = random::student_t_cdf(r.t_upper,df);
    r.p_value = std::max(r.p_lower,r.p_upper);
    const double t = random::student_t_quantile(1-options.alpha,df);
    r.conf_low = estimate-t*se;
    r.conf_high = estimate+t*se;
    r.equivalent = r.p_value < options.alpha;
    return r;
}

/// @brief Standard normal distribution function.
/// @param x Argument.
/// @return \f$\Phi(x)\f$.
double phi(double x) {
    return .5*std::erfc(-x/std::numbers::sqrt2);
}

/// @brief Maximum likelihood estimate of \f$p_1\f$ subject to \f$p_1-p_2=\delta\f$ (Miettinen–Nurminen cubic).
/// @param p1 Observed proportion in group 1.
/// @param n1 Group 1 size.
/// @param p2 Observed proportion in group 2.
/// @param n2 Group 2 size.
/// @param delta Hypothesized difference in (-1, 1).
/// @return Restricted \f$\tilde p_1\f$; \f$\tilde p_2=\tilde p_1-\delta\f$.
double restricted_p1(double p1, double n1, double p2, double n2, double delta) {
    const double theta = n2/n1;
    const double a = 1+theta;
    const double b = -(1+theta+p1+theta*p2+delta*(theta+2));
    const double c = delta*delta+delta*(2*p1+theta+1)+p1+theta*p2;
    const double d = -p1*delta*(1+delta);
    const double v = b*b*b/(27*a*a*a)-b*c/(6*a*a)+d/(2*a);
    const double s = b*b/(9*a*a)-c/(3*a);
    double tilde;
    if (s <= 0) {
        tilde = -b/(3*a);
    } else {
        const double u = (v < 0 ? -1.0 : 1.0)*std::sqrt(s);
        const double w = (std::numbers::pi+std::acos(std::clamp(v/(u*u*u),-1.0,1.0)))/3;
        tilde = 2*u*std::cos(w)-b/(3*a);
    }
    return std::clamp(tilde,std::max(0.0,delta),std::min(1.0,1+delta));
}

/// @brief Farrington–Manning score statistic for \f$H_0:p_1-p_2=\delta\f$.
/// @param p1 Observed proportion 1.
/// @param n1 Size 1.
/// @param p2 Observed proportion 2.
/// @param n2 Size 2.
/// @param delta Hypothesized difference.
/// @return z statistic (infinite when the restricted variance is zero).
double score_z(double p1, double n1, double p2, double n2, double delta) {
    const double t1 = restricted_p1(p1,n1,p2,n2,delta), t2 = t1-delta;
    const double variance = t1*(1-t1)/n1+t2*(1-t2)/n2;
    const double diff = p1-p2-delta;
    if (!(variance > 0))
        return diff == 0 ? 0.0 : (diff > 0 ? INFINITY : -INFINITY);
    return diff/std::sqrt(variance);
}

} // namespace

TostResult tost_mean(const std::vector<double>& x, const TostOptions& options) {
    double mean, variance;
    summarize(x,mean,variance);
    const double n = static_cast<double>(x.size());
    return finish_tost(mean,std::sqrt(variance/n),n-1,options);
}

TostResult tost_mean_difference(const std::vector<double>& x, const std::vector<double>& y,
                                const TostOptions& options) {
    double mx, vx, my, vy;
    summarize(x,mx,vx);
    summarize(y,my,vy);
    const double nx = static_cast<double>(x.size()), ny = static_cast<double>(y.size());
    if (options.equal_variance) {
        const double pooled = ((nx-1)*vx+(ny-1)*vy)/(nx+ny-2);
        return finish_tost(mx-my,std::sqrt(pooled*(1/nx+1/ny)),nx+ny-2,options);
    }
    const double ax = vx/nx, ay = vy/ny;
    const double df = (ax+ay)*(ax+ay)/(ax*ax/(nx-1)+ay*ay/(ny-1));
    return finish_tost(mx-my,std::sqrt(ax+ay),df,options);
}

RiskDifferenceTest noninferiority_risk_difference(std::size_t successes1, std::size_t n1, std::size_t successes2,
                                                  std::size_t n2, double margin, double alpha,
                                                  RiskDifferenceMethod method) {
    if (n1 == 0 || n2 == 0 || successes1 > n1 || successes2 > n2)
        throw std::invalid_argument("noninferiority_risk_difference: need positive sizes and counts not above them");
    if (!(margin > 0 && margin < 1) || !(alpha > 0 && alpha < .5))
        throw std::invalid_argument("noninferiority_risk_difference: margin must be in (0, 1) and alpha in (0, 0.5)");
    const double m1 = static_cast<double>(n1), m2 = static_cast<double>(n2);
    const double p1 = static_cast<double>(successes1)/m1, p2 = static_cast<double>(successes2)/m2;
    const double z_crit = random::normal_quantile(1-alpha);
    RiskDifferenceTest r;
    r.estimate = p1-p2;
    if (method == RiskDifferenceMethod::wald) {
        const double variance = p1*(1-p1)/m1+p2*(1-p2)/m2;
        if (!(variance > 0))
            throw std::invalid_argument("noninferiority_risk_difference: Wald variance is zero");
        r.restricted_p1 = p1;
        r.restricted_p2 = p2;
        r.statistic = (r.estimate+margin)/std::sqrt(variance);
        r.conf_low = r.estimate-z_crit*std::sqrt(variance);
        r.conf_high = r.estimate+z_crit*std::sqrt(variance);
    } else {
        r.restricted_p1 = restricted_p1(p1,m1,p2,m2,-margin);
        r.restricted_p2 = r.restricted_p1+margin;
        r.statistic = score_z(p1,m1,p2,m2,-margin);
        // z(delta) decreases in delta: lower limit solves z = z_crit, upper solves z = -z_crit.
        const auto solve = [&](double target, double lo, double hi) {
            for (int it = 0; it < 200 && hi-lo > 1e-13; ++it) {
                const double mid = .5*(lo+hi);
                (score_z(p1,m1,p2,m2,mid) > target ? lo : hi) = mid;
            }
            return .5*(lo+hi);
        };
        const double eps = 1e-12;
        r.conf_low = score_z(p1,m1,p2,m2,-1+eps) <= z_crit ? -1.0 : solve(z_crit,-1+eps,r.estimate);
        r.conf_high = score_z(p1,m1,p2,m2,1-eps) >= -z_crit ? 1.0 : solve(-z_crit,r.estimate,1-eps);
    }
    r.p_value = std::isinf(r.statistic) ? (r.statistic > 0 ? 0.0 : 1.0) : 1-phi(r.statistic);
    r.noninferior = r.p_value < alpha;
    return r;
}

} // namespace datamunge::stats
