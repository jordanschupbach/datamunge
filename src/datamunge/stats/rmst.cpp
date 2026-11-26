/// @file rmst.cpp
/// @brief Kaplan–Meier area to a horizon with its Greenwood-type variance.

#include <datamunge/stats/rmst.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal helpers for RMST estimation.
namespace {

/// @brief Restricted area under a Kaplan–Meier curve and its variance.
struct Area {
    double rmst{};     ///< Area from 0 to tau.
    double variance{}; ///< Greenwood-type variance of the area.
};

/// @brief Integrates the Kaplan–Meier curve to tau and accumulates its variance.
/// @param observations Right-censored records.
/// @param tau Horizon in (0, largest time].
/// @return Area and variance.
/// @throws std::invalid_argument For empty input, invalid times or tau.
Area restricted_area(const std::vector<SurvivalObservation>& observations, double tau) {
    if (observations.empty())
        throw std::invalid_argument("rmst: no observations");
    double last = 0;
    for (const auto& o : observations) {
        if (!std::isfinite(o.time) || o.time < 0)
            throw std::invalid_argument("rmst: times must be finite and nonnegative");
        last = std::max(last,o.time);
    }
    if (!(tau > 0) || !std::isfinite(tau) || tau > last)
        throw std::invalid_argument("rmst: tau must be positive and at most the largest observed time");
    // Event times up to tau with survival after each and its variance weight.
    std::vector<double> event_time, survival_after, weight;
    double s = 1;
    const std::size_t n = observations.size();
    std::vector<std::pair<double,bool>> sorted;
    for (const auto& o : observations)
        sorted.emplace_back(o.time,o.event);
    std::ranges::sort(sorted);
    for (std::size_t i = 0; i < n && sorted[i].first <= tau;) {
        const double t = sorted[i].first;
        const double at_risk = static_cast<double>(n-i);
        double deaths = 0;
        for (; i < n && sorted[i].first == t; ++i)
            deaths += sorted[i].second ? 1 : 0;
        if (deaths == 0)
            continue;
        s *= (at_risk-deaths)/at_risk;
        event_time.push_back(t);
        survival_after.push_back(s);
        weight.push_back(at_risk > deaths ? deaths/(at_risk*(at_risk-deaths)) : 0.0);
    }
    Area a;
    double previous_time = 0, previous_survival = 1;
    for (std::size_t k = 0; k < event_time.size(); ++k) {
        a.rmst += previous_survival*(event_time[k]-previous_time);
        previous_time = event_time[k];
        previous_survival = survival_after[k];
    }
    a.rmst += previous_survival*(tau-previous_time);
    // Area after each event time, accumulated backward from tau.
    double after = 0;
    for (std::size_t k = event_time.size(); k-- > 0;) {
        const double next = k+1 < event_time.size() ? event_time[k+1] : tau;
        after += survival_after[k]*(next-event_time[k]);
        a.variance += after*after*weight[k];
    }
    return a;
}

/// @brief Rejects a confidence level outside (0, 1).
/// @param confidence Proposed coverage.
/// @throws std::invalid_argument If out of range.
void check_confidence(double confidence) {
    if (!(confidence > 0 && confidence < 1))
        throw std::invalid_argument("rmst: confidence must be in (0, 1)");
}

/// @brief Wald interval and two-sided normal p-value.
/// @param estimate Point estimate.
/// @param se Standard error.
/// @param z Normal quantile.
/// @param null_value Value under the null.
/// @return Contrast on the same scale as the estimate.
RmstContrast wald(double estimate, double se, double z, double null_value = 0) {
    RmstContrast c{estimate,se,estimate-z*se,estimate+z*se,1};
    if (se > 0)
        c.p_value = 2*random::normal_cdf(-std::abs(estimate-null_value)/se);
    return c;
}

/// @brief Ratio of two restricted means with a log-scale delta-method interval.
/// @param num Numerator mean.
/// @param num_var Its variance.
/// @param den Denominator mean.
/// @param den_var Its variance.
/// @param z Normal quantile.
/// @return Ratio, log-scale standard error, interval and p-value; NaN if a mean is not positive.
RmstContrast log_ratio(double num, double num_var, double den, double den_var, double z) {
    if (!(num > 0) || !(den > 0))
        return {std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    const double log_ratio = std::log(num/den);
    const double se = std::sqrt(num_var/(num*num)+den_var/(den*den));
    const auto c = wald(log_ratio,se,z);
    return {std::exp(log_ratio),se,std::exp(c.lower),std::exp(c.upper),c.p_value};
}

} // namespace

RmstEstimate rmst(const std::vector<SurvivalObservation>& observations, double tau, double confidence) {
    check_confidence(confidence);
    const Area a = restricted_area(observations,tau);
    const double z = random::normal_quantile(.5+confidence/2), se = std::sqrt(a.variance);
    return {tau,a.rmst,se,a.rmst-z*se,a.rmst+z*se,tau-a.rmst};
}

RmstComparison rmst_compare(const std::vector<SurvivalObservation>& observations, const std::vector<int>& groups,
                            double tau, double confidence) {
    check_confidence(confidence);
    if (groups.size() != observations.size())
        throw std::invalid_argument("rmst_compare: need one group per observation");
    std::vector<int> labels = groups;
    std::ranges::sort(labels);
    labels.erase(std::unique(labels.begin(),labels.end()),labels.end());
    if (labels.size() != 2)
        throw std::invalid_argument("rmst_compare: need exactly two groups");
    RmstComparison c;
    std::array<double,2> variance{};
    for (std::size_t g = 0; g < 2; ++g) {
        c.groups[g] = labels[g];
        std::vector<SurvivalObservation> arm;
        for (std::size_t i = 0; i < groups.size(); ++i)
            if (groups[i] == labels[g])
                arm.push_back(observations[i]);
        c.arms[g] = rmst(arm,tau,confidence);
        variance[g] = c.arms[g].standard_error*c.arms[g].standard_error;
    }
    const double z = random::normal_quantile(.5+confidence/2);
    c.difference = wald(c.arms[1].rmst-c.arms[0].rmst,std::sqrt(variance[0]+variance[1]),z);
    c.ratio = log_ratio(c.arms[1].rmst,variance[1],c.arms[0].rmst,variance[0],z);
    c.time_lost_ratio = log_ratio(c.arms[1].rmtl,variance[1],c.arms[0].rmtl,variance[0],z);
    return c;
}

} // namespace datamunge::stats
