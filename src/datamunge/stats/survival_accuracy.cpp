/// @file survival_accuracy.cpp
/// @brief Censoring Kaplan–Meier weights, weighted pair counting and Brier scores.

#include <datamunge/stats/survival_accuracy.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace datamunge::stats {

/// @brief Internal helpers for survival prediction accuracy.
namespace {

/// @brief Kaplan–Meier estimate of the censoring survival \f$G\f$ as a right-continuous step function.
struct CensoringSurvival {
    std::vector<double> times;  ///< Distinct censoring times, increasing.
    std::vector<double> values; ///< \f$\hat G\f$ just after each time.

    /// @brief \f$\hat G(t)\f$.
    /// @param t Time.
    /// @return Censoring survival at t.
    [[nodiscard]] double at(double t) const {
        const auto k = std::ranges::upper_bound(times,t)-times.begin();
        return k == 0 ? 1.0 : values[static_cast<std::size_t>(k-1)];
    }

    /// @brief \f$\hat G(t-)\f$.
    /// @param t Time.
    /// @return Censoring survival just before t.
    [[nodiscard]] double before(double t) const {
        const auto k = std::ranges::lower_bound(times,t)-times.begin();
        return k == 0 ? 1.0 : values[static_cast<std::size_t>(k-1)];
    }
};

/// @brief Validates observations and builds the censoring Kaplan–Meier estimate.
/// @param obs Observations.
/// @param caller Name for error messages.
/// @return The estimate, with failures at a tied time leaving the censoring risk set first.
/// @throws std::invalid_argument For an empty sample or invalid times.
CensoringSurvival censoring_km(const std::vector<SurvivalObservation>& obs, const char* caller) {
    if (obs.empty())
        throw std::invalid_argument(std::string(caller)+": need observations");
    std::vector<std::size_t> order(obs.size());
    std::iota(order.begin(),order.end(),0);
    for (const auto& o : obs)
        if (!std::isfinite(o.time) || o.time < 0)
            throw std::invalid_argument(std::string(caller)+": times must be finite and nonnegative");
    std::ranges::sort(order,[&](std::size_t a, std::size_t b) { return obs[a].time < obs[b].time; });
    CensoringSurvival g;
    double s = 1, at_risk = static_cast<double>(obs.size());
    for (std::size_t k = 0; k < order.size();) {
        std::size_t end = k;
        double censored = 0, failed = 0;
        while (end < order.size() && obs[order[end]].time == obs[order[k]].time) {
            (obs[order[end]].event ? failed : censored) += 1;
            ++end;
        }
        at_risk -= failed;
        if (censored > 0) {
            s *= 1-censored/at_risk;
            g.times.push_back(obs[order[k]].time);
            g.values.push_back(s);
        }
        at_risk -= censored;
        k = end;
    }
    return g;
}

/// @brief Checks the evaluation grid and the matrix shape.
/// @param n Observations.
/// @param m Matrix of predictions.
/// @param times Grid.
/// @param g Censoring survival.
/// @param caller Name for error messages.
/// @throws std::invalid_argument For a mismatch, a non-increasing grid or a time with \f$\hat G(t)=0\f$.
void check_grid(std::size_t n, const linalg::DenseMatrix<double>& m, const std::vector<double>& times,
                const CensoringSurvival& g, const char* caller) {
    if (times.empty() || m.rows() != n || m.cols() != times.size())
        throw std::invalid_argument(std::string(caller)+": need one prediction column per time and one row per observation");
    for (std::size_t k = 0; k < times.size(); ++k) {
        if (!std::isfinite(times[k]) || times[k] < 0 || (k > 0 && !(times[k] > times[k-1])))
            throw std::invalid_argument(std::string(caller)+": times must be finite, nonnegative and increasing");
        if (!(g.at(times[k]) > 0))
            throw std::invalid_argument(std::string(caller)+": time beyond the support of the censoring distribution");
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < times.size(); ++k)
            if (!std::isfinite(m(i,k)))
                throw std::invalid_argument(std::string(caller)+": predictions must be finite");
}

/// @brief Left-step integral of values over the grid divided by its length.
/// @param times Grid.
/// @param values Values at the grid points.
/// @return The average, or the single value for a one-point grid.
double step_average(const std::vector<double>& times, const std::vector<double>& values) {
    if (times.size() == 1)
        return values[0];
    double area = 0;
    for (std::size_t k = 0; k+1 < times.size(); ++k)
        area += values[k]*(times[k+1]-times[k]);
    return area/(times.back()-times.front());
}

} // namespace

TimeDependentAuc time_dependent_auc(const std::vector<SurvivalObservation>& observations,
                                    const linalg::DenseMatrix<double>& risk, const std::vector<double>& times) {
    const auto g = censoring_km(observations,"time_dependent_auc");
    const std::size_t n = observations.size();
    check_grid(n,risk,times,g,"time_dependent_auc");
    TimeDependentAuc out{times,{},{},{}};
    for (std::size_t k = 0; k < times.size(); ++k) {
        const double t = times[k];
        // Controls sorted by risk; each case adds the count below plus half the ties.
        std::vector<double> controls;
        std::size_t cases = 0;
        double case_weight = 0, pairs = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (observations[i].time > t)
                controls.push_back(risk(i,k));
        std::ranges::sort(controls);
        for (std::size_t i = 0; i < n; ++i) {
            if (!(observations[i].event && observations[i].time <= t))
                continue;
            ++cases;
            const double w = 1/g.before(observations[i].time);
            const auto lower = std::ranges::lower_bound(controls,risk(i,k))-controls.begin();
            const auto upper = std::ranges::upper_bound(controls,risk(i,k))-controls.begin();
            case_weight += w;
            pairs += w*(static_cast<double>(lower)+0.5*static_cast<double>(upper-lower));
        }
        if (cases == 0 || controls.empty())
            throw std::invalid_argument("time_dependent_auc: every time needs cases and controls");
        // Control weights 1/G(t) are common to all controls and cancel.
        out.auc.push_back(pairs/(case_weight*static_cast<double>(controls.size())));
        out.cases.push_back(cases);
        out.controls.push_back(controls.size());
    }
    return out;
}

BrierScore brier_score(const std::vector<SurvivalObservation>& observations, const linalg::DenseMatrix<double>& survival,
                       const std::vector<double>& times) {
    const auto g = censoring_km(observations,"brier_score");
    const std::size_t n = observations.size();
    check_grid(n,survival,times,g,"brier_score");
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < times.size(); ++k)
            if (survival(i,k) < 0 || survival(i,k) > 1)
                throw std::invalid_argument("brier_score: predicted survival must lie in [0, 1]");
    // Pooled Kaplan–Meier prediction for the benchmark.
    std::vector<double> sorted_times;
    std::vector<double> km_values;
    {
        std::vector<std::size_t> order(n);
        std::iota(order.begin(),order.end(),0);
        std::ranges::sort(order,[&](std::size_t a, std::size_t b) { return observations[a].time < observations[b].time; });
        double s = 1, at_risk = static_cast<double>(n);
        for (std::size_t k = 0; k < n;) {
            std::size_t end = k;
            double events = 0;
            while (end < n && observations[order[end]].time == observations[order[k]].time)
                events += observations[order[end++]].event;
            if (events > 0) {
                s *= 1-events/at_risk;
                sorted_times.push_back(observations[order[k]].time);
                km_values.push_back(s);
            }
            at_risk -= static_cast<double>(end-k);
            k = end;
        }
    }
    const auto km = [&](double t) {
        const auto k = std::ranges::upper_bound(sorted_times,t)-sorted_times.begin();
        return k == 0 ? 1.0 : km_values[static_cast<std::size_t>(k-1)];
    };
    BrierScore out{times,{},{},0,0,0};
    for (std::size_t k = 0; k < times.size(); ++k) {
        const double t = times[k], gt = g.at(t), s0 = km(t);
        double model = 0, null = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto& o = observations[i];
            const double s = survival(i,k);
            if (o.time > t) {
                model += (1-s)*(1-s)/gt;
                null += (1-s0)*(1-s0)/gt;
            } else if (o.event) {
                const double w = 1/g.before(o.time);
                model += s*s*w;
                null += s0*s0*w;
            }
        }
        out.brier.push_back(model/static_cast<double>(n));
        out.null_brier.push_back(null/static_cast<double>(n));
    }
    out.integrated = step_average(times,out.brier);
    out.null_integrated = step_average(times,out.null_brier);
    out.explained = out.null_integrated > 0 ? 1-out.integrated/out.null_integrated : 0.0;
    return out;
}

} // namespace datamunge::stats
