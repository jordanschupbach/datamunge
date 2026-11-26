/// @file cox_counting.cpp
/// @brief Sweep-based risk-set sums for the counting-process partial likelihood and Newton fitting.

#include <datamunge/stats/cox_counting.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <cmath>
#include <numeric>
#include <string>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal helpers for counting-process Cox fitting.
namespace {

/// @brief Dense double matrix.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Sorted views of one stratum's records.
struct StratumLayout {
    int label{};                                  ///< Stratum label.
    std::vector<std::size_t> by_stop;             ///< Record indices by decreasing stop.
    std::vector<std::size_t> by_start;            ///< Record indices by decreasing start.
    std::vector<double> times;                    ///< Distinct event times, decreasing.
    std::vector<std::vector<std::size_t>> deaths; ///< Records with an event at each time.
};

/// @brief Data shared by every likelihood evaluation.
struct Layout {
    std::size_t n{};                              ///< Number of records.
    std::size_t p{};                              ///< Number of covariates.
    Matrix x;                                     ///< Covariates centered at their column means.
    std::vector<double> start;                    ///< Interval starts.
    std::vector<double> stop;                     ///< Interval stops.
    std::vector<StratumLayout> strata;            ///< Per-stratum sweeps, by increasing label.
};

/// @brief Risk-set moments at one event time, kept for residuals and diagnostics.
struct EventMoments {
    std::size_t stratum{};            ///< Index into Layout::strata.
    double time{};                    ///< Event time.
    std::vector<std::size_t> deaths;  ///< Records failing at the time.
    std::vector<double> mean;         ///< Average over tie fractions of the weighted mean covariates.
    Matrix variance;                  ///< Sum over tie fractions of the weighted covariate covariance.
    double hazard{};                  ///< \f$\sum_k 1/S_{0k}\f$ over tie fractions (shifted weight scale).
    std::vector<double> hazard_mean;  ///< \f$\sum_k \bar x_k/S_{0k}\f$.
    double dead_hazard{};             ///< \f$\sum_k (1-f_k)/S_{0k}\f$, the share seen by a failing record.
    std::vector<double> dead_hazard_mean; ///< \f$\sum_k (1-f_k)\bar x_k/S_{0k}\f$.
};

/// @brief Log partial likelihood, score, information and baseline increments.
struct Evaluation {
    double log_likelihood{};                     ///< Log partial likelihood.
    std::vector<double> score;                   ///< Gradient.
    Matrix information;                          ///< Negative Hessian.
    std::vector<std::vector<double>> increments; ///< Per stratum, hazard increments in decreasing time.
    std::vector<EventMoments> moments;           ///< Per event time (decreasing within stratum), if requested.
    double shift{};                              ///< Linear-predictor shift used in the risk weights.
};

/// @brief Running weighted sums over a risk set.
struct Sums {
    double s0{};             ///< Sum of weights.
    std::vector<double> s1;  ///< Weighted covariate sums.
    Matrix s2;               ///< Weighted outer-product sums.

    /// @brief Zeroed sums for p covariates.
    /// @param p Covariate count.
    explicit Sums(std::size_t p) : s1(p,0.0), s2(p,p) {}

    /// @brief Adds or subtracts one record.
    /// @param x Centered covariates.
    /// @param i Record.
    /// @param w Weight, negative to remove.
    void add(const Matrix& x, std::size_t i, double w) {
        const std::size_t p = s1.size();
        s0 += w;
        for (std::size_t a = 0; a < p; ++a) {
            s1[a] += w*x(i,a);
            for (std::size_t b = 0; b <= a; ++b)
                s2(a,b) += w*x(i,a)*x(i,b);
        }
    }
};

/// @brief Validates the data and builds centered covariates and per-stratum sorted sweeps.
/// @param records Intervals.
/// @param covariates Covariate rows.
/// @param strata Stratum labels (empty: one stratum).
/// @param reference Receives the covariate column means.
/// @param events Receives the number of events.
/// @param caller Name used in error messages.
/// @return The layout.
/// @throws std::invalid_argument For invalid intervals or covariates.
Layout make_layout(const std::vector<CountingRecord>& records, const Matrix& covariates, const std::vector<int>& strata,
                   std::vector<double>& reference, std::size_t& events, const char* caller) {
    const std::size_t n = records.size(), p = covariates.cols();
    Layout data;
    data.n = n;
    data.p = p;
    data.x = covariates;
    reference.assign(p,0.0);
    events = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& r = records[i];
        if (!std::isfinite(r.start) || !std::isfinite(r.stop) || !(r.start < r.stop))
            throw std::invalid_argument(std::string(caller)+": intervals need finite start < stop");
        data.start.push_back(r.start);
        data.stop.push_back(r.stop);
        for (std::size_t a = 0; a < p; ++a) {
            if (!std::isfinite(covariates(i,a)))
                throw std::invalid_argument(std::string(caller)+": covariates must be finite");
            reference[a] += covariates(i,a)/static_cast<double>(n);
        }
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t a = 0; a < p; ++a)
            data.x(i,a) -= reference[a];
    std::vector<int> labels = strata;
    if (labels.empty())
        labels.assign(n,0);
    std::vector<int> distinct = labels;
    std::ranges::sort(distinct);
    distinct.erase(std::unique(distinct.begin(),distinct.end()),distinct.end());
    for (const int label : distinct)
        data.strata.push_back({.label=label});
    for (std::size_t i = 0; i < n; ++i) {
        auto& s = data.strata[static_cast<std::size_t>(std::ranges::lower_bound(distinct,labels[i])-distinct.begin())];
        s.by_stop.push_back(i);
        s.by_start.push_back(i);
    }
    for (auto& s : data.strata) {
        std::ranges::sort(s.by_stop,[&](std::size_t a, std::size_t b) { return data.stop[a] > data.stop[b]; });
        std::ranges::sort(s.by_start,[&](std::size_t a, std::size_t b) { return data.start[a] > data.start[b]; });
        for (const std::size_t i : s.by_stop) {
            if (!records[i].event)
                continue;
            ++events;
            if (s.times.empty() || s.times.back() != data.stop[i]) {
                s.times.push_back(data.stop[i]);
                s.deaths.emplace_back();
            }
            s.deaths.back().push_back(i);
        }
    }
    return data;
}

/// @brief Evaluates the partial likelihood at beta by sweeping event times downward.
/// @param data Prepared layout.
/// @param beta Coefficients.
/// @param ties Efron or Breslow.
/// @param keep_moments Also record per-event-time moments.
/// @return Likelihood, score, information, baseline increments and optional moments.
Evaluation evaluate(const Layout& data, const std::vector<double>& beta, CoxTies ties, bool keep_moments = false) {
    const std::size_t n = data.n, p = data.p;
    std::vector<double> eta(n,0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t a = 0; a < p; ++a)
            eta[i] += data.x(i,a)*beta[a];
    const double shift = *std::ranges::max_element(eta);
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i)
        w[i] = std::exp(eta[i]-shift);
    Evaluation e{0,std::vector<double>(p,0.0),Matrix(p,p),{}};
    std::vector<char> active(n,0);
    for (std::size_t si = 0; si < data.strata.size(); ++si) {
        const auto& stratum = data.strata[si];
        const auto& by_stop = stratum.by_stop;
        const auto& by_start = stratum.by_start;
        const std::size_t m = by_stop.size();
        Sums risk(p);
        double peak = 0;
        std::size_t next_stop = 0, next_start = 0;
        e.increments.emplace_back();
        for (std::size_t k = 0; k < stratum.times.size(); ++k) {
            const double t = stratum.times[k];
            for (; next_stop < m && data.stop[by_stop[next_stop]] >= t; ++next_stop) {
                const std::size_t i = by_stop[next_stop];
                active[i] = 1;
                risk.add(data.x,i,w[i]);
            }
            bool removed = false;
            for (; next_start < m && data.start[by_start[next_start]] >= t; ++next_start) {
                const std::size_t i = by_start[next_start];
                if (active[i]) {
                    active[i] = 0;
                    risk.add(data.x,i,-w[i]);
                    removed = true;
                }
            }
            peak = std::max(peak,risk.s0);
            if (removed && risk.s0 < 1e-6*peak) {
                risk = Sums(p);
                for (const std::size_t i : by_stop)
                    if (active[i])
                        risk.add(data.x,i,w[i]);
                peak = risk.s0;
            }
            Sums dead(p);
            for (const std::size_t i : stratum.deaths[k]) {
                dead.add(data.x,i,w[i]);
                e.log_likelihood += eta[i]-shift;
                for (std::size_t a = 0; a < p; ++a)
                    e.score[a] += data.x(i,a);
            }
            const double d = static_cast<double>(stratum.deaths[k].size());
            double increment = 0;
            std::vector<double> s1(p);
            EventMoments moment{si,t,stratum.deaths[k],std::vector<double>(p,0.0),Matrix(p,p),0,
                                std::vector<double>(p,0.0),0,std::vector<double>(p,0.0)};
            for (std::size_t j = 0; j < stratum.deaths[k].size(); ++j) {
                const double f = ties == CoxTies::efron ? static_cast<double>(j)/d : 0.0;
                const double s0 = risk.s0-f*dead.s0;
                e.log_likelihood -= std::log(s0);
                increment += 1/s0;
                for (std::size_t a = 0; a < p; ++a) {
                    s1[a] = (risk.s1[a]-f*dead.s1[a])/s0;
                    e.score[a] -= s1[a];
                }
                for (std::size_t a = 0; a < p; ++a)
                    for (std::size_t b = 0; b <= a; ++b) {
                        const double v = (risk.s2(a,b)-f*dead.s2(a,b))/s0-s1[a]*s1[b];
                        e.information(a,b) += v;
                        if (keep_moments) {
                            moment.variance(a,b) += v;
                            moment.variance(b,a) = moment.variance(a,b);
                        }
                    }
                if (keep_moments) {
                    moment.hazard += 1/s0;
                    moment.dead_hazard += (1-f)/s0;
                    for (std::size_t a = 0; a < p; ++a) {
                        moment.mean[a] += s1[a]/d;
                        moment.hazard_mean[a] += s1[a]/s0;
                        moment.dead_hazard_mean[a] += (1-f)*s1[a]/s0;
                    }
                }
            }
            if (keep_moments)
                e.moments.push_back(std::move(moment));
            e.increments.back().push_back(increment*std::exp(-shift));
        }
    }
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b < a; ++b)
            e.information(b,a) = e.information(a,b);
    e.shift = shift;
    return e;
}

/// @brief Score residuals of every record (Efron fractions as survival's `coxscore`).
/// @param data Prepared layout.
/// @param beta Coefficients.
/// @param e Evaluation at beta with moments kept.
/// @return One row of residuals per record; the rows sum to the score.
Matrix score_residuals(const Layout& data, const std::vector<double>& beta, const Evaluation& e) {
    const std::size_t n = data.n, p = data.p;
    Matrix r(n,p);
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) {
        double eta = 0;
        for (std::size_t a = 0; a < p; ++a)
            eta += data.x(i,a)*beta[a];
        w[i] = std::exp(eta-e.shift);
    }
    // Per stratum, cumulative hazard sums in increasing time for interval differences.
    std::vector<std::vector<double>> times(data.strata.size()), cum_h(data.strata.size());
    std::vector<std::vector<std::vector<double>>> cum_hx(data.strata.size());
    for (const auto& m : e.moments) {
        times[m.stratum].push_back(m.time);
        cum_h[m.stratum].push_back(m.hazard);
        cum_hx[m.stratum].push_back(m.hazard_mean);
    }
    for (std::size_t s = 0; s < data.strata.size(); ++s) {
        std::ranges::reverse(times[s]);
        std::ranges::reverse(cum_h[s]);
        std::ranges::reverse(cum_hx[s]);
        for (std::size_t k = 1; k < times[s].size(); ++k) {
            cum_h[s][k] += cum_h[s][k-1];
            for (std::size_t a = 0; a < p; ++a)
                cum_hx[s][k][a] += cum_hx[s][k-1][a];
        }
    }
    for (std::size_t s = 0; s < data.strata.size(); ++s)
        for (const std::size_t i : data.strata[s].by_stop) {
            // Event times in (start, stop]: indices [lo, hi).
            const auto lo = static_cast<std::size_t>(std::ranges::upper_bound(times[s],data.start[i])-times[s].begin());
            const auto hi = static_cast<std::size_t>(std::ranges::upper_bound(times[s],data.stop[i])-times[s].begin());
            if (hi <= lo)
                continue;
            const double h = cum_h[s][hi-1]-(lo ? cum_h[s][lo-1] : 0.0);
            for (std::size_t a = 0; a < p; ++a) {
                const double hx = cum_hx[s][hi-1][a]-(lo ? cum_hx[s][lo-1][a] : 0.0);
                r(i,a) -= w[i]*(data.x(i,a)*h-hx);
            }
        }
    for (const auto& m : e.moments)
        for (const std::size_t i : m.deaths)
            for (std::size_t a = 0; a < p; ++a)
                r(i,a) += data.x(i,a)-m.mean[a]+w[i]*((data.x(i,a)*m.hazard-m.hazard_mean[a])-
                                                      (data.x(i,a)*m.dead_hazard-m.dead_hazard_mean[a]));
    return r;
}

/// @brief Inverts a symmetric positive definite matrix.
/// @param a Matrix to invert.
/// @return Its inverse.
/// @throws std::runtime_error If a is not positive definite.
Matrix spd_inverse(const Matrix& a) {
    const auto factor = linalg::cholesky(a);
    if (!factor.ok)
        throw std::runtime_error("fit_cox_counting: information matrix is singular");
    const std::size_t p = a.rows();
    Matrix inverse(p,p);
    for (std::size_t j = 0; j < p; ++j) {
        std::vector<double> unit(p,0.0);
        unit[j] = 1;
        const auto column = factor.solve(unit);
        for (std::size_t i = 0; i < p; ++i)
            inverse(i,j) = column[i];
    }
    return inverse;
}

/// @brief Quadratic form v' A v.
/// @param a Matrix.
/// @param v Vector.
/// @return The form.
double quadratic(const Matrix& a, const std::vector<double>& v) {
    double s = 0;
    for (std::size_t i = 0; i < v.size(); ++i)
        for (std::size_t j = 0; j < v.size(); ++j)
            s += v[i]*a(i,j)*v[j];
    return s;
}

/// @brief Chi-square test from a statistic.
/// @param statistic Statistic.
/// @param df Degrees of freedom.
/// @return Test with p-value.
ChiSquareTest chi_square(double statistic, std::size_t df) {
    return {statistic,df,detail::chi_squared_test_survival(std::max(statistic,0.0),static_cast<double>(df))};
}

} // namespace

double CoxCountingFit::predict_survival(double time, const std::vector<double>& covariates, int stratum) const {
    if (!(time >= 0) || covariates.size() != coefficients.size())
        throw std::invalid_argument("CoxCountingFit::predict_survival: invalid time or covariates");
    const auto baseline = std::ranges::find(baselines,stratum,&CoxBaseline::stratum);
    if (baseline == baselines.end())
        throw std::invalid_argument("CoxCountingFit::predict_survival: unknown stratum");
    double h = 0;
    for (std::size_t k = 0; k < baseline->event_times.size() && baseline->event_times[k] <= time; ++k)
        h = baseline->cumulative_hazard[k];
    double eta = 0;
    for (std::size_t a = 0; a < coefficients.size(); ++a)
        eta += (covariates[a]-reference[a])*coefficients[a];
    return std::exp(-h*std::exp(eta));
}

CoxCountingFit fit_cox_counting(const std::vector<CountingRecord>& records, const linalg::DenseMatrix<double>& covariates,
                                const CoxCountingOptions& options) {
    const std::size_t n = records.size(), p = covariates.cols();
    if (n == 0 || covariates.rows() != n || p == 0)
        throw std::invalid_argument("fit_cox_counting: need one nonempty covariate row per record");
    if (!options.strata.empty() && options.strata.size() != n)
        throw std::invalid_argument("fit_cox_counting: need one stratum label per record");
    if (!options.clusters.empty() && options.clusters.size() != n)
        throw std::invalid_argument("fit_cox_counting: need one cluster label per record");
    if (options.ties != CoxTies::efron && options.ties != CoxTies::breslow)
        throw std::invalid_argument("fit_cox_counting: unknown tie method");
    CoxCountingFit fit;
    Layout data = make_layout(records,covariates,options.strata,fit.reference,fit.events,"fit_cox_counting");
    if (fit.events == 0)
        throw std::invalid_argument("fit_cox_counting: no events");

    std::vector<double> beta(p,0.0);
    auto current = evaluate(data,beta,options.ties);
    fit.null_log_likelihood = current.log_likelihood;
    fit.score = chi_square(quadratic(spd_inverse(current.information),current.score),p);
    while (fit.iterations < options.max_iterations) {
        const auto factor = linalg::cholesky(current.information);
        if (!factor.ok)
            throw std::runtime_error("fit_cox_counting: information matrix is singular");
        const auto step = factor.solve(current.score);
        double scale = 1;
        Evaluation trial;
        std::vector<double> candidate(p);
        for (int halving = 0;; ++halving) {
            for (std::size_t a = 0; a < p; ++a)
                candidate[a] = beta[a]+scale*step[a];
            trial = evaluate(data,candidate,options.ties);
            if (std::isfinite(trial.log_likelihood) &&
                trial.log_likelihood >= current.log_likelihood-1e-12*std::abs(current.log_likelihood))
                break;
            if (halving == 30)
                throw std::runtime_error("fit_cox_counting: step halving failed");
            scale /= 2;
        }
        const double change = std::abs(1-current.log_likelihood/trial.log_likelihood);
        beta = candidate;
        current = std::move(trial);
        ++fit.iterations;
        if (change <= options.tolerance) {
            fit.converged = true;
            break;
        }
    }
    fit.coefficients = beta;
    fit.log_likelihood = current.log_likelihood;
    fit.covariance = spd_inverse(current.information);
    for (std::size_t a = 0; a < p; ++a)
        fit.standard_errors.push_back(std::sqrt(fit.covariance(a,a)));
    fit.likelihood_ratio = chi_square(2*(fit.log_likelihood-fit.null_log_likelihood),p);
    fit.wald = chi_square(quadratic(current.information,beta),p);
    if (!options.clusters.empty()) {
        const auto with_moments = evaluate(data,beta,options.ties,true);
        const Matrix residuals = score_residuals(data,beta,with_moments);
        std::vector<int> labels = options.clusters;
        std::ranges::sort(labels);
        labels.erase(std::unique(labels.begin(),labels.end()),labels.end());
        fit.clusters = labels.size();
        Matrix dfbeta(labels.size(),p);
        for (std::size_t i = 0; i < n; ++i) {
            const auto c = static_cast<std::size_t>(std::ranges::lower_bound(labels,options.clusters[i])-labels.begin());
            for (std::size_t a = 0; a < p; ++a)
                for (std::size_t b = 0; b < p; ++b)
                    dfbeta(c,a) += residuals(i,b)*fit.covariance(b,a);
        }
        fit.robust_covariance = Matrix(p,p);
        for (std::size_t c = 0; c < labels.size(); ++c)
            for (std::size_t a = 0; a < p; ++a)
                for (std::size_t b = 0; b < p; ++b)
                    fit.robust_covariance(a,b) += dfbeta(c,a)*dfbeta(c,b);
        for (std::size_t a = 0; a < p; ++a)
            fit.robust_standard_errors.push_back(std::sqrt(fit.robust_covariance(a,a)));
        fit.robust_wald = chi_square(quadratic(spd_inverse(fit.robust_covariance),beta),p);
    }
    for (std::size_t s = 0; s < data.strata.size(); ++s) {
        CoxBaseline baseline{.stratum=data.strata[s].label};
        baseline.event_times.assign(data.strata[s].times.rbegin(),data.strata[s].times.rend());
        double h = 0;
        for (auto it = current.increments[s].rbegin(); it != current.increments[s].rend(); ++it)
            baseline.cumulative_hazard.push_back(h += *it);
        fit.baselines.push_back(std::move(baseline));
    }
    return fit;
}

ProportionalHazardsTest test_proportional_hazards(const std::vector<CountingRecord>& records,
                                                  const linalg::DenseMatrix<double>& covariates, const CoxCountingFit& fit,
                                                  const CoxCountingOptions& options, TimeTransform transform) {
    const std::size_t n = records.size(), p = covariates.cols();
    if (n == 0 || covariates.rows() != n || p == 0 || fit.coefficients.size() != p)
        throw std::invalid_argument("test_proportional_hazards: data do not match the fit");
    if (!options.strata.empty() && options.strata.size() != n)
        throw std::invalid_argument("test_proportional_hazards: need one stratum label per record");
    std::vector<double> reference;
    std::size_t events = 0;
    const Layout data = make_layout(records,covariates,options.strata,reference,events,"test_proportional_hazards");
    if (events == 0)
        throw std::invalid_argument("test_proportional_hazards: no events");

    // Transformed time of every record's stop time, centered over event records.
    std::vector<double> g(n);
    switch (transform) {
    case TimeTransform::identity:
        g = data.stop;
        break;
    case TimeTransform::log:
        for (std::size_t i = 0; i < n; ++i) {
            if (!(data.stop[i] > 0))
                throw std::invalid_argument("test_proportional_hazards: log transform needs positive times");
            g[i] = std::log(data.stop[i]);
        }
        break;
    case TimeTransform::rank: {
        std::vector<std::size_t> order(n);
        std::iota(order.begin(),order.end(),0);
        std::ranges::sort(order,[&](std::size_t a, std::size_t b) { return data.stop[a] < data.stop[b]; });
        for (std::size_t i = 0; i < n;) {
            std::size_t j = i;
            while (j < n && data.stop[order[j]] == data.stop[order[i]])
                ++j;
            for (std::size_t k = i; k < j; ++k)
                g[order[k]] = (static_cast<double>(i+j)+1)/2;
            i = j;
        }
        break;
    }
    case TimeTransform::km: {
        // Pooled counting-process Kaplan–Meier, then 1 - S(t-) at each stop time.
        std::vector<double> stops = data.stop, starts = data.start, event_times;
        std::ranges::sort(stops);
        std::ranges::sort(starts);
        for (std::size_t i = 0; i < n; ++i)
            if (records[i].event)
                event_times.push_back(data.stop[i]);
        std::ranges::sort(event_times);
        std::vector<double> times, survival;
        double s = 1;
        for (std::size_t i = 0; i < event_times.size();) {
            const double t = event_times[i];
            std::size_t j = i;
            while (j < event_times.size() && event_times[j] == t)
                ++j;
            const auto exited = static_cast<double>(std::ranges::lower_bound(stops,t)-stops.begin());
            const auto entered = static_cast<double>(std::ranges::lower_bound(starts,t)-starts.begin());
            const double at_risk = entered-exited;
            s *= 1-static_cast<double>(j-i)/at_risk;
            times.push_back(t);
            survival.push_back(s);
            i = j;
        }
        for (std::size_t i = 0; i < n; ++i) {
            const auto before = std::ranges::lower_bound(times,data.stop[i])-times.begin();
            g[i] = 1-(before == 0 ? 1.0 : survival[static_cast<std::size_t>(before-1)]);
        }
        break;
    }
    default:
        throw std::invalid_argument("test_proportional_hazards: unknown transform");
    }
    double g_mean = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (records[i].event)
            g_mean += g[i]/static_cast<double>(events);

    const Evaluation e = evaluate(data,fit.coefficients,options.ties,true);
    Matrix imat(2*p,2*p);
    std::vector<double> u(2*p,0.0);
    struct Row {
        std::size_t stratum, record;
        double time;
        std::vector<double> residual;
    };
    std::vector<Row> rows;
    for (const auto& m : e.moments) {
        const double gk = g[m.deaths.front()]-g_mean;
        for (std::size_t a = 0; a < p; ++a)
            for (std::size_t b = 0; b < p; ++b) {
                imat(a,b) += m.variance(a,b);
                imat(a,b+p) += gk*m.variance(a,b);
                imat(a+p,b) += gk*m.variance(a,b);
                imat(a+p,b+p) += gk*gk*m.variance(a,b);
            }
        for (const std::size_t i : m.deaths) {
            Row row{m.stratum,i,m.time,std::vector<double>(p)};
            for (std::size_t a = 0; a < p; ++a) {
                row.residual[a] = data.x(i,a)-m.mean[a];
                u[a+p] += gk*row.residual[a];
            }
            rows.push_back(std::move(row));
        }
    }
    const auto quadratic_solve = [&](const std::vector<std::size_t>& keep) {
        Matrix sub(keep.size(),keep.size());
        std::vector<double> v(keep.size());
        for (std::size_t a = 0; a < keep.size(); ++a) {
            v[a] = u[keep[a]];
            for (std::size_t b = 0; b < keep.size(); ++b)
                sub(a,b) = imat(keep[a],keep[b]);
        }
        const auto factor = linalg::cholesky(sub);
        if (!factor.ok)
            throw std::runtime_error("test_proportional_hazards: joint information is singular");
        const auto solution = factor.solve(v);
        double q = 0;
        for (std::size_t a = 0; a < keep.size(); ++a)
            q += solution[a]*v[a];
        return q;
    };
    ProportionalHazardsTest result;
    std::vector<std::size_t> base(p);
    std::iota(base.begin(),base.end(),0);
    for (std::size_t a = 0; a < p; ++a) {
        auto keep = base;
        keep.push_back(p+a);
        result.terms.push_back(chi_square(quadratic_solve(keep),1));
    }
    std::vector<std::size_t> all(2*p);
    std::iota(all.begin(),all.end(),0);
    result.global = chi_square(quadratic_solve(all),p);

    // Scaling: events per stratum for columns that vary within it.
    const std::size_t strata = data.strata.size();
    std::vector<std::vector<double>> used(strata,std::vector<double>(p,0.0));
    for (std::size_t s = 0; s < strata; ++s) {
        double deaths = 0;
        for (const auto& d : data.strata[s].deaths)
            deaths += static_cast<double>(d.size());
        for (std::size_t a = 0; a < p; ++a) {
            const auto& members = data.strata[s].by_stop;
            for (const std::size_t i : members)
                if (data.x(i,a) != data.x(members.front(),a)) {
                    used[s][a] = deaths;
                    break;
                }
        }
    }
    Matrix vmean(p,p);
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b < p; ++b) {
            double w = 0;
            for (std::size_t s = 0; s < strata; ++s)
                w += std::min(used[s][a],used[s][b]);
            vmean(a,b) = imat(a,b)/(w == 0 ? 1.0 : w);
        }
    std::ranges::sort(rows,[](const Row& a, const Row& b) {
        if (a.stratum != b.stratum)
            return a.stratum < b.stratum;
        if (a.time != b.time)
            return a.time < b.time;
        return a.record < b.record;
    });
    result.schoenfeld = Matrix(rows.size(),p);
    result.scaled_schoenfeld = Matrix(rows.size(),p,std::numeric_limits<double>::quiet_NaN());
    std::vector<std::optional<linalg::CholeskyDecomposition<double>>> factors(strata);
    std::vector<std::vector<std::size_t>> active(strata);
    for (std::size_t s = 0; s < strata; ++s) {
        for (std::size_t a = 0; a < p; ++a)
            if (used[s][a] > 0)
                active[s].push_back(a);
        if (active[s].empty())
            continue;
        Matrix sub(active[s].size(),active[s].size());
        for (std::size_t a = 0; a < active[s].size(); ++a)
            for (std::size_t b = 0; b < active[s].size(); ++b)
                sub(a,b) = vmean(active[s][a],active[s][b]);
        factors[s] = linalg::cholesky(sub);
        if (!factors[s]->ok)
            throw std::runtime_error("test_proportional_hazards: residual scaling matrix is singular");
    }
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto& row = rows[r];
        result.event_records.push_back(row.record);
        result.event_strata.push_back(data.strata[row.stratum].label);
        result.event_times.push_back(row.time);
        result.transformed_times.push_back(g[row.record]);
        for (std::size_t a = 0; a < p; ++a)
            result.schoenfeld(r,a) = row.residual[a];
        if (!factors[row.stratum])
            continue;
        std::vector<double> v;
        for (const std::size_t a : active[row.stratum])
            v.push_back(row.residual[a]);
        const auto scaled = factors[row.stratum]->solve(v);
        for (std::size_t k = 0; k < active[row.stratum].size(); ++k) {
            const std::size_t a = active[row.stratum][k];
            result.scaled_schoenfeld(r,a) = scaled[k]+fit.coefficients[a];
        }
    }
    return result;
}

} // namespace datamunge::stats
