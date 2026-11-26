/// @file fine_gray.cpp
/// @brief IPCW-weighted partial likelihood, Newton fitting and the Fine–Gray sandwich variance.

#include <datamunge/stats/fine_gray.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal helpers for Fine–Gray regression.
namespace {

/// @brief Dense double matrix.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Data sorted by time with failure codes, censoring weights and covariates.
struct Data {
    std::size_t n{};                    ///< Observations.
    std::size_t p{};                    ///< Covariates.
    std::vector<double> t;              ///< Sorted times.
    std::vector<int> code;              ///< 1 cause of interest, 2 other cause, 0 censored.
    std::vector<std::size_t> group;     ///< Censoring group index.
    std::size_t groups{};               ///< Number of censoring groups.
    Matrix x;                           ///< Covariates in sorted order.
    Matrix G;                           ///< Censoring survival G_g(t_i-), groups by observations.
    std::vector<std::size_t> fail_start; ///< First sorted index of each distinct failure time.
};

/// @brief Weight of observation j in the subdistribution risk set at failure time s (index i).
/// @param d Data.
/// @param j Observation.
/// @param i Index of a failure at time s.
/// @param e exp(eta_j).
/// @return Risk weight, or 0 if j is not in the set.
double risk_weight(const Data& d, std::size_t j, std::size_t i, double e) {
    if (d.t[j] >= d.t[i])
        return e;
    if (d.code[j] == 2)
        return e*d.G(d.group[j],i)/d.G(d.group[j],j);
    return 0;
}

/// @brief Weighted log partial likelihood with score and information.
struct Evaluation {
    double value{};            ///< Log likelihood.
    std::vector<double> score; ///< Gradient.
    Matrix information;        ///< Negative Hessian.
};

/// @brief Linear predictors exp(x beta).
/// @param d Data.
/// @param beta Coefficients.
/// @return Risk scores.
std::vector<double> risk_scores(const Data& d, const std::vector<double>& beta) {
    std::vector<double> e(d.n);
    for (std::size_t i = 0; i < d.n; ++i) {
        double eta = 0;
        for (std::size_t a = 0; a < d.p; ++a)
            eta += d.x(i,a)*beta[a];
        e[i] = std::exp(eta);
    }
    return e;
}

/// @brief Evaluates the weighted partial likelihood (port of crrfsv).
/// @param d Data.
/// @param beta Coefficients.
/// @return Value, score and information.
Evaluation evaluate(const Data& d, const std::vector<double>& beta) {
    const std::size_t p = d.p;
    const auto e = risk_scores(d,beta);
    Evaluation r{0,std::vector<double>(p,0.0),Matrix(p,p)};
    for (const std::size_t i : d.fail_start) {
        double s0 = 0, deaths = 0;
        std::vector<double> s1(p,0.0);
        Matrix s2(p,p);
        for (std::size_t j = 0; j < d.n; ++j) {
            const double w = risk_weight(d,j,i,e[j]);
            if (w == 0)
                continue;
            s0 += w;
            for (std::size_t a = 0; a < p; ++a) {
                s1[a] += w*d.x(j,a);
                for (std::size_t b = 0; b <= a; ++b)
                    s2(a,b) += w*d.x(j,a)*d.x(j,b);
            }
        }
        for (std::size_t j = i; j < d.n && d.t[j] == d.t[i]; ++j)
            if (d.code[j] == 1) {
                deaths += 1;
                r.value += std::log(e[j]);
                for (std::size_t a = 0; a < p; ++a)
                    r.score[a] += d.x(j,a);
            }
        r.value -= deaths*std::log(s0);
        for (std::size_t a = 0; a < p; ++a) {
            r.score[a] -= deaths*s1[a]/s0;
            for (std::size_t b = 0; b <= a; ++b)
                r.information(a,b) += deaths*(s2(a,b)/s0-s1[a]*s1[b]/(s0*s0));
        }
    }
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b < a; ++b)
            r.information(b,a) = r.information(a,b);
    return r;
}

/// @brief Inverts a symmetric positive definite matrix.
/// @param a Matrix.
/// @return Inverse.
/// @throws std::runtime_error If not positive definite.
Matrix spd_inverse(const Matrix& a) {
    const auto factor = linalg::cholesky(a);
    if (!factor.ok)
        throw std::runtime_error("fit_fine_gray: information matrix is singular");
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

/// @brief Sum of outer products of score residuals with the censoring correction (port of crrvv).
/// @param d Data.
/// @param beta Coefficients at the estimate.
/// @return The middle matrix of the sandwich.
Matrix meat(const Data& d, const std::vector<double>& beta) {
    const std::size_t n = d.n, p = d.p, ng = d.groups;
    const auto e = risk_scores(d,beta);
    // Risk-set sums at every failure (identical within ties).
    std::vector<double> s0(n,0.0);
    Matrix s1(n,p);
    for (const std::size_t i : d.fail_start) {
        double a0 = 0;
        std::vector<double> a1(p,0.0);
        for (std::size_t j = 0; j < n; ++j) {
            const double w = risk_weight(d,j,i,e[j]);
            a0 += w;
            for (std::size_t a = 0; a < p; ++a)
                a1[a] += w*d.x(j,a);
        }
        for (std::size_t j = i; j < n && d.t[j] == d.t[i]; ++j) {
            s0[j] = a0;
            for (std::size_t a = 0; a < p; ++a)
                s1(j,a) = a1[a];
        }
    }
    std::vector<std::size_t> failures;
    for (std::size_t i = 0; i < n; ++i)
        if (d.code[i] == 1)
            failures.push_back(i);
    // Score residuals.
    Matrix st(n,p);
    for (std::size_t i = 0; i < n; ++i) {
        for (const std::size_t j : failures) {
            const double w = risk_weight(d,i,j,e[i]);
            if (w == 0)
                continue;
            for (std::size_t a = 0; a < p; ++a)
                st(i,a) -= (d.x(i,a)-s1(j,a)/s0[j])*w/s0[j];
        }
        if (d.code[i] == 1)
            for (std::size_t a = 0; a < p; ++a)
                st(i,a) += d.x(i,a)-s1(i,a)/s0[i];
    }
    // Censoring correction. For failures j1 at or after the current block, by censoring group:
    // B_g = sum G_g(t_j1-)/S0_j1 and C_g = sum G_g(t_j1-) xbar_j1 / S0_j1 (suffix sums), and for
    // other-cause failures j2 before the block A4_g = sum e/G_g(t_j2-), A3_g = sum e x/G_g(t_j2-).
    Matrix B(n+1,ng);
    std::vector<Matrix> C(ng,Matrix(n+1,p));
    // Suffix over sorted index: value at index i covers failures with index >= i.
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t g = 0; g < ng; ++g) {
            B(i,g) = B(i+1,g);
            for (std::size_t a = 0; a < p; ++a)
                C[g](i,a) = C[g](i+1,a);
        }
        if (d.code[i] == 1) {
            const std::size_t g = d.group[i];
            const double w = d.G(g,i)/s0[i];
            B(i,g) += w;
            for (std::size_t a = 0; a < p; ++a)
                C[g](i,a) += w*s1(i,a)/s0[i];
        }
    }
    std::vector<double> A4(ng,0.0), icrsk(ng,0.0);
    Matrix A3(ng,p), qu(ng,p), ss2(ng,p);
    for (std::size_t i = 0; i < n; ++i)
        icrsk[d.group[i]] += 1;
    std::size_t prefix = 0; // other-cause failures with index < prefix are in A3/A4
    Matrix meat_matrix(p,p);
    for (std::size_t lc = 0; lc < n;) {
        std::size_t end = lc;
        while (end < n && d.t[end] == d.t[lc])
            ++end;
        for (; prefix < lc; ++prefix)
            if (d.code[prefix] == 2) {
                const std::size_t g = d.group[prefix];
                const double w = e[prefix]/d.G(g,prefix);
                A4[g] += w;
                for (std::size_t a = 0; a < p; ++a)
                    A3(g,a) += w*d.x(prefix,a);
            }
        bool censored = false;
        for (std::size_t j = lc; j < end; ++j)
            censored = censored || d.code[j] == 0;
        if (censored) {
            for (std::size_t g = 0; g < ng; ++g)
                for (std::size_t a = 0; a < p; ++a)
                    qu(g,a) = A3(g,a)*B(lc,g)-A4[g]*C[g](lc,a);
            for (std::size_t j = lc; j < end; ++j)
                if (d.code[j] == 0) {
                    const std::size_t g = d.group[j];
                    for (std::size_t a = 0; a < p; ++a)
                        ss2(g,a) -= qu(g,a)/(icrsk[g]*icrsk[g]);
                }
        }
        for (std::size_t i = lc; i < end; ++i) {
            const std::size_t g = d.group[i];
            std::vector<double> s(p);
            for (std::size_t a = 0; a < p; ++a) {
                s[a] = st(i,a)+ss2(g,a);
                if (d.code[i] == 0)
                    s[a] += qu(g,a)/icrsk[g];
            }
            for (std::size_t a = 0; a < p; ++a)
                for (std::size_t b = 0; b < p; ++b)
                    meat_matrix(a,b) += s[a]*s[b];
        }
        for (std::size_t j = lc; j < end; ++j)
            icrsk[d.group[j]] -= 1;
        lc = end;
    }
    return meat_matrix;
}

} // namespace

double FineGrayFit::predict_incidence(double time, const std::vector<double>& covariates) const {
    if (covariates.size() != coefficients.size())
        throw std::invalid_argument("FineGrayFit::predict_incidence: wrong covariate length");
    double eta = 0;
    for (std::size_t a = 0; a < coefficients.size(); ++a)
        eta += covariates[a]*coefficients[a];
    double h = 0;
    for (std::size_t k = 0; k < event_times.size() && event_times[k] <= time; ++k)
        h = baseline_cumulative_hazard[k];
    return 1-std::exp(-h*std::exp(eta));
}

FineGrayFit fit_fine_gray(const std::vector<CompetingRiskObservation>& observations,
                          const linalg::DenseMatrix<double>& covariates, const FineGrayOptions& options) {
    const std::size_t n = observations.size(), p = covariates.cols();
    if (n == 0 || covariates.rows() != n || p == 0)
        throw std::invalid_argument("fit_fine_gray: need one nonempty covariate row per observation");
    if (!options.censoring_groups.empty() && options.censoring_groups.size() != n)
        throw std::invalid_argument("fit_fine_gray: need one censoring group per observation");
    if (!(options.confidence > 0 && options.confidence < 1) || options.cause <= 0)
        throw std::invalid_argument("fit_fine_gray: invalid confidence or cause");
    std::vector<std::size_t> order(n);
    std::iota(order.begin(),order.end(),0);
    std::ranges::stable_sort(order,[&](std::size_t a, std::size_t b) { return observations[a].time < observations[b].time; });
    std::vector<int> labels = options.censoring_groups;
    if (labels.empty())
        labels.assign(n,0);
    std::vector<int> distinct = labels;
    std::ranges::sort(distinct);
    distinct.erase(std::unique(distinct.begin(),distinct.end()),distinct.end());
    Data d;
    d.n = n;
    d.p = p;
    d.groups = distinct.size();
    d.x = Matrix(n,p);
    std::size_t failures = 0;
    for (std::size_t k = 0; k < n; ++k) {
        const auto& o = observations[order[k]];
        if (!std::isfinite(o.time) || o.time < 0 || o.cause < 0)
            throw std::invalid_argument("fit_fine_gray: invalid time or cause");
        d.t.push_back(o.time);
        d.code.push_back(o.cause == 0 ? 0 : o.cause == options.cause ? 1 : 2);
        failures += d.code.back() == 1;
        d.group.push_back(static_cast<std::size_t>(std::ranges::lower_bound(distinct,labels[order[k]])-distinct.begin()));
        for (std::size_t a = 0; a < p; ++a) {
            if (!std::isfinite(covariates(order[k],a)))
                throw std::invalid_argument("fit_fine_gray: covariates must be finite");
            d.x(k,a) = covariates(order[k],a);
        }
    }
    if (failures == 0)
        throw std::invalid_argument("fit_fine_gray: no failures from the cause");
    // Kaplan–Meier of censoring within each group, evaluated just before each time.
    d.G = Matrix(d.groups,n,1.0);
    for (std::size_t g = 0; g < d.groups; ++g) {
        std::vector<double> times, values;
        double s = 1;
        std::size_t at_risk = 0;
        for (std::size_t k = 0; k < n; ++k)
            at_risk += d.group[k] == g;
        for (std::size_t k = 0; k < n;) {
            std::size_t end = k;
            double censored = 0, members = 0;
            while (end < n && d.t[end] == d.t[k]) {
                if (d.group[end] == g) {
                    members += 1;
                    censored += d.code[end] == 0;
                }
                ++end;
            }
            if (censored > 0) {
                s *= 1-censored/static_cast<double>(at_risk);
                times.push_back(d.t[k]);
                values.push_back(s);
            }
            at_risk -= static_cast<std::size_t>(members);
            k = end;
        }
        for (std::size_t k = 0; k < n; ++k) {
            const auto before = std::ranges::lower_bound(times,d.t[k])-times.begin();
            d.G(g,k) = before == 0 ? 1.0 : values[static_cast<std::size_t>(before-1)];
        }
    }
    for (std::size_t k = 0; k < n; ++k)
        if (d.code[k] == 1 && (d.fail_start.empty() || d.t[d.fail_start.back()] != d.t[k]))
            d.fail_start.push_back(k);

    FineGrayFit fit;
    std::vector<double> beta(p,0.0);
    auto current = evaluate(d,beta);
    fit.null_log_likelihood = current.value;
    while (fit.iterations < options.max_iterations) {
        const auto factor = linalg::cholesky(current.information);
        if (!factor.ok)
            throw std::runtime_error("fit_fine_gray: information matrix is singular");
        const auto step = factor.solve(current.score);
        double scale = 1;
        std::vector<double> candidate(p);
        Evaluation trial;
        for (int halving = 0;; ++halving) {
            for (std::size_t a = 0; a < p; ++a)
                candidate[a] = beta[a]+scale*step[a];
            trial = evaluate(d,candidate);
            if (std::isfinite(trial.value) && trial.value >= current.value-1e-12*std::abs(current.value))
                break;
            if (halving == 30)
                throw std::runtime_error("fit_fine_gray: step halving failed");
            scale /= 2;
        }
        const double change = std::abs(trial.value-current.value)/(std::abs(trial.value)+1e-10);
        beta = candidate;
        current = std::move(trial);
        ++fit.iterations;
        if (change <= options.tolerance) {
            fit.converged = true;
            break;
        }
    }
    fit.coefficients = beta;
    fit.log_likelihood = current.value;
    fit.information = current.information;
    const Matrix inverse = spd_inverse(current.information);
    const Matrix middle = meat(d,beta);
    fit.covariance = Matrix(p,p);
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b < p; ++b) {
            double v = 0;
            for (std::size_t c = 0; c < p; ++c)
                for (std::size_t k = 0; k < p; ++k)
                    v += inverse(a,c)*middle(c,k)*inverse(k,b);
            fit.covariance(a,b) = v;
        }
    const double z = random::normal_quantile(.5+options.confidence/2);
    for (std::size_t a = 0; a < p; ++a) {
        const double se = std::sqrt(fit.covariance(a,a));
        fit.standard_errors.push_back(se);
        fit.hazard_ratios.push_back(std::exp(beta[a]));
        fit.hazard_ratio_lower.push_back(std::exp(beta[a]-z*se));
        fit.hazard_ratio_upper.push_back(std::exp(beta[a]+z*se));
        fit.p_values.push_back(2*random::normal_cdf(-std::abs(beta[a])/se));
    }
    // Baseline jumps at covariates zero (port of crrfit).
    const auto e = risk_scores(d,beta);
    double h = 0;
    for (const std::size_t i : d.fail_start) {
        double s0 = 0, deaths = 0;
        for (std::size_t j = 0; j < n; ++j)
            s0 += risk_weight(d,j,i,e[j]);
        for (std::size_t j = i; j < n && d.t[j] == d.t[i]; ++j)
            deaths += d.code[j] == 1;
        h += deaths/s0;
        fit.event_times.push_back(d.t[i]);
        fit.baseline_cumulative_hazard.push_back(h);
    }
    return fit;
}

} // namespace datamunge::stats
