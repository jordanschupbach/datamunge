/// @file msm.cpp
/// @brief Time-varying inverse probability weights and marginal structural model fits.
#include <datamunge/causal/msm.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>
#include "detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Double-valued covariate matrix.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Fitted probabilities and coefficients of a pooled logistic model.
struct LogisticFit {
    std::vector<double> probability; ///< Fitted P(event = 1) on the selected rows, in row order.
    std::vector<double> coefficients; ///< Intercept and slopes.
};

/// @brief Fits a logistic regression of the event on the covariates over the selected rows.
///
/// Newton-Raphson (IRLS) on centered and scaled covariates, which is fast for the long person-time
/// data of weight models; the coefficients are transformed back to the original scale.
LogisticFit pooled_logistic(const std::vector<int>& event, const Matrix& x, const std::vector<std::size_t>& rows) {
    const std::size_t m = rows.size(), p = x.cols() + 1;
    if (m <= p)
        throw std::runtime_error("longitudinal_weights: too few rows for a weight model");
    std::vector<double> center(x.cols(), 0.0), scale(x.cols(), 1.0);
    for (std::size_t j = 0; j < x.cols(); ++j) {
        double s = 0, ss = 0;
        for (const std::size_t i : rows)
            s += x(i, j);
        center[j] = s / static_cast<double>(m);
        for (const std::size_t i : rows)
            ss += (x(i, j) - center[j]) * (x(i, j) - center[j]);
        scale[j] = ss > 0 ? std::sqrt(ss / static_cast<double>(m)) : 1.0;
    }
    const auto z = [&](std::size_t i, std::size_t j) { return j == 0 ? 1.0 : (x(i, j - 1) - center[j - 1]) / scale[j - 1]; };
    std::vector<double> beta(p, 0.0), eta(m, 0.0), mu(m, 0.5);
    double deviance = std::numeric_limits<double>::infinity();
    bool converged = false;
    for (int it = 0; it < 50 && !converged; ++it) {
        Matrix info(p, p);
        std::vector<double> score(p, 0.0);
        for (std::size_t r = 0; r < m; ++r) {
            const std::size_t i = rows[r];
            const double w = mu[r] * (1 - mu[r]), resid = event[i] - mu[r];
            for (std::size_t a = 0; a < p; ++a) {
                const double za = z(i, a);
                score[a] += za * resid;
                for (std::size_t b = 0; b <= a; ++b)
                    info(a, b) += w * za * z(i, b);
            }
        }
        for (std::size_t a = 0; a < p; ++a)
            for (std::size_t b = a + 1; b < p; ++b)
                info(a, b) = info(b, a);
        const auto chol = linalg::cholesky(info);
        if (!chol.ok)
            throw std::runtime_error("longitudinal_weights: singular weight model (collinear covariates or separation)");
        const auto step = chol.solve(score);
        // Step halving on the deviance keeps the iteration monotone.
        double factor = 1.0, next_deviance = 0;
        std::vector<double> trial(p), trial_eta(m), trial_mu(m);
        for (int half = 0; half < 30; ++half) {
            for (std::size_t a = 0; a < p; ++a)
                trial[a] = beta[a] + factor * step[a];
            next_deviance = 0;
            for (std::size_t r = 0; r < m; ++r) {
                double e = 0;
                for (std::size_t a = 0; a < p; ++a)
                    e += z(rows[r], a) * trial[a];
                trial_eta[r] = e;
                trial_mu[r] = 1 / (1 + std::exp(-e));
                const double pr = event[rows[r]] ? trial_mu[r] : 1 - trial_mu[r];
                next_deviance -= 2 * std::log(std::max(pr, 1e-300));
            }
            if (next_deviance <= deviance + 1e-12 * std::abs(deviance) || !std::isfinite(deviance))
                break;
            factor /= 2;
        }
        converged = std::isfinite(deviance) && std::abs(deviance - next_deviance) < 1e-10 * (std::abs(next_deviance) + 0.1);
        beta = trial;
        eta = trial_eta;
        mu = trial_mu;
        deviance = next_deviance;
    }
    if (!converged)
        throw std::runtime_error("longitudinal_weights: logistic weight model did not converge");
    if (std::ranges::any_of(mu, [](double v) { return v < 1e-10 || v > 1 - 1e-10; }))
        throw std::runtime_error("longitudinal_weights: logistic weight model separates the events");
    LogisticFit out;
    out.probability = mu;
    out.coefficients.assign(p, 0.0);
    out.coefficients[0] = beta[0];
    for (std::size_t j = 1; j < p; ++j) {
        out.coefficients[j] = beta[j] / scale[j - 1];
        out.coefficients[0] -= out.coefficients[j] * center[j - 1];
    }
    return out;
}

/// @brief R's type-7 quantile of a sample.
double quantile7(std::vector<double> v, double p) {
    std::ranges::sort(v);
    const double h = (static_cast<double>(v.size()) - 1) * p;
    const auto lo = static_cast<std::size_t>(std::floor(h));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (h - static_cast<double>(lo)) * (v[hi] - v[lo]);
}
} // namespace

LongitudinalWeights longitudinal_weights(const std::vector<std::size_t>& id, const LongitudinalWeightModel& model,
                                         const LongitudinalWeightOptions& options) {
    const std::size_t n = id.size();
    if (n == 0 || model.event.size() != n || model.denominator.rows() != n ||
        (options.stabilized && model.numerator.cols() > 0 && model.numerator.rows() != n))
        throw std::invalid_argument("longitudinal_weights: id, events and covariates must have one entry per row");
    if (!(options.truncation >= 0 && options.truncation < 0.5))
        throw std::invalid_argument("longitudinal_weights: truncation must lie in [0, 0.5)");
    detail::finite_matrix(model.denominator);
    detail::finite_matrix(model.numerator);
    std::set<std::size_t> finished;
    for (std::size_t i = 0; i < n; ++i) {
        if (model.event[i] != 0 && model.event[i] != 1)
            throw std::invalid_argument("longitudinal_weights: events must be 0 or 1");
        if (i > 0 && id[i] != id[i - 1] && !finished.insert(id[i - 1]).second)
            throw std::invalid_argument("longitudinal_weights: each individual's rows must be contiguous");
        if (i > 0 && id[i] != id[i - 1] && finished.contains(id[i]))
            throw std::invalid_argument("longitudinal_weights: each individual's rows must be contiguous");
    }
    LongitudinalWeights out;
    out.at_risk.assign(n, true);
    if (options.risk_set == WeightRiskSet::until_first_event) {
        bool seen = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (i == 0 || id[i] != id[i - 1])
                seen = false;
            out.at_risk[i] = !seen;
            seen = seen || model.event[i] == 1;
        }
    }
    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < n; ++i)
        if (out.at_risk[i])
            rows.push_back(i);
    const double na = std::numeric_limits<double>::quiet_NaN();
    out.numerator_probability.assign(n, na);
    out.denominator_probability.assign(n, na);
    const Matrix numerator_x = model.numerator.cols() > 0 ? model.numerator : Matrix(n, 0);
    const auto den = pooled_logistic(model.event, model.denominator, rows);
    out.denominator_coefficients = den.coefficients;
    LogisticFit num;
    if (options.stabilized) {
        num = pooled_logistic(model.event, numerator_x, rows);
        out.numerator_coefficients = num.coefficients;
    }
    for (std::size_t r = 0; r < rows.size(); ++r) {
        out.denominator_probability[rows[r]] = den.probability[r];
        out.numerator_probability[rows[r]] = options.stabilized ? num.probability[r] : 1.0;
    }
    out.weights.assign(n, 1.0);
    double cumulative = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (i == 0 || id[i] != id[i - 1])
            cumulative = 1.0;
        if (out.at_risk[i]) {
            const double pd = out.denominator_probability[i];
            const double pn = out.numerator_probability[i];
            cumulative *= model.event[i] ? (options.stabilized ? pn : 1.0) / pd
                                         : (options.stabilized ? 1 - pn : 1.0) / (1 - pd);
        }
        out.weights[i] = cumulative;
    }
    out.truncated = out.weights;
    if (options.truncation > 0) {
        const double lo = quantile7(out.weights, options.truncation);
        const double hi = quantile7(out.weights, 1 - options.truncation);
        for (double& w : out.truncated)
            w = std::clamp(w, lo, hi);
    }
    out.min = *std::ranges::min_element(out.weights);
    out.max = *std::ranges::max_element(out.weights);
    for (const double w : out.weights)
        out.mean += w / static_cast<double>(n);
    return out;
}

MsmFit fit_marginal_structural_model(const std::vector<double>& outcome, const linalg::DenseMatrix<double>& design,
                                     const std::vector<double>& weights, const std::vector<std::size_t>& id,
                                     MsmFamily family, double confidence) {
    const std::size_t n = outcome.size(), p = design.cols() + 1;
    if (n == 0 || weights.size() != n || id.size() != n || (design.cols() > 0 && design.rows() != n))
        throw std::invalid_argument("fit_marginal_structural_model: outcome, design, weights and id must match");
    if (!(confidence > 0 && confidence < 1))
        throw std::invalid_argument("fit_marginal_structural_model: confidence must lie in (0, 1)");
    detail::finite_matrix(design);
    for (std::size_t i = 0; i < n; ++i) {
        if (!(weights[i] >= 0) || !std::isfinite(weights[i]) || !std::isfinite(outcome[i]))
            throw std::invalid_argument("fit_marginal_structural_model: weights must be nonnegative and values finite");
        if (family == MsmFamily::logistic && outcome[i] != 0 && outcome[i] != 1)
            throw std::invalid_argument("fit_marginal_structural_model: logistic outcomes must be 0 or 1");
    }
    const auto row = [&](std::size_t i, std::size_t j) { return j == 0 ? 1.0 : design(i, j - 1); };
    std::vector<double> beta(p, 0.0);
    MsmFit out;
    const auto mean_of = [&](double eta) { return family == MsmFamily::logistic ? 1 / (1 + std::exp(-eta)) : eta; };
    const auto variance_of = [&](double mu) { return family == MsmFamily::logistic ? mu * (1 - mu) : 1.0; };
    Matrix info(p, p);
    for (std::size_t it = 0; it < 100; ++it) {
        info = Matrix(p, p);
        std::vector<double> score(p, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double eta = 0;
            for (std::size_t j = 0; j < p; ++j)
                eta += row(i, j) * beta[j];
            const double mu = mean_of(eta), v = variance_of(mu);
            for (std::size_t a = 0; a < p; ++a) {
                score[a] += weights[i] * row(i, a) * (outcome[i] - mu);
                for (std::size_t b = 0; b < p; ++b)
                    info(a, b) += weights[i] * v * row(i, a) * row(i, b);
            }
        }
        const auto chol = linalg::cholesky(info);
        if (!chol.ok)
            throw std::runtime_error("fit_marginal_structural_model: rank-deficient weighted design");
        const auto step = chol.solve(score);
        double change = 0;
        for (std::size_t j = 0; j < p; ++j) {
            beta[j] += step[j];
            change = std::max(change, std::abs(step[j]) / (1 + std::abs(beta[j])));
        }
        out.iterations = it + 1;
        if (family == MsmFamily::linear || change < 1e-10) {
            out.converged = true;
            break;
        }
    }
    // Information at the solution and clustered score sums.
    info = Matrix(p, p);
    std::map<std::size_t, std::vector<double>> cluster;
    for (std::size_t i = 0; i < n; ++i) {
        double eta = 0;
        for (std::size_t j = 0; j < p; ++j)
            eta += row(i, j) * beta[j];
        const double mu = mean_of(eta), v = variance_of(mu);
        auto& u = cluster[id[i]];
        u.resize(p, 0.0);
        for (std::size_t a = 0; a < p; ++a) {
            u[a] += weights[i] * row(i, a) * (outcome[i] - mu);
            for (std::size_t b = 0; b < p; ++b)
                info(a, b) += weights[i] * v * row(i, a) * row(i, b);
        }
    }
    out.clusters = cluster.size();
    if (out.clusters < 2)
        throw std::invalid_argument("fit_marginal_structural_model: need at least two individuals");
    const auto chol = linalg::cholesky(info);
    if (!chol.ok)
        throw std::runtime_error("fit_marginal_structural_model: rank-deficient weighted design");
    std::vector<std::vector<double>> bread_u;
    bread_u.reserve(cluster.size());
    for (const auto& [g, u] : cluster)
        bread_u.push_back(chol.solve(u)); // B u_g
    const double G = static_cast<double>(out.clusters);
    const double z = random::normal_quantile(0.5 + confidence / 2);
    out.coefficients = beta;
    for (std::size_t j = 0; j < p; ++j) {
        double v = 0;
        for (const auto& bu : bread_u)
            v += bu[j] * bu[j];
        const double se = std::sqrt(v * G / (G - 1));
        out.standard_errors.push_back(se);
        out.z_values.push_back(beta[j] / se);
        out.p_values.push_back(2 * (1 - random::normal_cdf(std::abs(beta[j] / se))));
        out.lower.push_back(beta[j] - z * se);
        out.upper.push_back(beta[j] + z * se);
    }
    return out;
}

} // namespace datamunge::causal
