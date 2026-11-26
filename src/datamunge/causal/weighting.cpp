/// @file weighting.cpp
/// @brief Entropy balancing and weighted treatment contrasts.
#include <datamunge/causal/weighting.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/survey_calibration.hpp>
#include "detail.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Double-valued covariate matrix.
using Matrix = linalg::DenseMatrix<double>;

/// @brief True when every value of a column is 0 or 1.
bool zero_one(const Matrix& x, std::size_t j) {
    for (std::size_t i = 0; i < x.rows(); ++i)
        if (x(i, j) != 0.0 && x(i, j) != 1.0)
            return false;
    return true;
}

/// @brief Moment functions to balance: the covariates and, for two moments, squares of non-binary ones.
Matrix moment_columns(const Matrix& x, std::size_t moments) {
    std::vector<std::size_t> squared;
    if (moments == 2)
        for (std::size_t j = 0; j < x.cols(); ++j)
            if (!zero_one(x, j))
                squared.push_back(j);
    Matrix c(x.rows(), x.cols() + squared.size());
    for (std::size_t i = 0; i < x.rows(); ++i) {
        for (std::size_t j = 0; j < x.cols(); ++j)
            c(i, j) = x(i, j);
        for (std::size_t k = 0; k < squared.size(); ++k)
            c(i, x.cols() + k) = x(i, squared[k]) * x(i, squared[k]);
    }
    return c;
}

/// @brief Base-weighted mean of each moment over the units selected by a predicate.
template<class Select>
std::vector<double> target_means(const Matrix& c, const std::vector<double>& q, Select select) {
    std::vector<double> m(c.cols(), 0.0);
    double total = 0;
    for (std::size_t i = 0; i < c.rows(); ++i)
        if (select(i)) {
            total += q[i];
            for (std::size_t j = 0; j < c.cols(); ++j)
                m[j] += q[i] * c(i, j);
        }
    for (double& v : m)
        v /= total;
    return m;
}

/// @brief Base-weighted standard deviation of each covariate over the selected units (0/1: sqrt(p(1-p))).
template<class Select>
std::vector<double> reference_sd(const Matrix& x, const std::vector<double>& q, Select select) {
    const auto mean = target_means(x, q, select);
    std::vector<double> sd(x.cols(), 0.0);
    double total = 0, total2 = 0;
    for (std::size_t i = 0; i < x.rows(); ++i)
        if (select(i)) {
            total += q[i];
            total2 += q[i] * q[i];
            for (std::size_t j = 0; j < x.cols(); ++j)
                sd[j] += q[i] * (x(i, j) - mean[j]) * (x(i, j) - mean[j]);
        }
    for (std::size_t j = 0; j < x.cols(); ++j)
        sd[j] = zero_one(x, j) ? std::sqrt(mean[j] * (1 - mean[j])) : std::sqrt(sd[j] / (total - total2 / total));
    return sd;
}
} // namespace

EntropyBalanceResult entropy_balance(const std::vector<int>& treatment, const linalg::DenseMatrix<double>& covariates,
                                     const EntropyBalanceOptions& options) {
    const std::size_t n = treatment.size();
    std::array<std::size_t, 2> count{};
    for (const int a : treatment) {
        if (a != 0 && a != 1)
            throw std::invalid_argument("entropy_balance: treatment must be 0 or 1");
        ++count[static_cast<std::size_t>(a)];
    }
    if (count[0] == 0 || count[1] == 0)
        throw std::invalid_argument("entropy_balance: both treatment groups need units");
    if (covariates.rows() != n || covariates.cols() == 0)
        throw std::invalid_argument("entropy_balance: covariates need one row per unit and at least one column");
    detail::finite_matrix(covariates);
    if (options.moments != 1 && options.moments != 2)
        throw std::invalid_argument("entropy_balance: moments must be 1 or 2");
    const std::vector<double> q = options.base_weights.empty() ? std::vector<double>(n, 1.0) : options.base_weights;
    if (q.size() != n || !std::ranges::all_of(q, [](double v) { return v > 0 && std::isfinite(v); }))
        throw std::invalid_argument("entropy_balance: base weights must be positive and one per unit");

    const Matrix c = moment_columns(covariates, options.moments);
    const auto all = [](std::size_t) { return true; };
    const auto in_group = [&](int arm) { return [&, arm](std::size_t i) { return treatment[i] == arm; }; };
    const auto target = options.estimand == WeightingEstimand::att   ? target_means(c, q, in_group(1))
                        : options.estimand == WeightingEstimand::atc ? target_means(c, q, in_group(0))
                                                                     : target_means(c, q, all);
    const auto sd_all = reference_sd(c, q, all);

    EntropyBalanceResult out;
    out.weights = q;
    out.converged = true;
    for (int arm = 0; arm < 2; ++arm) {
        const bool reweight = options.estimand == WeightingEstimand::ate ||
                              (options.estimand == WeightingEstimand::att && arm == 0) ||
                              (options.estimand == WeightingEstimand::atc && arm == 1);
        if (!reweight)
            continue;
        std::vector<std::size_t> rows;
        for (std::size_t i = 0; i < n; ++i)
            if (treatment[i] == arm)
                rows.push_back(i);
        // Constraints: total weight, and centered, scaled moments summing to zero.
        Matrix aux(rows.size(), c.cols() + 1);
        std::vector<double> base(rows.size());
        double total = 0;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            base[r] = q[rows[r]];
            total += base[r];
            aux(r, 0) = 1.0;
            for (std::size_t j = 0; j < c.cols(); ++j)
                aux(r, j + 1) = (c(rows[r], j) - target[j]) / (sd_all[j] > 0 ? sd_all[j] : 1.0);
        }
        std::vector<double> targets(c.cols() + 1, 0.0);
        targets[0] = total;
        const auto fit = stats::calibrate_weights(base, aux, targets,
                                                  {.distance = stats::CalibrationDistance::raking,
                                                   .tolerance = options.tolerance,
                                                   .max_iterations = options.max_iterations});
        out.iterations += fit.iterations;
        out.converged = out.converged && fit.converged;
        for (std::size_t r = 0; r < rows.size(); ++r)
            out.weights[rows[r]] = fit.weights[r];
    }
    std::array<double, 2> sw{}, sw2{}, largest{};
    for (std::size_t i = 0; i < n; ++i) {
        const auto g = static_cast<std::size_t>(treatment[i]);
        sw[g] += out.weights[i];
        sw2[g] += out.weights[i] * out.weights[i];
        largest[g] = std::max(largest[g], out.weights[i]);
    }
    for (std::size_t g = 0; g < 2; ++g) {
        out.effective_sample_size[g] = sw[g] * sw[g] / sw2[g];
        out.max_weight_share = std::max(out.max_weight_share, largest[g] / sw[g]);
    }
    // Balance on the estimand's SD scale: treated (ATT), control (ATC) or pooled within-group (ATE).
    std::vector<double> sd;
    if (options.estimand == WeightingEstimand::att)
        sd = reference_sd(covariates, q, in_group(1));
    else if (options.estimand == WeightingEstimand::atc)
        sd = reference_sd(covariates, q, in_group(0));
    else {
        const auto s1 = reference_sd(covariates, q, in_group(1)), s0 = reference_sd(covariates, q, in_group(0));
        for (std::size_t j = 0; j < s1.size(); ++j)
            sd.push_back(std::sqrt((s1[j] * s1[j] + s0[j] * s0[j]) / 2));
    }
    out.before = covariate_balance(treatment, covariates, q, sd);
    out.after = covariate_balance(treatment, covariates, out.weights, sd);
    return out;
}

MatchedEffect weighted_effect(const std::vector<double>& outcome, const std::vector<int>& treatment,
                              const std::vector<double>& weights, double confidence) {
    return detail::weighted_contrast(outcome, treatment, weights, {}, false, confidence);
}

MatchedEffect weighted_effect(const std::vector<double>& outcome, const std::vector<int>& treatment,
                              const std::vector<double>& weights, const linalg::DenseMatrix<double>& covariates,
                              double confidence) {
    const std::size_t n = treatment.size();
    if (outcome.size() != n || weights.size() != n || covariates.rows() != n)
        throw std::invalid_argument("weighted_effect: outcome, treatment, weights and covariates must match");
    if (!(confidence > 0 && confidence < 1))
        throw std::invalid_argument("weighted_effect: confidence must lie in (0, 1)");
    detail::finite_matrix(covariates);
    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < n; ++i) {
        if (!(weights[i] >= 0) || !std::isfinite(weights[i]))
            throw std::invalid_argument("weighted_effect: weights must be finite and nonnegative");
        if (weights[i] == 0)
            continue;
        if (!std::isfinite(outcome[i]) || (treatment[i] != 0 && treatment[i] != 1))
            throw std::invalid_argument("weighted_effect: outcomes must be finite and treatment 0 or 1");
        rows.push_back(i);
    }
    const std::size_t p = covariates.cols() + 2, m = rows.size();
    if (m <= p)
        throw std::invalid_argument("weighted_effect: too few weighted units for the adjusted regression");
    // Design (1, a, standardized covariates).
    Matrix x(m, p);
    for (std::size_t r = 0; r < m; ++r) {
        x(r, 0) = 1.0;
        x(r, 1) = treatment[rows[r]];
    }
    for (std::size_t j = 0; j < covariates.cols(); ++j) {
        double mean = 0, ss = 0;
        for (const std::size_t i : rows)
            mean += covariates(i, j) / static_cast<double>(m);
        for (const std::size_t i : rows)
            ss += (covariates(i, j) - mean) * (covariates(i, j) - mean);
        const double scale = ss > 0 ? std::sqrt(ss / static_cast<double>(m)) : 1.0;
        for (std::size_t r = 0; r < m; ++r)
            x(r, j + 2) = (covariates(rows[r], j) - mean) / scale;
    }
    Matrix xtwx(p, p);
    std::vector<double> xtwy(p, 0.0);
    for (std::size_t r = 0; r < m; ++r) {
        const double w = weights[rows[r]];
        for (std::size_t a = 0; a < p; ++a) {
            xtwy[a] += w * x(r, a) * outcome[rows[r]];
            for (std::size_t b = 0; b < p; ++b)
                xtwx(a, b) += w * x(r, a) * x(r, b);
        }
    }
    const auto chol = linalg::cholesky(xtwx);
    if (!chol.ok)
        throw std::runtime_error("weighted_effect: rank-deficient weighted design");
    const auto beta = chol.solve(xtwy);
    std::vector<double> e1(p, 0.0);
    e1[1] = 1.0;
    const auto row = chol.solve(e1); // second row of (X'WX)^{-1}
    double meat = 0;
    for (std::size_t r = 0; r < m; ++r) {
        double fit = 0, proj = 0;
        for (std::size_t a = 0; a < p; ++a) {
            fit += x(r, a) * beta[a];
            proj += x(r, a) * row[a];
        }
        const double u = weights[rows[r]] * (outcome[rows[r]] - fit) * proj;
        meat += u * u;
    }
    MatchedEffect out;
    out.estimate = beta[1];
    out.standard_error = std::sqrt(meat);
    out.df = static_cast<double>(m - p);
    out.p_value = 2 * (1 - random::student_t_cdf(std::abs(out.estimate / out.standard_error), out.df));
    const double q = random::student_t_quantile(0.5 + confidence / 2, out.df);
    out.lower = out.estimate - q * out.standard_error;
    out.upper = out.estimate + q * out.standard_error;
    return out;
}

} // namespace datamunge::causal
