/// @file matching.cpp
/// @brief Propensity score matching, balance statistics and matched effects.
#include <datamunge/causal/matching.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/glm.hpp>
#include "detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Double-valued covariate matrix.
using Matrix = linalg::DenseMatrix<double>;
/// @brief Marker for units without a matched set.
constexpr std::size_t no_set = std::numeric_limits<std::size_t>::max();

/// @brief Validates indicators and returns the size of each group.
std::array<std::size_t, 2> group_sizes(const std::vector<int>& treatment) {
    std::array<std::size_t, 2> counts{};
    for (const int a : treatment) {
        if (a != 0 && a != 1)
            throw std::invalid_argument("matching: treatment must be 0 or 1");
        ++counts[static_cast<std::size_t>(a)];
    }
    if (counts[0] == 0 || counts[1] == 0)
        throw std::invalid_argument("matching: both treatment groups need units");
    return counts;
}

/// @brief Checks that a covariate matrix has one finite row per unit (zero columns allowed).
void check_covariates(const Matrix& x, std::size_t n) {
    if (x.cols() > 0 && x.rows() != n)
        throw std::invalid_argument("matching: covariates need one row per unit");
    detail::finite_matrix(x);
}

/// @brief True when every value of a column is 0 or 1 (a binary indicator).
bool is_binary(const Matrix& x, std::size_t j) {
    for (std::size_t i = 0; i < x.rows(); ++i)
        if (x(i, j) != 0.0 && x(i, j) != 1.0)
            return false;
    return true;
}

/// @brief Weighted mean and normalized-weight unbiased variance of one group of a column.
std::pair<double, double> weighted_moments(const Matrix& x, std::size_t j, const std::vector<int>& a, int arm,
                                           const std::vector<double>& w) {
    double sw = 0, sw2 = 0, m = 0;
    for (std::size_t i = 0; i < x.rows(); ++i)
        if (a[i] == arm && w[i] > 0) {
            sw += w[i];
            sw2 += w[i] * w[i];
            m += w[i] * x(i, j);
        }
    m /= sw;
    double ss = 0;
    for (std::size_t i = 0; i < x.rows(); ++i)
        if (a[i] == arm && w[i] > 0)
            ss += w[i] * (x(i, j) - m) * (x(i, j) - m);
    const double denom = sw - sw2 / sw; // sum(w) (1 - sum(normalized w^2))
    return {m, denom > 0 ? ss / denom : std::numeric_limits<double>::quiet_NaN()};
}

/// @brief Largest gap between the two groups' weighted empirical CDFs of a column.
double ecdf_distance(const Matrix& x, std::size_t j, const std::vector<int>& a, const std::vector<double>& w) {
    std::vector<std::size_t> order;
    std::array<double, 2> total{};
    for (std::size_t i = 0; i < x.rows(); ++i)
        if (w[i] > 0) {
            order.push_back(i);
            total[static_cast<std::size_t>(a[i])] += w[i];
        }
    std::ranges::sort(order, [&](std::size_t l, std::size_t r) { return x(l, j) < x(r, j); });
    std::array<double, 2> cdf{};
    double gap = 0;
    for (std::size_t k = 0; k < order.size(); ++k) {
        const std::size_t i = order[k];
        cdf[static_cast<std::size_t>(a[i])] += w[i] / total[static_cast<std::size_t>(a[i])];
        if (k + 1 == order.size() || x(order[k + 1], j) != x(i, j)) // evaluate after all ties
            gap = std::max(gap, std::abs(cdf[1] - cdf[0]));
    }
    return gap;
}

/// @brief Full-sample treated standard deviations: sample SD, or sqrt(p(1-p)) for binary columns.
std::vector<double> treated_sd(const std::vector<int>& a, const Matrix& x) {
    const std::vector<double> ones(a.size(), 1.0);
    std::vector<double> sd(x.cols());
    for (std::size_t j = 0; j < x.cols(); ++j) {
        const auto [m, v] = weighted_moments(x, j, a, 1, ones);
        sd[j] = is_binary(x, j) ? std::sqrt(m * (1 - m)) : std::sqrt(v);
    }
    return sd;
}
} // namespace

std::vector<double> propensity_scores(const std::vector<int>& treatment, const linalg::DenseMatrix<double>& covariates) {
    (void)group_sizes(treatment);
    const std::size_t n = treatment.size();
    check_covariates(covariates, n);
    dstruct::DataFrame frame;
    frame.add_column("treated", std::vector<double>(treatment.begin(), treatment.end()));
    std::string formula = "treated ~ 1";
    for (std::size_t j = 0; j < covariates.cols(); ++j) {
        const auto name = "covariate" + std::to_string(j + 1);
        std::vector<double> column(n);
        for (std::size_t i = 0; i < n; ++i)
            column[i] = covariates(i, j);
        frame.add_column(name, column);
        formula += " + " + name;
    }
    const stats::GLM model(frame, formula, {.family = stats::GLMFamily::Binomial});
    if (!model.converged() || model.boundary())
        throw std::runtime_error("propensity_scores: logistic fit did not converge or separates the groups");
    return model.fitted_values();
}

CovariateBalance covariate_balance(const std::vector<int>& treatment, const linalg::DenseMatrix<double>& covariates,
                                   const std::vector<double>& weights, const std::vector<double>& reference_sd) {
    (void)group_sizes(treatment);
    const std::size_t n = treatment.size();
    check_covariates(covariates, n);
    if (!weights.empty() && weights.size() != n)
        throw std::invalid_argument("covariate_balance: weights need one entry per unit");
    if (!reference_sd.empty() && reference_sd.size() != covariates.cols())
        throw std::invalid_argument("covariate_balance: reference_sd needs one entry per covariate");
    const std::vector<double> w = weights.empty() ? std::vector<double>(n, 1.0) : weights;
    std::array<double, 2> total{};
    for (std::size_t i = 0; i < n; ++i) {
        if (!(w[i] >= 0) || !std::isfinite(w[i]))
            throw std::invalid_argument("covariate_balance: weights must be finite and nonnegative");
        total[static_cast<std::size_t>(treatment[i])] += w[i];
    }
    if (!(total[0] > 0 && total[1] > 0))
        throw std::invalid_argument("covariate_balance: each group needs positive total weight");
    const auto sd = reference_sd.empty() ? treated_sd(treatment, covariates) : reference_sd;
    CovariateBalance out;
    const double na = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t j = 0; j < covariates.cols(); ++j) {
        const auto [mt, vt] = weighted_moments(covariates, j, treatment, 1, w);
        const auto [mc, vc] = weighted_moments(covariates, j, treatment, 0, w);
        out.treated_mean.push_back(mt);
        out.control_mean.push_back(mc);
        out.standardized_difference.push_back(sd[j] > 0 ? (mt - mc) / sd[j] : na);
        out.variance_ratio.push_back(is_binary(covariates, j) ? na : vt / vc);
        out.ecdf_max.push_back(ecdf_distance(covariates, j, treatment, w));
    }
    return out;
}

PropensityMatch match_on_propensity(const std::vector<int>& treatment, const std::vector<double>& propensity,
                                    const linalg::DenseMatrix<double>& covariates, const PropensityMatchOptions& options) {
    (void)group_sizes(treatment);
    const std::size_t n = treatment.size();
    if (propensity.size() != n)
        throw std::invalid_argument("match_on_propensity: propensity needs one entry per unit");
    check_covariates(covariates, n);
    if (options.ratio == 0)
        throw std::invalid_argument("match_on_propensity: ratio must be positive");
    if (options.caliper && !(*options.caliper > 0))
        throw std::invalid_argument("match_on_propensity: caliper must be positive");
    std::vector<double> d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double p = propensity[i];
        if (!(p > 0 && p < 1))
            throw std::invalid_argument("match_on_propensity: propensities must lie strictly in (0, 1)");
        d[i] = options.logit_distance ? std::log(p / (1 - p)) : p;
    }
    PropensityMatch out;
    out.propensity = propensity;
    out.matches.assign(n, {});
    out.caliper_width = std::numeric_limits<double>::infinity();
    if (options.caliper) {
        const double mean = std::accumulate(d.begin(), d.end(), 0.0) / static_cast<double>(n);
        double ss = 0;
        for (const double v : d)
            ss += (v - mean) * (v - mean);
        out.caliper_width = *options.caliper * std::sqrt(ss / static_cast<double>(n - 1));
    }
    std::vector<std::size_t> treated, controls;
    for (std::size_t i = 0; i < n; ++i)
        (treatment[i] == 1 ? treated : controls).push_back(i);
    if (options.order == MatchOrder::largest)
        std::ranges::stable_sort(treated, [&](std::size_t l, std::size_t r) { return d[l] > d[r]; });
    else if (options.order == MatchOrder::smallest)
        std::ranges::stable_sort(treated, [&](std::size_t l, std::size_t r) { return d[l] < d[r]; });
    // Closest admissible control to treated unit t, skipping used controls and existing matches.
    std::vector<bool> used(n, false);
    const auto nearest = [&](std::size_t t) {
        std::size_t best = no_set;
        double best_distance = std::numeric_limits<double>::infinity();
        for (const std::size_t c : controls) {
            if ((!options.replace && used[c]) || std::ranges::find(out.matches[t], c) != out.matches[t].end())
                continue;
            const double dist = std::abs(d[t] - d[c]);
            // Exact ties follow MatchIt's sorted search: the earlier row above (or at) the treated
            // score and the later row below it.
            const bool later_wins = d[c] < d[t];
            if (dist <= out.caliper_width && (dist < best_distance || (dist == best_distance && later_wins))) {
                best_distance = dist;
                best = c;
            }
        }
        return best;
    };
    if (options.replace) {
        for (const std::size_t t : treated)
            for (std::size_t k = 0; k < options.ratio; ++k) {
                const std::size_t c = nearest(t);
                if (c == no_set)
                    break;
                out.matches[t].push_back(c);
            }
    } else {
        for (std::size_t round = 0; round < options.ratio; ++round)
            for (const std::size_t t : treated) {
                if (round > 0 && out.matches[t].size() < round)
                    continue; // unmatched in an earlier round
                const std::size_t c = nearest(t);
                if (c == no_set)
                    continue;
                out.matches[t].push_back(c);
                used[c] = true;
            }
    }
    out.weights.assign(n, 0.0);
    out.subclass.assign(n, no_set);
    for (const std::size_t t : treated) {
        const auto& m = out.matches[t];
        if (m.empty())
            continue;
        out.weights[t] = 1.0;
        ++out.matched_treated;
        if (!options.replace)
            out.subclass[t] = t;
        for (const std::size_t c : m) {
            out.weights[c] += 1.0 / static_cast<double>(m.size());
            if (!options.replace)
                out.subclass[c] = t;
        }
    }
    double control_total = 0;
    for (const std::size_t c : controls)
        if (out.weights[c] > 0) {
            ++out.matched_controls;
            control_total += out.weights[c];
        }
    if (out.matched_treated == 0)
        throw std::invalid_argument("match_on_propensity: no treated unit has a match within the caliper");
    for (const std::size_t c : controls)
        out.weights[c] *= static_cast<double>(out.matched_controls) / control_total;
    const auto sd = treated_sd(treatment, covariates);
    out.before = covariate_balance(treatment, covariates, {}, sd);
    out.after = covariate_balance(treatment, covariates, out.weights, sd);
    return out;
}

PropensityMatch propensity_score_match(const std::vector<int>& treatment, const linalg::DenseMatrix<double>& covariates,
                                       const PropensityMatchOptions& options) {
    return match_on_propensity(treatment, propensity_scores(treatment, covariates), covariates, options);
}

namespace detail {
MatchedEffect weighted_contrast(const std::vector<double>& outcome, const std::vector<int>& treatment,
                                const std::vector<double>& weights, const std::vector<std::size_t>& clusters,
                                bool small_sample, double confidence) {
    const std::size_t n = treatment.size();
    if (outcome.size() != n || weights.size() != n || (!clusters.empty() && clusters.size() != n))
        throw std::invalid_argument("effect: outcome, treatment, weights and clusters must describe the same units");
    if (!(confidence > 0 && confidence < 1))
        throw std::invalid_argument("effect: confidence must lie in (0, 1)");
    std::array<double, 2> sw{}, sy{};
    std::size_t used = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!(weights[i] >= 0) || !std::isfinite(weights[i]))
            throw std::invalid_argument("effect: weights must be finite and nonnegative");
        if (weights[i] == 0)
            continue;
        if (!std::isfinite(outcome[i]) || (treatment[i] != 0 && treatment[i] != 1))
            throw std::invalid_argument("effect: outcomes must be finite and treatment 0 or 1");
        const auto g = static_cast<std::size_t>(treatment[i]);
        sw[g] += weights[i];
        sy[g] += weights[i] * outcome[i];
        ++used;
    }
    if (used < 3 || sw[0] <= 0 || sw[1] <= 0)
        throw std::invalid_argument("effect: too few weighted units in a group");
    const std::array<double, 2> mean{sy[0] / sw[0], sy[1] / sw[1]};
    // Row (1) of (X'WX)^{-1} for the design (1, a): (-1/sw0, 1/sw0 + 1/sw1).
    const double b10 = -1 / sw[0], b11 = 1 / sw[0] + 1 / sw[1];
    // Score sums per cluster (or per unit), projected on that row: u = w e (b10 + b11 a).
    std::vector<double> score(n, 0.0);
    std::vector<bool> present(n, false);
    for (std::size_t i = 0; i < n; ++i) {
        if (weights[i] == 0)
            continue;
        const std::size_t g = clusters.empty() ? i : clusters[i];
        if (g >= n)
            throw std::invalid_argument("effect: weighted unit without a cluster");
        const double e = outcome[i] - mean[static_cast<std::size_t>(treatment[i])];
        score[g] += weights[i] * e * (b10 + b11 * treatment[i]);
        present[g] = true;
    }
    double meat = 0;
    std::size_t groups = 0;
    for (std::size_t g = 0; g < n; ++g)
        if (present[g]) {
            ++groups;
            meat += score[g] * score[g];
        }
    const double nu = static_cast<double>(used), k = 2.0, G = static_cast<double>(groups);
    double adjust = 1.0;
    if (small_sample)
        adjust = clusters.empty() ? nu / (nu - k) : G / (G - 1) * (nu - 1) / (nu - k);
    MatchedEffect out;
    out.estimate = mean[1] - mean[0];
    out.standard_error = std::sqrt(adjust * meat);
    out.df = nu - k;
    const double t = out.estimate / out.standard_error;
    out.p_value = 2 * (1 - random::student_t_cdf(std::abs(t), out.df));
    const double q = random::student_t_quantile(0.5 + confidence / 2, out.df);
    out.lower = out.estimate - q * out.standard_error;
    out.upper = out.estimate + q * out.standard_error;
    return out;
}
} // namespace detail

MatchedEffect matched_effect(const std::vector<double>& outcome, const std::vector<int>& treatment,
                             const PropensityMatch& match, double confidence) {
    if (match.subclass.size() != treatment.size())
        throw std::invalid_argument("matched_effect: outcome, treatment and match must describe the same units");
    const bool clustered = std::ranges::any_of(match.subclass, [](std::size_t g) { return g != no_set; });
    return detail::weighted_contrast(outcome, treatment, match.weights,
                                     clustered ? match.subclass : std::vector<std::size_t>{}, true, confidence);
}

} // namespace datamunge::causal
