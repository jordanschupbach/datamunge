/// @file normal_models.cpp
/// @brief Exact posteriors of Normal mean and hierarchical Normal models.

#include <datamunge/bayes/normal_models.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace datamunge::bayes {

/// @brief Conjugate updates shared by the Normal models.
namespace {

/// @brief Throws unless y and sigma are non-empty, equal-length and sigma is positive.
void check_data(const std::vector<double>& y, const std::vector<double>& sigma, const char* who) {
    if (y.empty() || y.size() != sigma.size())
        throw std::invalid_argument(std::string(who) + ": y and sigma must be non-empty and the same length");
    for (const double s : sigma)
        if (!(s > 0.0)) throw std::invalid_argument(std::string(who) + ": sigma must be positive");
}

/// @brief Total variances sigma_j^2 + tau^2.
std::vector<double> total_variances(const std::vector<double>& sigma, const double tau) {
    std::vector<double> v;
    v.reserve(sigma.size());
    for (const double s : sigma) v.push_back(s * s + tau * tau);
    return v;
}

/// @brief Posterior of mu given tau, from total variances V_j.
NormalPosterior mu_given_variances(const std::vector<double>& y, const std::vector<double>& v, const double mu0,
                                   const double mu_sd) {
    double precision = 1.0 / (mu_sd * mu_sd);
    double weighted = mu0 * precision;
    for (std::size_t j = 0; j < y.size(); ++j) {
        precision += 1.0 / v[j];
        weighted += y[j] / v[j];
    }
    return {weighted / precision, std::sqrt(1.0 / precision)};
}

/// @brief log p(y | tau) from the conjugate update of mu.
double log_marginal(const std::vector<double>& y, const std::vector<double>& v, const double mu0,
                    const double mu_sd, const NormalPosterior& post) {
    double out = std::log(post.sd / mu_sd) - 0.5 * (mu0 * mu0 / (mu_sd * mu_sd) - post.mean * post.mean / (post.sd * post.sd));
    for (std::size_t j = 0; j < y.size(); ++j)
        out -= 0.5 * std::log(2.0 * std::numbers::pi * v[j]) + 0.5 * y[j] * y[j] / v[j];
    return out;
}

} // namespace

NormalPosterior normal_mean_posterior(const std::vector<double>& y, const std::vector<double>& sigma,
                                      const double mu0, const double tau0) {
    check_data(y, sigma, "normal_mean_posterior");
    if (!(tau0 > 0.0)) throw std::invalid_argument("normal_mean_posterior: tau0 must be positive");
    double precision = std::isinf(tau0) ? 0.0 : 1.0 / (tau0 * tau0);
    double weighted = std::isinf(tau0) ? 0.0 : mu0 * precision;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double w = 1.0 / (sigma[i] * sigma[i]);
        precision += w;
        weighted += w * y[i];
    }
    return {weighted / precision, std::sqrt(1.0 / precision)};
}

double hierarchical_normal_log_marginal(const std::vector<double>& y, const std::vector<double>& sigma,
                                        const double mu0, const double mu_sd, const double tau) {
    check_data(y, sigma, "hierarchical_normal_log_marginal");
    if (!(mu_sd > 0.0) || std::isinf(mu_sd) || !(tau >= 0.0))
        throw std::invalid_argument("hierarchical_normal_log_marginal: need finite mu_sd > 0 and tau >= 0");
    const auto v = total_variances(sigma, tau);
    return log_marginal(y, v, mu0, mu_sd, mu_given_variances(y, v, mu0, mu_sd));
}

HierarchicalNormalPosterior hierarchical_normal_posterior(const std::vector<double>& y,
                                                          const std::vector<double>& sigma, const double mu0,
                                                          const double mu_sd, const double tau_scale,
                                                          const std::size_t grid_points) {
    check_data(y, sigma, "hierarchical_normal_posterior");
    if (!(mu_sd > 0.0) || std::isinf(mu_sd) || !(tau_scale > 0.0) || grid_points < 10)
        throw std::invalid_argument("hierarchical_normal_posterior: need finite mu_sd > 0, tau_scale > 0, grid_points >= 10");
    const std::size_t n = grid_points, J = y.size();
    HierarchicalNormalPosterior out;
    std::vector<double> logw(n), mu_m(n), mu_s(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double u = (static_cast<double>(i) + 0.5) / static_cast<double>(n);
        const double tau = tau_scale * std::tan(std::numbers::pi * u / 2.0);
        const auto v = total_variances(sigma, tau);
        const auto post = mu_given_variances(y, v, mu0, mu_sd);
        out.tau.push_back(tau);
        logw[i] = log_marginal(y, v, mu0, mu_sd, post);
        mu_m[i] = post.mean;
        mu_s[i] = post.sd;
    }
    const double top = *std::max_element(logw.begin(), logw.end());
    std::vector<double> w(n);
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) total += (w[i] = std::exp(logw[i] - top));
    for (double& wi : w) wi /= total;
    out.log_evidence = top + std::log(total / static_cast<double>(n));

    // Density in tau: p(tau|y) = p(y|tau) p(tau) / p(y).
    for (std::size_t i = 0; i < n; ++i) {
        const double z = out.tau[i] / tau_scale;
        const double prior = 2.0 / (std::numbers::pi * tau_scale * (1.0 + z * z));
        out.tau_density.push_back(std::exp(logw[i] - out.log_evidence) * prior);
    }

    double m1 = 0.0, m2 = 0.0, t1 = 0.0, t2 = 0.0;
    std::vector<double> th1(J, 0.0), th2(J, 0.0), shrink(J, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double tau = out.tau[i];
        m1 += w[i] * mu_m[i];
        m2 += w[i] * (mu_s[i] * mu_s[i] + mu_m[i] * mu_m[i]);
        t1 += w[i] * tau;
        t2 += w[i] * tau * tau;
        for (std::size_t j = 0; j < J; ++j) {
            const double b = sigma[j] * sigma[j] / (sigma[j] * sigma[j] + tau * tau);
            const double mean = b * mu_m[i] + (1.0 - b) * y[j];
            const double var = (1.0 - b) * sigma[j] * sigma[j] + b * b * mu_s[i] * mu_s[i];
            th1[j] += w[i] * mean;
            th2[j] += w[i] * (var + mean * mean);
            shrink[j] += w[i] * b;
        }
    }
    out.mu_mean = m1;
    out.mu_sd = std::sqrt(std::max(0.0, m2 - m1 * m1));
    out.tau_mean = t1;
    out.tau_sd = std::sqrt(std::max(0.0, t2 - t1 * t1));
    for (std::size_t j = 0; j < J; ++j) {
        out.theta_mean.push_back(th1[j]);
        out.theta_sd.push_back(std::sqrt(std::max(0.0, th2[j] - th1[j] * th1[j])));
    }
    out.shrinkage = shrink;

    // Quantiles from the cumulative weights (each node carries mass w_i over its cell).
    const auto quantile = [&](const double p) {
        double cum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (cum + w[i] >= p) {
                const double lo = i == 0 ? 0.0 : 0.5 * (out.tau[i - 1] + out.tau[i]);
                const double hi = i + 1 < n ? 0.5 * (out.tau[i] + out.tau[i + 1]) : out.tau[i];
                return lo + (hi - lo) * (p - cum) / w[i];
            }
            cum += w[i];
        }
        return out.tau.back();
    };
    out.tau_q05 = quantile(0.05);
    out.tau_q50 = quantile(0.50);
    out.tau_q95 = quantile(0.95);
    return out;
}

} // namespace datamunge::bayes
