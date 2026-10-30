/// @file rwm.cpp
/// @brief Random-walk Metropolis with dual-averaging and adaptive-Metropolis warmup.

#include <datamunge/bayes/rwm.hpp>

#include <datamunge/bayes/detail/dual_averaging.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::bayes {

/// @brief Covariance and Cholesky helpers for the adaptive proposal.
namespace {

/// @brief Sample covariance of draws[first, last), shrunk towards 1e-3 I as in Stan's metric
///        adaptation: (n / (n + 5)) S + 1e-3 (5 / (n + 5)) I.
std::vector<std::vector<double>> regularized_covariance(const std::vector<std::vector<double>>& draws) {
    const std::size_t n = draws.size(), d = draws.front().size();
    std::vector<double> mean(d, 0.0);
    for (const auto& x : draws)
        for (std::size_t j = 0; j < d; ++j) mean[j] += x[j] / static_cast<double>(n);
    std::vector<std::vector<double>> cov(d, std::vector<double>(d, 0.0));
    for (const auto& x : draws)
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = 0; j < d; ++j) cov[i][j] += (x[i] - mean[i]) * (x[j] - mean[j]);
    const double nn = static_cast<double>(n);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j) {
            cov[i][j] = cov[i][j] / (nn - 1.0) * nn / (nn + 5.0);
            if (i == j) cov[i][j] += 1e-3 * 5.0 / (nn + 5.0);
        }
    return cov;
}

/// @brief Lower-triangular Cholesky factor of a symmetric positive-definite matrix.
std::vector<std::vector<double>> cholesky_lower(const std::vector<std::vector<double>>& a) {
    const std::size_t d = a.size();
    std::vector<std::vector<double>> l(d, std::vector<double>(d, 0.0));
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = a[i][j];
            for (std::size_t k = 0; k < j; ++k) sum -= l[i][k] * l[j][k];
            if (i == j) {
                if (!(sum > 0.0)) throw std::runtime_error("RandomWalkMetropolis: adapted covariance is not positive definite");
                l[i][i] = std::sqrt(sum);
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    return l;
}

} // namespace

RandomWalkMetropolis::RandomWalkMetropolis(RWMOptions options) : options_(options) {}

RWMResult RandomWalkMetropolis::sample(optim::ArbitraryFunction& log_posterior, const std::vector<double>& initial_params) const {
    const std::size_t d = initial_params.size();
    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    std::vector<double> x = initial_params;
    double log_p_x = log_posterior.evaluate(x);

    detail::DualAveraging adaptation(options_.initial_step_size, options_.target_accept_rate);
    double step_size = options_.initial_step_size;

    RWMResult result;
    result.samples.reserve(options_.num_samples);
    // Adaptive Metropolis: isotropic until the switch point, then shaped by chol (L).
    const bool shaped = options_.adapt_covariance && options_.num_warmup >= 8;
    const std::size_t switch_at = options_.num_warmup / 2;
    std::vector<std::vector<double>> chol;
    std::vector<std::vector<double>> window;
    std::size_t accepted = 0;
    const std::size_t total_iters = options_.num_warmup + options_.num_samples;

    for (std::size_t iter = 0; iter < total_iters; ++iter) {
        if (shaped && iter == switch_at) {
            result.proposal_covariance = regularized_covariance(window);
            chol = cholesky_lower(result.proposal_covariance);
            step_size = 2.38 / std::sqrt(static_cast<double>(d));
            adaptation = detail::DualAveraging(step_size, options_.target_accept_rate);
        }
        std::vector<double> proposal(d);
        if (chol.empty()) {
            for (std::size_t j = 0; j < d; ++j) proposal[j] = x[j] + step_size * normal(rng);
        } else {
            std::vector<double> z(d);
            for (double& v : z) v = normal(rng);
            for (std::size_t i = 0; i < d; ++i) {
                double shift = 0.0;
                for (std::size_t k = 0; k <= i; ++k) shift += chol[i][k] * z[k];
                proposal[i] = x[i] + step_size * shift;
            }
        }
        const double log_p_proposal = log_posterior.evaluate(proposal);

        const double accept_prob = std::min(1.0, std::exp(log_p_proposal - log_p_x));
        if (unif01(rng) < accept_prob) {
            x = proposal;
            log_p_x = log_p_proposal;
            ++accepted;
        }

        if (shaped && iter >= switch_at / 2 && iter < switch_at) window.push_back(x);
        if (iter < options_.num_warmup) {
            adaptation.update(accept_prob);
            step_size = adaptation.step_size();
            if (iter + 1 == options_.num_warmup) step_size = adaptation.finalized_step_size();
        } else {
            result.samples.push_back(x);
        }
    }

    result.accept_rate = static_cast<double>(accepted) / static_cast<double>(total_iters);
    result.final_step_size = step_size;
    return result;
}

} // namespace datamunge::bayes
