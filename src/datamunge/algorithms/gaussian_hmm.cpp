/// @file gaussian_hmm.cpp
/// @brief Gaussian-emission HMM fitting and decoding, and discrete HMM sampling.

#include <datamunge/algorithms/gaussian_hmm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>

namespace datamunge::algorithms {

/// @brief Emission densities and the scaled forward-backward pass.
namespace {

/// @brief Normal density.
double normal_pdf(double x, double mu, double sd) {
    const double z = (x - mu) / sd;
    return std::exp(-0.5 * z * z) / (sd * std::sqrt(2.0 * std::numbers::pi));
}

/// @brief Throws unless the model is K-state consistent.
void check_model(const GaussianHmm& m) {
    const std::size_t K = m.initial.size();
    if (K == 0 || m.transition.size() != K || m.mean.size() != K || m.sd.size() != K)
        throw std::invalid_argument("GaussianHmm: inconsistent sizes");
    for (const auto& row : m.transition)
        if (row.size() != K) throw std::invalid_argument("GaussianHmm: transition must be K x K");
}

/// @brief Scaled forward-backward quantities.
struct Pass {
    std::vector<std::vector<double>> alpha;  ///< normalised forward variables.
    std::vector<std::vector<double>> beta;   ///< scaled backward variables.
    std::vector<std::vector<double>> emit;   ///< emission densities b_i(x_t).
    std::vector<double>              scale;  ///< per-step normalisers (sum of unnormalised alpha).
    double                           log_likelihood = 0.0;  ///< sum of log scale.
};

/// @brief Runs the scaled forward-backward recursions.
Pass forward_backward_pass(const GaussianHmm& m, const std::vector<double>& x) {
    const std::size_t K = m.initial.size(), T = x.size();
    Pass p;
    p.alpha.assign(T, std::vector<double>(K));
    p.beta.assign(T, std::vector<double>(K));
    p.emit.assign(T, std::vector<double>(K));
    p.scale.assign(T, 0.0);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t i = 0; i < K; ++i) p.emit[t][i] = std::max(normal_pdf(x[t], m.mean[i], m.sd[i]), 1e-300);
    for (std::size_t t = 0; t < T; ++t) {
        double c = 0.0;
        for (std::size_t j = 0; j < K; ++j) {
            double s = 0.0;
            if (t == 0) s = m.initial[j];
            else
                for (std::size_t i = 0; i < K; ++i) s += p.alpha[t - 1][i] * m.transition[i][j];
            p.alpha[t][j] = s * p.emit[t][j];
            c += p.alpha[t][j];
        }
        for (auto& a : p.alpha[t]) a /= c;
        p.scale[t] = c;
        p.log_likelihood += std::log(c);
    }
    for (std::size_t i = 0; i < K; ++i) p.beta[T - 1][i] = 1.0;
    for (std::size_t t = T - 1; t-- > 0;)
        for (std::size_t i = 0; i < K; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < K; ++j) s += m.transition[i][j] * p.emit[t + 1][j] * p.beta[t + 1][j];
            p.beta[t][i] = s / p.scale[t + 1];
        }
    return p;
}

}  // namespace

GaussianHmm gaussian_hmm_initial(const std::vector<double>& x, std::size_t K) {
    if (x.empty() || K == 0) throw std::invalid_argument("gaussian_hmm_initial: need data and K > 0");
    std::vector<double> sorted = x;
    std::sort(sorted.begin(), sorted.end());
    double mean = 0.0, var = 0.0;
    for (double v : x) mean += v / static_cast<double>(x.size());
    for (double v : x) var += (v - mean) * (v - mean) / static_cast<double>(x.size());
    GaussianHmm m;
    m.initial.assign(K, 1.0 / static_cast<double>(K));
    m.transition.assign(K, std::vector<double>(K, K > 1 ? 0.2 / static_cast<double>(K - 1) : 1.0));
    for (std::size_t i = 0; i < K; ++i) {
        if (K > 1) m.transition[i][i] = 0.8;
        const auto idx = static_cast<std::size_t>((static_cast<double>(i) + 0.5) / static_cast<double>(K) * static_cast<double>(x.size()));
        m.mean.push_back(sorted[std::min(idx, x.size() - 1)]);
        m.sd.push_back(std::max(std::sqrt(var) / static_cast<double>(K), 1e-3));
    }
    return m;
}

std::vector<std::vector<double>> gaussian_hmm_posteriors(const GaussianHmm& model, const std::vector<double>& x,
                                                         double* log_likelihood) {
    check_model(model);
    if (x.empty()) throw std::invalid_argument("gaussian_hmm_posteriors: empty series");
    const Pass p = forward_backward_pass(model, x);
    std::vector<std::vector<double>> g(x.size(), std::vector<double>(model.initial.size()));
    for (std::size_t t = 0; t < x.size(); ++t) {
        double s = 0.0;
        for (std::size_t i = 0; i < g[t].size(); ++i) s += g[t][i] = p.alpha[t][i] * p.beta[t][i];
        for (auto& v : g[t]) v /= s;
    }
    if (log_likelihood) *log_likelihood = p.log_likelihood;
    return g;
}

GaussianHmmFit baum_welch_gaussian(const std::vector<double>& x, GaussianHmm initial, const GaussianHmmOptions& options) {
    check_model(initial);
    if (x.size() < 2) throw std::invalid_argument("baum_welch_gaussian: need at least two observations");
    const std::size_t K = initial.initial.size(), T = x.size();
    GaussianHmmFit fit;
    fit.model      = std::move(initial);
    fit.parameters = options.independent ? (K - 1) + 2 * K : (K - 1) + K * (K - 1) + 2 * K;
    double previous = -std::numeric_limits<double>::infinity();
    for (std::size_t iter = 0; iter < options.max_iterations; ++iter) {
        const Pass p = forward_backward_pass(fit.model, x);
        fit.log_likelihood_history.push_back(p.log_likelihood);
        fit.iterations = iter + 1;
        // gamma and xi
        std::vector<std::vector<double>> gamma(T, std::vector<double>(K));
        for (std::size_t t = 0; t < T; ++t) {
            double s = 0.0;
            for (std::size_t i = 0; i < K; ++i) s += gamma[t][i] = p.alpha[t][i] * p.beta[t][i];
            for (auto& v : gamma[t]) v /= s;
        }
        std::vector<std::vector<double>> xi_sum(K, std::vector<double>(K, 0.0));
        for (std::size_t t = 0; t + 1 < T; ++t) {
            double s = 0.0;
            std::vector<std::vector<double>> xi(K, std::vector<double>(K));
            for (std::size_t i = 0; i < K; ++i)
                for (std::size_t j = 0; j < K; ++j)
                    s += xi[i][j] = p.alpha[t][i] * fit.model.transition[i][j] * p.emit[t + 1][j] * p.beta[t + 1][j];
            for (std::size_t i = 0; i < K; ++i)
                for (std::size_t j = 0; j < K; ++j) xi_sum[i][j] += xi[i][j] / s;
        }
        // M-step
        GaussianHmm& m = fit.model;
        std::vector<double> occupancy(K, 0.0), occupancy_all(K, 0.0);
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t i = 0; i < K; ++i) {
                occupancy_all[i] += gamma[t][i];
                if (t + 1 < T) occupancy[i] += gamma[t][i];
            }
        if (options.independent) {
            for (std::size_t i = 0; i < K; ++i) {
                m.initial[i] = occupancy_all[i] / static_cast<double>(T);
                for (std::size_t r = 0; r < K; ++r) m.transition[r][i] = m.initial[i];
            }
        } else {
            m.initial = gamma[0];
            for (std::size_t i = 0; i < K; ++i)
                for (std::size_t j = 0; j < K; ++j) m.transition[i][j] = occupancy[i] > 0 ? xi_sum[i][j] / occupancy[i] : 1.0 / static_cast<double>(K);
        }
        for (std::size_t i = 0; i < K; ++i) {
            double mu = 0.0, var = 0.0;
            for (std::size_t t = 0; t < T; ++t) mu += gamma[t][i] * x[t];
            mu /= occupancy_all[i];
            for (std::size_t t = 0; t < T; ++t) var += gamma[t][i] * (x[t] - mu) * (x[t] - mu);
            m.mean[i] = mu;
            m.sd[i]   = std::max(std::sqrt(var / occupancy_all[i]), options.min_sd);
        }
        if (iter > 0 && p.log_likelihood - previous < options.tolerance) break;
        previous = p.log_likelihood;
    }
    gaussian_hmm_posteriors(fit.model, x, &fit.log_likelihood);
    return fit;
}

std::vector<std::size_t> gaussian_hmm_viterbi(const GaussianHmm& model, const std::vector<double>& x) {
    check_model(model);
    const std::size_t K = model.initial.size(), T = x.size();
    if (T == 0) return {};
    auto lg = [](double v) { return v > 0 ? std::log(v) : -std::numeric_limits<double>::infinity(); };
    auto le = [&](std::size_t t, std::size_t i) { return std::log(std::max(normal_pdf(x[t], model.mean[i], model.sd[i]), 1e-300)); };
    std::vector<std::vector<double>>      delta(T, std::vector<double>(K));
    std::vector<std::vector<std::size_t>> back(T, std::vector<std::size_t>(K, 0));
    for (std::size_t i = 0; i < K; ++i) delta[0][i] = lg(model.initial[i]) + le(0, i);
    for (std::size_t t = 1; t < T; ++t)
        for (std::size_t j = 0; j < K; ++j) {
            double best = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < K; ++i) {
                const double v = delta[t - 1][i] + lg(model.transition[i][j]);
                if (v > best) {
                    best       = v;
                    back[t][j] = i;
                }
            }
            delta[t][j] = best + le(t, j);
        }
    std::vector<std::size_t> path(T);
    path[T - 1] = static_cast<std::size_t>(std::max_element(delta[T - 1].begin(), delta[T - 1].end()) - delta[T - 1].begin());
    for (std::size_t t = T - 1; t > 0; --t) path[t - 1] = back[t][path[t]];
    return path;
}

HmmSample sample_hmm(const HiddenMarkovModel& model, std::size_t T, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    auto draw = [&](const std::vector<double>& p) {
        std::discrete_distribution<std::size_t> d(p.begin(), p.end());
        return d(rng);
    };
    HmmSample s;
    if (T == 0) return s;
    std::size_t q = draw(model.initial);
    for (std::size_t t = 0; t < T; ++t) {
        if (t > 0) q = draw(model.transition[q]);
        s.states.push_back(q);
        s.observations.push_back(draw(model.emission[q]));
    }
    return s;
}

}  // namespace datamunge::algorithms
