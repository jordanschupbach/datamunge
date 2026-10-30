/// @file hmc.cpp
/// @brief Hamiltonian Monte Carlo with dual-averaging warmup, and the exact Gaussian leapfrog
///        acceptance rate.

#include <datamunge/bayes/hmc.hpp>

#include <datamunge/bayes/detail/dual_averaging.hpp>
#include <datamunge/bayes/detail/leapfrog.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>

namespace datamunge::bayes {

HMC::HMC(HMCOptions options) : options_(options) {}

HMCResult HMC::sample(optim::DifferentiableFunction& log_posterior, const std::vector<double>& initial_params) const {
    const std::size_t d = initial_params.size();
    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    std::vector<double> q = initial_params;
    detail::DualAveraging adaptation(options_.initial_step_size, options_.target_accept_rate);
    double step_size = options_.initial_step_size;

    HMCResult result;
    result.samples.reserve(options_.num_samples);
    std::size_t accepted = 0;
    const std::size_t total_iters = options_.num_warmup + options_.num_samples;

    for (std::size_t iter = 0; iter < total_iters; ++iter) {
        std::vector<double> p0(d);
        for (double& v : p0) v = normal(rng);

        const double h0 = detail::hamiltonian(log_posterior, q, p0);
        detail::LeapfrogState state{q, p0};
        for (std::size_t step = 0; step < options_.num_leapfrog_steps; ++step)
            state = detail::leapfrog_step(log_posterior, state, step_size);
        const double h1 = detail::hamiltonian(log_posterior, state.q, state.p);

        const double accept_prob = std::min(1.0, std::exp(h0 - h1));
        if (unif01(rng) < accept_prob) {
            q = state.q;
            ++accepted;
        }

        if (iter < options_.num_warmup) {
            adaptation.update(accept_prob);
            step_size = adaptation.step_size();
            if (iter + 1 == options_.num_warmup) step_size = adaptation.finalized_step_size();
        } else {
            result.samples.push_back(q);
        }
    }

    result.accept_rate = static_cast<double>(accepted) / static_cast<double>(total_iters);
    result.final_step_size = step_size;
    return result;
}

double leapfrog_gaussian_acceptance(const double step_size, const std::size_t num_steps, const double sd,
                                   const std::size_t angles) {
    if (!(step_size > 0.0) || num_steps == 0 || !(sd > 0.0) || angles < 8)
        throw std::invalid_argument("leapfrog_gaussian_acceptance: arguments must be positive (angles >= 8)");
    const double h = step_size / sd;
    double total = 0.0;
    for (std::size_t k = 0; k < angles; ++k) {
        const double phi = 2.0 * std::numbers::pi * (static_cast<double>(k) + 0.5) / static_cast<double>(angles);
        double q = std::cos(phi), p = std::sin(phi);
        for (std::size_t l = 0; l < num_steps; ++l) {
            p -= 0.5 * h * q;
            q += h * p;
            p -= 0.5 * h * q;
        }
        const double f = 0.5 * (q * q + p * p - 1.0);
        total += f > 0.0 ? 1.0 / (1.0 + 2.0 * f) : 1.0;
    }
    return total / static_cast<double>(angles);
}

} // namespace datamunge::bayes
