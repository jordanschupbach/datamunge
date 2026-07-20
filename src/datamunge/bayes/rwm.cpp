#include <datamunge/bayes/rwm.hpp>

#include <datamunge/bayes/detail/dual_averaging.hpp>

#include <algorithm>
#include <cmath>
#include <random>

namespace datamunge::bayes {

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
    std::size_t accepted = 0;
    const std::size_t total_iters = options_.num_warmup + options_.num_samples;

    for (std::size_t iter = 0; iter < total_iters; ++iter) {
        std::vector<double> proposal(d);
        for (std::size_t j = 0; j < d; ++j) proposal[j] = x[j] + step_size * normal(rng);
        const double log_p_proposal = log_posterior.evaluate(proposal);

        const double accept_prob = std::min(1.0, std::exp(log_p_proposal - log_p_x));
        if (unif01(rng) < accept_prob) {
            x = proposal;
            log_p_x = log_p_proposal;
            ++accepted;
        }

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
