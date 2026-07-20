#include <datamunge/bayes/gibbs.hpp>

#include <datamunge/bayes/detail/dual_averaging.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::bayes {

GibbsSampler::GibbsSampler(GibbsOptions options) : options_(options) {}

GibbsResult GibbsSampler::sample(optim::ArbitraryFunction& log_posterior, const std::vector<double>& initial_params) const {
    const std::size_t d = initial_params.size();
    std::vector<double> step_sizes = options_.initial_step_sizes;
    if (step_sizes.size() == 1 && d > 1) step_sizes.assign(d, step_sizes[0]);
    if (step_sizes.size() != d)
        throw std::invalid_argument("GibbsSampler::sample: initial_step_sizes must have length 1 or match initial_params's length");

    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    std::vector<double> x = initial_params;
    double log_p_x = log_posterior.evaluate(x);

    std::vector<detail::DualAveraging> adaptations;
    adaptations.reserve(d);
    for (std::size_t j = 0; j < d; ++j) adaptations.emplace_back(step_sizes[j], options_.target_accept_rate);

    GibbsResult result;
    result.samples.reserve(options_.num_samples);
    std::vector<double> accept_counts(d, 0.0);
    const std::size_t total_iters = options_.num_warmup + options_.num_samples;

    for (std::size_t iter = 0; iter < total_iters; ++iter) {
        for (std::size_t j = 0; j < d; ++j) {
            std::vector<double> proposal = x;
            proposal[j] = x[j] + step_sizes[j] * normal(rng);
            const double log_p_proposal = log_posterior.evaluate(proposal);

            const double accept_prob = std::min(1.0, std::exp(log_p_proposal - log_p_x));
            if (unif01(rng) < accept_prob) {
                x = proposal;
                log_p_x = log_p_proposal;
                accept_counts[j] += 1.0;
            }

            if (iter < options_.num_warmup) {
                adaptations[j].update(accept_prob);
                step_sizes[j] = adaptations[j].step_size();
                if (iter + 1 == options_.num_warmup) step_sizes[j] = adaptations[j].finalized_step_size();
            }
        }
        if (iter >= options_.num_warmup) result.samples.push_back(x);
    }

    result.accept_rates.resize(d);
    for (std::size_t j = 0; j < d; ++j) result.accept_rates[j] = accept_counts[j] / static_cast<double>(total_iters);
    result.final_step_sizes = step_sizes;
    return result;
}

} // namespace datamunge::bayes
