#include <datamunge/bayes/bayes.hpp>
#include <datamunge/optim/function_types.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using datamunge::bayes::GibbsOptions;
using datamunge::bayes::GibbsSampler;
using datamunge::bayes::ImportanceSampling;
using datamunge::bayes::ImportanceSamplingOptions;
using datamunge::bayes::RandomWalkMetropolis;
using datamunge::bayes::RWMOptions;
using datamunge::optim::ArbitraryFunction;

namespace {

// An unnormalized log-density for a correlated bivariate Gaussian, mean (2, -1), covariance
// [[1, 0.5], [0.5, 1]] -- deliberately unnormalized (no -0.5*log((2pi)^d |Sigma|) term) so
// ImportanceSampling's log_evidence estimate has a known analytic target to compare against.
class UnnormalizedGaussian : public ArbitraryFunction {
  public:
    double evaluate(const std::vector<double>& x) override {
        const double dx = x[0] - 2.0;
        const double dy = x[1] + 1.0;
        // Sigma^-1 = (1/0.75) * [[1, -0.5], [-0.5, 1]] for Sigma = [[1, 0.5], [0.5, 1]].
        const double quad = (dx * dx - dx * dy + dy * dy) / 0.75;
        return -0.5 * quad;
    }
};

double mean_of(const std::vector<std::vector<double>>& samples, std::size_t dim) {
    double s = 0.0;
    for (const auto& row : samples) s += row[dim];
    return s / static_cast<double>(samples.size());
}

double weighted_mean_of(const std::vector<std::vector<double>>& samples, const std::vector<double>& weights, std::size_t dim) {
    double s = 0.0;
    for (std::size_t i = 0; i < samples.size(); ++i) s += weights[i] * samples[i][dim];
    return s;
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Target: correlated bivariate Gaussian, true mean (2, -1), true log-normalizer ~1.6941\n\n";

    UnnormalizedGaussian target;

    std::cout << "=================== Random-Walk Metropolis ===================\n";
    RWMOptions rwm_options;
    rwm_options.num_samples = 4000;
    rwm_options.num_warmup = 1000;
    const auto rwm_result = RandomWalkMetropolis(rwm_options).sample(target, {0.0, 0.0});
    std::cout << "posterior mean estimate: (" << mean_of(rwm_result.samples, 0) << ", " << mean_of(rwm_result.samples, 1)
              << ") (true: (2, -1))\n";
    std::cout << "acceptance rate: " << rwm_result.accept_rate << " (target 0.234)\n\n";

    std::cout << "=================== Metropolis-within-Gibbs ===================\n";
    GibbsOptions gibbs_options;
    gibbs_options.num_samples = 4000;
    gibbs_options.num_warmup = 1000;
    const auto gibbs_result = GibbsSampler(gibbs_options).sample(target, {0.0, 0.0});
    std::cout << "posterior mean estimate: (" << mean_of(gibbs_result.samples, 0) << ", " << mean_of(gibbs_result.samples, 1)
              << ") (true: (2, -1))\n";
    std::cout << "per-coordinate acceptance rates: (" << gibbs_result.accept_rates[0] << ", " << gibbs_result.accept_rates[1]
              << ") (target 0.44)\n\n";

    std::cout << "=================== Importance Sampling ===================\n";
    ImportanceSamplingOptions is_options;
    is_options.num_samples = 5000;
    const std::vector<double> proposal_mean = {2.0, -1.0}; // centered exactly at the true mean
    const std::vector<std::vector<double>> proposal_covariance = {{1.5, 0.0}, {0.0, 1.5}}; // a bit wider than the target
    const auto is_result = ImportanceSampling(is_options).sample(target, proposal_mean, proposal_covariance);

    std::cout << "posterior mean estimate: (" << weighted_mean_of(is_result.samples, is_result.normalized_weights, 0) << ", "
              << weighted_mean_of(is_result.samples, is_result.normalized_weights, 1) << ") (true: (2, -1))\n";
    std::cout << "effective sample size: " << is_result.effective_sample_size << " / " << is_options.num_samples << "\n";
    std::cout << "estimated log-normalizer: " << is_result.log_evidence << " (true: ~1.6941)\n";

    return 0;
}
