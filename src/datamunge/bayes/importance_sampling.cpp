#include <datamunge/bayes/importance_sampling.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace datamunge::bayes {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
} // namespace

ImportanceSampling::ImportanceSampling(ImportanceSamplingOptions options) : options_(options) {}

ImportanceSamplingResult ImportanceSampling::sample(optim::ArbitraryFunction& log_target, const std::vector<double>& proposal_mean,
                                                     const std::vector<std::vector<double>>& proposal_covariance) const {
    const std::size_t d = proposal_mean.size();
    if (proposal_covariance.size() != d)
        throw std::invalid_argument("ImportanceSampling::sample: proposal_covariance must be d x d, matching proposal_mean");
    linalg::DenseMatrix<double> cov(d, d, 0.0);
    for (std::size_t i = 0; i < d; ++i) {
        if (proposal_covariance[i].size() != d)
            throw std::invalid_argument("ImportanceSampling::sample: proposal_covariance must be d x d, matching proposal_mean");
        for (std::size_t j = 0; j < d; ++j) cov(i, j) = proposal_covariance[i][j];
    }

    const auto chol = linalg::cholesky(cov);
    if (!chol.ok) throw std::invalid_argument("ImportanceSampling::sample: proposal_covariance is not positive definite");

    double log_det_cov = 0.0;
    for (std::size_t i = 0; i < d; ++i) log_det_cov += 2.0 * std::log(chol.L(i, i));
    const double log_norm_const = -0.5 * static_cast<double>(d) * std::log(kTwoPi) - 0.5 * log_det_cov;

    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    const std::size_t n = options_.num_samples;
    ImportanceSamplingResult result;
    result.samples.reserve(n);
    std::vector<double> log_weights(n);

    for (std::size_t s = 0; s < n; ++s) {
        std::vector<double> z(d);
        for (double& v : z) v = normal(rng);

        // x = mean + L*z (a draw from N(mean, covariance), since covariance = L*L').
        std::vector<double> x(d);
        for (std::size_t i = 0; i < d; ++i) {
            double v = proposal_mean[i];
            for (std::size_t k = 0; k <= i; ++k) v += chol.L(i, k) * z[k];
            x[i] = v;
        }

        // log N(x; mean, covariance) simplifies to log_norm_const - 0.5*z'z directly, since
        // z = L^-1(x - mean) by construction -- no extra linear solve needed per sample.
        double quad = 0.0;
        for (const double v : z) quad += v * v;
        const double log_proposal = log_norm_const - 0.5 * quad;

        log_weights[s] = log_target.evaluate(x) - log_proposal;
        result.samples.push_back(std::move(x));
    }

    double max_lw = -std::numeric_limits<double>::infinity();
    for (const double lw : log_weights) max_lw = std::max(max_lw, lw);

    std::vector<double> raw(n);
    double sum_exp = 0.0;
    for (std::size_t s = 0; s < n; ++s) {
        raw[s] = std::exp(log_weights[s] - max_lw);
        sum_exp += raw[s];
    }

    result.normalized_weights.resize(n);
    double sum_sq = 0.0;
    for (std::size_t s = 0; s < n; ++s) {
        result.normalized_weights[s] = raw[s] / sum_exp;
        sum_sq += result.normalized_weights[s] * result.normalized_weights[s];
    }
    result.effective_sample_size = 1.0 / sum_sq;
    result.log_evidence = max_lw + std::log(sum_exp) - std::log(static_cast<double>(n));

    return result;
}

} // namespace datamunge::bayes
