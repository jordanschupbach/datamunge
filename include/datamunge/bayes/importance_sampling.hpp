#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::bayes {

struct ImportanceSamplingOptions {
    std::size_t num_samples{1000};
    std::uint64_t seed{42};
};

struct ImportanceSamplingResult {
    /// @brief Draws from the N(proposal_mean, proposal_covariance) proposal (not the target).
    std::vector<std::vector<double>> samples;
    /// @brief Self-normalized importance weights (sum to 1), in samples order.
    std::vector<double> normalized_weights;
    /// @brief 1 / sum(normalized_weight_i^2) -- how many "effective" independent draws the
    ///        weighted sample is worth. Low relative to num_samples signals a poor proposal
    ///        (too narrow, wrong location, or lighter-tailed than the target).
    double effective_sample_size{0.0};
    /// @brief log-mean-exp(log_target(x_i) - log_proposal(x_i)) -- the standard importance-
    ///        sampling estimator of log p(y) (the target's normalizing constant), directly
    ///        comparable to e.g. INLAMixedModel::log_marginal_likelihood().
    double log_evidence{0.0};
};

/// @brief Importance sampling against a multivariate Gaussian proposal N(mean, covariance) --
///        a natural choice when a Laplace approximation is available and reasonably close to
///        the target's shape (e.g. bayes::MAP's mode paired with a numerically estimated
///        Hessian there). Produces a weighted i.i.d. sample in one shot rather than a Markov
///        chain -- no warmup, nothing to check for mixing -- but accuracy depends entirely on
///        how well the Gaussian proposal covers the target's mass; effective_sample_size is
///        the diagnostic to check before trusting the result.
class ImportanceSampling {
public:
    explicit ImportanceSampling(ImportanceSamplingOptions options = {});

    /// @brief @p log_target need only be evaluable (optim::ArbitraryFunction), not
    ///        differentiable. @p proposal_covariance (a plain d x d nested vector, not
    ///        linalg::DenseMatrix -- matrices never cross the SWIG boundary as DenseMatrix
    ///        anywhere in this library, only as nested vectors) must be symmetric positive
    ///        definite.
    ImportanceSamplingResult sample(optim::ArbitraryFunction& log_target, const std::vector<double>& proposal_mean,
                                     const std::vector<std::vector<double>>& proposal_covariance) const;

private:
    ImportanceSamplingOptions options_;
};

} // namespace datamunge::bayes
