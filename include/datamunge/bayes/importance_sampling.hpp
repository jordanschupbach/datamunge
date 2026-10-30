#pragma once

/// @file importance_sampling.hpp
/// @brief Self-normalised importance sampling with a Gaussian proposal, and its asymptotic
///        efficiency on a Gaussian target.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::bayes {

/// Settings for ImportanceSampling.
struct ImportanceSamplingOptions {
    /// @brief Number of proposal draws.
    std::size_t num_samples{1000};
    /// @brief Random seed.
    std::uint64_t seed{42};
};

/// Output of ImportanceSampling::sample().
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
    /// @brief Stores the sampler settings.
    /// @param options Sampler settings.
    explicit ImportanceSampling(ImportanceSamplingOptions options = {});

    /// @brief @p log_target need only be evaluable (optim::ArbitraryFunction), not
    ///        differentiable. @p proposal_covariance (a plain d x d nested vector, not
    ///        linalg::DenseMatrix -- matrices never cross the SWIG boundary as DenseMatrix
    ///        anywhere in this library, only as nested vectors) must be symmetric positive
    ///        definite.
    ImportanceSamplingResult sample(optim::ArbitraryFunction& log_target, const std::vector<double>& proposal_mean,
                                     const std::vector<std::vector<double>>& proposal_covariance) const;

    /// @brief The options this ImportanceSampling was configured with.
    /// @return The stored options.
    [[nodiscard]] const ImportanceSamplingOptions& options() const noexcept { return options_; }

private:
    ImportanceSamplingOptions options_; ///< Sampler settings.
};

/// @brief Asymptotic effective-sample-size fraction \f$N_{\text{eff}}/N\f$ of importance sampling
///        a \f$d\f$-dimensional Gaussian target with a Gaussian proposal of the same mean and
///        covariance scaled by \f$k^2\f$.
///
/// \f$N_{\text{eff}}/N\to1/E_q[w^2]\f$ for weights \f$w=p/q\f$ normalised to mean 1. In one
/// dimension \f$E_q[w^2]=\int p^2/q=k^2/\sqrt{2k^2-1}\f$, finite only for \f$k^2>1/2\f$, and the
/// dimensions multiply, giving \f$\bigl(\sqrt{2k^2-1}/k^2\bigr)^d\f$. It is 1 at \f$k=1\f$ and
/// 0 (infinite weight variance) for \f$k\le1/\sqrt2\f$.
/// @param scale Proposal scale factor \f$k>0\f$.
/// @param dimension Dimension \f$d\ge1\f$.
/// @return The efficiency in [0, 1].
/// @throws std::invalid_argument for non-positive @p scale or @p dimension.
double gaussian_importance_efficiency(double scale, int dimension);

} // namespace datamunge::bayes
