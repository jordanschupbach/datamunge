#pragma once
/// @file posterior_collapse.hpp
/// @brief Explicit KL warmup and finite empirical-mixture posterior diagnostics.
#include "datamunge/nn/sequence_variational_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Return a source-rounded constant or linear zero-based KL warmup weight.
///
/// Constant mode returns1. Warmup returns min(1,float(step)/float(ramp_steps-1));
/// defaults ramp18, so steps0..17 include both endpoints and later steps stay1.
/// Computation intentionally uses Float32 rounding. No counters, model state,
/// mutation or RNG; O(1) work/storage and safe independent concurrent calls.
/// @param warmup True selects the linear ramp, false constant1.
/// @param step Nonnegative zero-based optimizer step, supplied by the caller.
/// @param ramp_steps Ramp length at least2, default18, validated even in constant mode.
/// @return Float32 beta in[0,1].
/// @throws std::invalid_argument For negative step or ramp_steps<2.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto first=dm::nn::models::CollapseBeta(true,0);  // 0
/// const auto last=dm::nn::models::CollapseBeta(true,17); // 1
/// @endcode
/// @par References
/// - Bowman et al. (2016), Generating Sentences from a Continuous Space,
///   https://arxiv.org/abs/1511.06349 . Uses the teaching experiment's18-step linear schedule.
/// @see AnnealedSequenceObjective(), SequenceVariationalAutoencoder()
inline float CollapseBeta(bool warmup, int step, int ramp_steps = 18) {
  if (step < 0 || ramp_steps < 2)
    throw std::invalid_argument("KL warmup requires step>=0 and ramp>=2");
  return warmup ? std::min(1.f, float(step) / float(ramp_steps - 1)) : 1.f;
}

/// @brief Own host-side analytic and sampled finite-mixture posterior diagnostics.
///
/// Values are doubles and independent of all tensor lifetimes, without autograd
/// paths or fitting state. Signed Monte Carlo estimates are retained. A small KL
/// or zero active-unit count alone is not a semantic/usefulness certificate.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto mu=n::zeros({8,4}), lv=n::zeros({8,4});
/// const auto s=n::models::CollapseStatistics(mu,lv,{n::ones({8,4})});
/// // Analytic KL0, active units0, empirical mixture information0.
/// @endcode
/// @see CollapseStatistics()
struct CollapseSummary {
  std::vector<double> unit_kl;     ///< Mean analytic KL for each of D coordinates, nats.
  std::vector<double> mu_variance; ///< Population (divide by N) variance of each posterior mean.
  double              analytic_kl = 0; ///< Sum of per-unit analytic KL, averaged over cases.
  double sampled_conditional_kl   = 0; ///< Mean log q_i(z_i)-log p(z_i) over all aligned draws.
  double mixture_information =
      0; ///< Mean log q_i(z_i)-log qbar(z_i), signed finite-sample estimate.
  double       aggregate_kl = 0; ///< Mean log qbar(z_i)-log p(z_i), signed finite-sample estimate.
  std::int64_t active_units =
      0; ///< Number of coordinates with population mean variance strictly above threshold.
};

/// @brief Diagnose diagonal Gaussian posteriors against their complete empirical mixture.
///
/// mu/log_variance are finite matching Float32/64[N,D], N>1,D>0. Each of S>0
/// samples is the same shape/dtype, with row i aligned to posterior i; the caller
/// supplies actual draws (their distribution is not verified). Analytic KL and
/// population variance of mu use all N cases. Active units have variance strictly
/// greater than active_threshold, default.01. All computations use host double
/// arithmetic, preserving the original source loop/reduction order for Float32.
/// qbar(z)=sum_k q_k(z)/N includes each sampled row's OWN posterior. Log densities
/// use a shifted log-sum-exp. The same samples supply conditional KL, information
/// and aggregate KL; their samplewise decomposition holds up to summation roundoff.
/// This empirical self-including mixture has finite-sample bias: these estimates
/// are not population mutual information, true aggregate KL, or semantic scores.
/// Sampled conditional KL need not equal analytic KL; negative estimates remain.
/// Strides accepted. No clipping, hidden draws, gradients or mutation. Outputs own
/// host vectors; independent calls are safe. Work O(S*N*N*D), temporary storage
/// O(N*D+N), without a stored pairwise tensor. Extreme finite values can overflow
/// or underflow Gaussian densities; nonfinite results throw instead of being repaired.
/// @param mu Finite posterior means[N,D], N>1,D>0.
/// @param log_variance Finite same-shape/dtype posterior log variances.
/// @param samples Nonempty collection of finite aligned draws[N,D], same dtype.
/// @param active_threshold Finite nonnegative variance threshold, default.01.
/// @return Owning per-unit and aggregate analytic/sampled diagnostics in nats.
/// @throws std::invalid_argument For invalid tensors, samples, products or threshold.
/// @throws std::overflow_error For nonfinite computed densities, moments or diagnostics.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto mu=n::zeros({8,4}),lv=n::zeros({8,4});
/// const auto out=n::models::CollapseStatistics(mu,lv,{n::zeros({8,4})});
/// // All four KL/information totals and all active-unit counts are exactly zero.
/// @endcode
/// @par References
/// - Hoffman and Johnson (2016), ELBO surgery: yet another way to carve up the
///   variational evidence lower bound,
///   https://approximateinference.org/archives/2016/accepted/HoffmanJohnson2016.pdf .
///   Uses an explicitly finite empirical mixture, not a population estimator.
/// @see CollapseSummary, DiagonalGaussianKl(), CollapseBeta()
inline CollapseSummary CollapseStatistics(
    const Tensor&              mu,
    const Tensor&              log_variance,
    const std::vector<Tensor>& samples,
    double                     active_threshold = .01) {
  gaussian_detail::Match(mu, log_variance);
  const auto n = mu.size(0), d = mu.size(1);
  if (n <= 1 || samples.empty() || !std::isfinite(active_threshold) || active_threshold < 0 ||
      samples.size() > std::size_t(std::numeric_limits<std::int64_t>::max() / n))
    throw std::invalid_argument(
        "posterior diagnostics require N>1, draws and a nonnegative finite threshold");
  for (const auto& draw : samples)
    gaussian_detail::Match(mu, draw);
  const auto means = mu.contiguous().to_vector<double>(),
             logs  = log_variance.contiguous().to_vector<double>();
  CollapseSummary out;
  out.unit_kl.assign(d, 0);
  out.mu_variance.assign(d, 0);
  for (std::int64_t j = 0; j < d; ++j) {
    double mean = 0;
    for (std::int64_t b = 0; b < n; ++b)
      mean += means[b * d + j];
    mean /= n;
    for (std::int64_t b = 0; b < n; ++b) {
      const double m = means[b * d + j], v = logs[b * d + j];
      out.unit_kl[j] += .5 * (m * m + std::exp(v) - 1 - v) / n;
      out.mu_variance[j] += (m - mean) * (m - mean) / n;
    }
    if (!std::isfinite(out.unit_kl[j]) || !std::isfinite(out.mu_variance[j]))
      throw std::overflow_error("posterior analytic moments overflowed");
    out.analytic_kl += out.unit_kl[j];
    out.active_units += out.mu_variance[j] > active_threshold;
  }
  for (const auto& draw : samples) {
    const auto          z = draw.contiguous().to_vector<double>();
    std::vector<double> component(n);
    for (std::int64_t i = 0; i < n; ++i) {
      double logp = 0;
      for (std::int64_t k = 0; k < n; ++k) {
        double logq = 0;
        for (std::int64_t j = 0; j < d; ++j) {
          const double diff = z[i * d + j] - means[k * d + j], variance = logs[k * d + j];
          logq += -.5 * (diff * diff * std::exp(-variance) + variance +
                         std::log(2 * 3.14159265358979323846));
        }
        if (!std::isfinite(logq))
          throw std::overflow_error("posterior component density overflowed");
        component[k] = logq;
      }
      const double own = component[i];
      for (std::int64_t j = 0; j < d; ++j) {
        const double v = z[i * d + j];
        logp += -.5 * (v * v + std::log(2 * 3.14159265358979323846));
      }
      const double maximum = *std::max_element(component.begin(), component.end());
      double       sum     = 0;
      for (double v : component)
        sum += std::exp(v - maximum);
      const double mixture = maximum + std::log(sum) - std::log(double(n));
      const double scale   = 1. / (n * samples.size());
      out.sampled_conditional_kl += (own - logp) * scale;
      out.mixture_information += (own - mixture) * scale;
      out.aggregate_kl += (mixture - logp) * scale;
    }
  }
  if (!std::isfinite(out.analytic_kl) || !std::isfinite(out.sampled_conditional_kl) ||
      !std::isfinite(out.mixture_information) || !std::isfinite(out.aggregate_kl))
    throw std::overflow_error("posterior mixture diagnostics overflowed");
  return out;
}
} // namespace datamunge::nn::models
