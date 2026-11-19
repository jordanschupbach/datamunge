#pragma once

/// @file survival_losses.hpp
/// @brief Differentiable objectives for right-censored survival models.

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Mean negative Cox partial log likelihood with Breslow ties.
///
/// Uses the inclusive risk set at each event time. Normalizes by the
/// number of events, returning zero when all observations are censored.
/// The explicit risk-set matrix costs O(N²) space and time.
/// @param risk Predicted log risk, shape [N], N>0.
/// @param time Finite observed event or censoring times, shape [N].
/// @param event Binary event indicators, shape [N].
/// @return Differentiable scalar loss.
/// @throws std::invalid_argument Invalid tensor geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto loss = n::cox_partial_nll(n::zeros({2}), n::tensor({1.,2.}), n::tensor({1.,0.}));
/// // log(2)
/// @endcode
/// @par References
/// - Cox (1972), Regression Models and Life-Tables. doi:10.1111/j.2517-6161.1972.tb00899.x
inline Tensor cox_partial_nll(const Tensor& risk, const Tensor& time, const Tensor& event) {
  if (!risk.defined() || risk.dim() != 1 || risk.size(0) == 0 ||
      !time.defined() || !event.defined() || time.shape() != risk.shape() || event.shape() != risk.shape())
    throw std::invalid_argument("Cox objective expects matching nonempty vectors");
  const auto eligible = greater_equal(expand_dims(time,0),expand_dims(time,1));
  const auto masked = where(eligible,broadcast_to(expand_dims(risk,0),{risk.size(0),risk.size(0)}),
                           -std::numeric_limits<double>::infinity());
  const auto observed = event.to(risk.dtype());
  const auto count = sum(observed);
  return sum(observed*(logsumexp(masked,{1})-risk))/maximum(count,ones_like(count));
}

/// @brief Likelihood and ranking contributions to a discrete competing-risk loss.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto result = n::deep_hit_loss(n::full({2,2},.5),n::tensor({1.,2.}),n::tensor({1,0},{2},n::DType::Int64),1,1);
/// auto total = result.nll + .2 * result.ranking;
/// @endcode
/// @see deep_hit_loss()
struct DeepHitLoss {
  Tensor nll; ///< Mean negative log probability of the observed event or censoring set.
  Tensor ranking; ///< Mean exponential ranking penalty over comparable pairs.
};

/// @brief Compute DeepHit likelihood and cause-specific ranking loss.
///
/// Each event uses its cause and discretized observed time. A censored row
/// uses probability strictly after its bin, including the beyond-horizon bin.
/// Ranking evaluates a subject's event cause and bin against every subject
/// with strictly later observation time. Empty comparable sets contribute zero.
/// @param mass Normalized joint probabilities [N,causes*bins+1], cause-major, tail last.
/// @param time Nonnegative finite observation times [N], floored to unit bins and clipped at bins-1.
/// @param cause Integer labels [N], zero for censoring and 1..causes for events.
/// @param causes Positive count of competing event causes.
/// @param bins Positive count of event bins per cause.
/// @return Live loss components, with ranking temperature 0.25.
/// @throws std::invalid_argument Invalid geometry or labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto result = n::deep_hit_loss(n::full({1,2},.5), n::zeros({1}), n::tensor({0},{1},n::DType::Int64),1,1);
/// // Negative log likelihood is log(2), ranking is zero.
/// @endcode
/// @par References
/// - Lee et al. (2018), DeepHit: A Deep Learning Approach to Survival Analysis
///   With Competing Risks. https://doi.org/10.1609/aaai.v32i1.11842
/// @see DeepHitLoss, cox_partial_nll()
inline DeepHitLoss deep_hit_loss(const Tensor& mass, const Tensor& time, const Tensor& cause,
                                  int causes, int bins) {
  if (!mass.defined() || mass.dim() != 2 || mass.size(0) == 0 || causes <= 0 || bins <= 0 ||
      mass.size(1) != std::int64_t(causes)*bins+1 || !time.defined() || !cause.defined() ||
      time.shape() != Shape{mass.size(0)} || cause.shape() != time.shape() ||
      (cause.dtype() != DType::Int32 && cause.dtype() != DType::Int64) ||
      !all(logical_and(greater_equal(cause,0.),less_equal(cause,causes))).item<bool>())
    throw std::invalid_argument("invalid DeepHit mass, time or cause geometry");
  const auto bin = expand_dims(clip(time,0.,bins-1.).to(DType::Int64),1);
  const auto grid = arange(0,bins,1,DType::Int64);
  const auto observed = greater(cause,0.);
  const auto later = greater(grid,bin);
  std::vector<Tensor> likelihood, prefix;
  for (int k = 1; k <= causes; ++k) {
    const auto same = expand_dims(equal(cause,k),1);
    likelihood.push_back(where(expand_dims(observed,1),logical_and(same,equal(grid,bin)),later));
    prefix.push_back(logical_and(same,less_equal(grid,bin)));
  }
  likelihood.push_back(expand_dims(equal(cause,0.),1));
  prefix.push_back(expand_dims(zeros_like(observed),1));
  const auto probability = sum(mass * concat(likelihood,1).to(mass.dtype()),{1});
  const auto nll = -mean(log(maximum(probability,full_like(probability,1e-12))));
  const auto mask = concat(prefix,1).to(mass.dtype());
  const auto relative = matmul(mask,mass,false,true)-sum(mask*mass,{1},true);
  const auto comparable = logical_and(expand_dims(observed,1),
      greater(expand_dims(time,0),expand_dims(time,1))).to(mass.dtype());
  const auto count = sum(comparable);
  const auto ranking = sum(comparable * exp(relative/.25))/maximum(count,ones_like(count));
  return {nll,ranking};
}
}  // namespace datamunge::nn
