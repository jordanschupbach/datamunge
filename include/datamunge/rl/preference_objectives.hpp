/// @file
/// @brief Sequence scores and direct preference optimization.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/rl/mappo.hpp"

namespace datamunge::rl {

/// @brief Normalize bigram counts with symmetric additive smoothing.
/// @param counts Finite nonnegative CPU Float32 counts [rows,vocabulary].
/// @param pseudocount Finite positive count added to each entry, default 0.5.
/// @return Row-normalized Float32 log probabilities. Empty count rows become uniform.
/// @throws std::invalid_argument Invalid counts, geometry or smoothing.
/// @par Example
/// @code
/// const auto logs=dm::rl::smoothed_bigram_logp(dm::nn::zeros({3,2}));
/// std::print("{:.3f}\n",logs.to_vector<float>()[0]); // -0.693
/// @endcode
/// @see bigram_completion_logp()
inline nn::Tensor smoothed_bigram_logp(const nn::Tensor& counts,float pseudocount=.5f) {
  namespace n=nn;
  if (!counts.defined() || counts.dim()!=2 || counts.numel()==0 ||
      counts.dtype()!=n::DType::Float32 || counts.device().backend!=compute::Backend::CPU ||
      !(pseudocount>0 && std::isfinite(pseudocount))) {
    throw std::invalid_argument("smoothed_bigram_logp: invalid counts or smoothing");
  }
  auto logs=counts.to_vector<float>();
  const auto vocabulary=counts.size(1);
  for (std::int64_t row=0;row<counts.size(0);++row) {
    const auto base=row*vocabulary;
    double total=0;
    for (std::int64_t token=0;token<vocabulary;++token) {
      const auto value=logs[base+token];
      if (!(value>=0 && std::isfinite(value))) {
        throw std::invalid_argument("smoothed_bigram_logp: invalid count");
      }
      total+=double(value)+pseudocount;
    }
    for (std::int64_t token=0;token<vocabulary;++token) {
      auto& entry=logs[base+token];
      entry=std::log((double(entry)+pseudocount)/total);
    }
  }
  return n::from_vector(logs,counts.shape());
}

/// @brief Sum log probabilities for two-token completions under a prompt-specific bigram.
/// @param table Floating CPU log probabilities [prompts*(vocabulary+1),vocabulary].
/// Rows group by prompt, then by previous token, with the start row last in each group.
/// @param prompts CPU Int32 or Int64 prompt IDs [batch].
/// @param completions CPU Int32 or Int64 token IDs [batch,candidates,2].
/// @return Completion log probabilities [batch,candidates].
/// @throws std::invalid_argument Invalid shapes, storage or IDs.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto out=dm::rl::bigram_completion_logp(n::full({3,2},-1.),
///     n::from_vector<int>({0}),n::from_vector<int>({0,1},{1,1,2}));
/// std::print("{:.0f}\n",out.item()); // -2
/// @endcode
inline nn::Tensor bigram_completion_logp(const nn::Tensor& table,const nn::Tensor& prompts,
                                          const nn::Tensor& completions) {
  namespace n=nn;
  if (!table.defined() || !prompts.defined() || !completions.defined() ||
      table.dim()!=2 || table.numel()==0 || table.size(0)<=table.size(1) ||
      table.size(0)%(table.size(1)+1)!=0 ||
      !n::is_floating(table.dtype()) || prompts.dim()!=1 || prompts.size(0)<1 ||
      completions.dim()!=3 || completions.size(0)!=prompts.size(0) ||
      completions.size(1)<1 || completions.size(2)!=2) {
    throw std::invalid_argument("bigram_completion_logp: incompatible shapes");
  }
  for (const auto* t:{&table,&prompts,&completions}) {
    if (t->device().backend!=compute::Backend::CPU) throw std::invalid_argument("Bigram CPU data required");
  }
  for (const auto* t:{&prompts,&completions}) {
    if (t->dtype()!=n::DType::Int32 && t->dtype()!=n::DType::Int64) {
      throw std::invalid_argument("Bigram IDs must be integer");
    }
  }
  const auto vocabulary=table.size(1),count=completions.size(1);
  const auto ids=prompts.to_vector<std::int64_t>(),tokens=completions.to_vector<std::int64_t>();
  std::vector<std::int64_t> first,second;
  for (std::size_t row=0;row<ids.size();++row) {
    if (ids[row]<0 || ids[row]>=table.size(0)/(vocabulary+1)) {
      throw std::invalid_argument("Bigram prompt out of range");
    }
    const auto base=ids[row]*(vocabulary+1);
    for (std::int64_t candidate=0;candidate<count;++candidate) {
      const auto i=2*(row*count+candidate);
      const auto a=tokens[i],b=tokens[i+1];
      if (a<0 || a>=vocabulary || b<0 || b>=vocabulary) {
        throw std::invalid_argument("Bigram token out of range");
      }
      first.push_back((base+vocabulary)*vocabulary+a);
      second.push_back((base+a)*vocabulary+b);
    }
  }
  const auto flat=table.reshape({-1});
  return (n::gather(flat,n::from_vector(first))+n::gather(flat,n::from_vector(second)))
      .reshape({prompts.size(0),count});
}

/// @brief Compute DPO loss and margins relative to frozen reference sequence scores.
/// @param policy Floating CPU log probabilities [batch,2], preferred then rejected.
/// @param reference Same-shaped reference log probabilities, detached.
/// @param beta Finite positive inverse temperature, default one.
/// @return Mean logistic loss and preferred-minus-rejected log-ratio margin [batch].
/// @throws std::invalid_argument Invalid geometry, dtype, storage or beta.
/// @par Example
/// @code
/// const auto x=dm::nn::zeros({2,2});
/// const auto [loss,margin]=dm::rl::dpo_loss(x,x);
/// std::print("{:.3f}\n",loss.item()); // 0.693
/// @endcode
/// @par References
/// - Rafailov et al. (2023), Direct Preference Optimization.
///   https://arxiv.org/abs/2305.18290
inline std::pair<nn::Tensor,nn::Tensor> dpo_loss(const nn::Tensor& policy,
    const nn::Tensor& reference,double beta=1.) {
  namespace n=nn;
  if (!policy.defined() || !reference.defined() || policy.dim()!=2 || policy.size(0)<1 ||
      policy.size(1)!=2 || reference.shape()!=policy.shape() || !n::is_floating(policy.dtype()) ||
      reference.dtype()!=policy.dtype() || policy.device().backend!=compute::Backend::CPU ||
      reference.device().backend!=compute::Backend::CPU || !(beta>0 && std::isfinite(beta))) {
    throw std::invalid_argument("dpo_loss: invalid sequence scores or beta");
  }
  const auto ratio=policy-reference.detach();
  const auto preferred=n::slice(ratio,{0,0},{ratio.size(0),1});
  const auto rejected=n::slice(ratio,{0,1},{ratio.size(0),1});
  const auto margin=(beta*(preferred-rejected)).reshape({ratio.size(0)});
  return {n::mean(n::softplus(-margin)),margin};
}

/// @brief Score equal-length trajectory pairs with a shared per-step reward network.
/// @param params Parameter store for the tanh hidden layer and bias-free reward head.
/// @param segments Floating CPU tensor [batch,2,steps,features], all dimensions nonempty.
/// @param hidden Positive hidden width.
/// @return Step rewards [batch,2,steps], total returns [batch,2], and left-win logits [batch].
/// @throws std::invalid_argument Invalid segment geometry or hidden width.
/// @par Example
/// @code
/// dm::nn::ParameterStore params(1);
/// const auto out=dm::rl::preference_segment_scores(params,dm::nn::zeros({2,2,3,4}),16);
/// std::print("{}\n",out[0].size(2)); // 3
/// @endcode
/// @par References
/// - Christiano et al. (2017), Deep Reinforcement Learning from Human Preferences.
///   https://arxiv.org/abs/1706.03741
inline std::array<nn::Tensor,3> preference_segment_scores(nn::ParameterStore& params,
    const nn::Tensor& segments,std::int64_t hidden) {
  namespace n=nn;
  if (!segments.defined() || segments.dim()!=4 || segments.numel()==0 ||
      segments.size(1)!=2 || !n::is_floating(segments.dtype()) ||
      segments.device().backend!=compute::Backend::CPU || hidden<1) {
    throw std::invalid_argument("preference_segment_scores: invalid segments or width");
  }
  const auto encoded=n::dense(params,"hidden",segments,hidden,n::Activation::Tanh);
  const auto weights=params.get("reward/W",{hidden,1},n::init::uniform(.2));
  const auto rewards=n::matmul(encoded,weights).reshape({segments.size(0),2,segments.size(2)});
  const auto returns=n::sum(rewards,{2});
  const auto delta=n::slice(returns,{0,0},{segments.size(0),1})-
      n::slice(returns,{0,1},{segments.size(0),1});
  return {rewards,returns,delta.reshape({segments.size(0)})};
}

/// @brief Combine token-level clipped PPO with a frozen reference KL penalty.
/// @param logp Finite CPU floating log-policy rows [batch,vocabulary].
/// @param reference Matching reference log probabilities, detached.
/// @param actions CPU integer selected tokens [batch].
/// @param old Selected log probabilities at collection [batch].
/// @param advantage Fixed token advantages [batch].
/// @param value Current value prediction [batch].
/// @param target Fixed value targets [batch].
/// @param beta Finite nonnegative reference KL weight, default 0.12.
/// @return Total loss, clipped policy loss, mean exact categorical KL, and ratios [batch].
/// @throws std::invalid_argument Invalid inputs or beta.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto zero=n::zeros({1});
/// const auto out=dm::rl::rlhf_ppo_loss(n::zeros({1,1}),n::zeros({1,1}),
///     n::from_vector<int>({0}),zero,n::ones({1}),zero,zero);
/// std::print("{:.0f}\n",out[0].item()); // -1
/// @endcode
/// @see mappo_loss(), dpo_loss()
inline std::array<nn::Tensor,4> rlhf_ppo_loss(const nn::Tensor& logp,const nn::Tensor& reference,
    const nn::Tensor& actions,const nn::Tensor& old,const nn::Tensor& advantage,
    const nn::Tensor& value,const nn::Tensor& target,double beta=.12) {
  namespace n=nn;
  if (!logp.defined() || logp.dim()!=2 || logp.numel()==0 || !reference.defined() ||
      reference.shape()!=logp.shape() || reference.dtype()!=logp.dtype() ||
      reference.device().backend!=compute::Backend::CPU || !actions.defined() || !old.defined() ||
      actions.dim()!=1 || old.dim()!=1 || !(beta>=0 && std::isfinite(beta))) {
    throw std::invalid_argument("rlhf_ppo_loss: invalid inputs or beta");
  }
  const auto batch=logp.size(0);
  const auto [loss,policy,ratio]=mappo_loss(logp.reshape({batch,1,logp.size(1)}),
      actions.reshape({actions.size(0),1}),old.reshape({old.size(0),1}),advantage,value,target,.2,.5,0.);
  const auto kl=n::mean(n::sum(n::exp(logp)*(logp-reference.detach()),{1}));
  return {loss+beta*kl,policy,kl,ratio.reshape({batch})};
}

} // namespace datamunge::rl
