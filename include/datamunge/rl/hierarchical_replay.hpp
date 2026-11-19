/// @file
/// @brief Discrete hierarchical off-policy replay rules.
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Pick a HIRO replay goal by action likelihood under the current worker.
///
/// Sum selected action log probabilities over valid steps. Among the
/// admissible goals, prefer the requested goal in a tie, then the
/// lowest remaining goal index.
/// @param logits Worker action logits [goals,steps,actions], Float32 CPU.
/// @param actions Replayed actions [valid_steps], CPU Int32 or Int64. Trailing logits are unused.
/// @param first Lowest admissible goal index.
/// @param last Highest admissible goal index, inclusive.
/// @param requested Original goal index within [first,last].
/// @return Selected goal and vector of log-likelihood scores [goals].
/// @throws std::invalid_argument Invalid tensor geometry, action or goal.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto [goal,scores]=dm::rl::hiro_replay_goal(n::zeros({2,3,2}),
///     n::from_vector<int>({1}),0,1,1);
/// std::print("{}\n",goal); // 1
/// @endcode
/// @par References
/// - Nachum et al. (2018), Data-Efficient Hierarchical Reinforcement Learning.
///   https://arxiv.org/abs/1805.08296
inline std::pair<std::int64_t,nn::Tensor> hiro_replay_goal(const nn::Tensor& logits,
    const nn::Tensor& actions,std::int64_t first,std::int64_t last,std::int64_t requested) {
  namespace n=nn;
  if (!logits.defined() || logits.dim()!=3 || logits.dtype()!=n::DType::Float32 ||
      logits.device().backend!=compute::Backend::CPU || logits.size(2)<1 ||
      !actions.defined() || actions.dim()!=1 || actions.size(0)<1 || actions.size(0)>logits.size(1) ||
      (actions.dtype()!=n::DType::Int32 && actions.dtype()!=n::DType::Int64) ||
      actions.device().backend!=compute::Backend::CPU ||
      first<0 || last>=logits.size(0) || requested<first || requested>last) {
    throw std::invalid_argument("hiro_replay_goal: invalid shapes or goal range");
  }
  const auto ids=actions.to_vector<std::int64_t>();
  if (!std::all_of(ids.begin(),ids.end(),[&](auto id){return id>=0 && id<logits.size(2);})) {
    throw std::invalid_argument("hiro_replay_goal: action out of range");
  }
  const auto valid=n::slice(logits,{0,0,0},{logits.size(0),actions.size(0),logits.size(2)});
  const auto selected=n::one_hot(actions,logits.size(2));
  const auto scores=n::sum(n::sum(n::log_softmax(valid,-1)*selected,{2}),{1});
  const auto values=scores.to_vector<float>();
  auto goal=requested;
  for (auto candidate=first;candidate<=last;++candidate) {
    if (values[candidate]>values[goal]) goal=candidate;
  }
  return {goal,scores};
}

/// @brief Label a HAC segment with its achieved goal and subgoal-test outcome.
/// @param requested Intended discrete goal.
/// @param achieved Final discrete state of the segment.
/// @param tested Whether the worker ran a subgoal test without exploration.
/// @return Replay goal and whether the manager should receive a failed-test penalty.
/// @par Example
/// @code
/// const auto [goal,failed]=dm::rl::hac_replay_goal(4,3,true);
/// std::print("{} {}\n",goal,failed); // 3 true
/// @endcode
/// @par References
/// - Levy et al. (2019), Learning Multi-Level Hierarchies with Hindsight.
///   https://arxiv.org/abs/1712.00948
inline std::pair<int,bool> hac_replay_goal(int requested,int achieved,bool tested) {
  return {achieved,tested && achieved!=requested};
}

} // namespace datamunge::rl
