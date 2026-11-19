/// @file
/// @brief Joint replay actions for centralized deterministic actor updates.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/ops.hpp"

namespace datamunge::rl {

/// @brief Replace one agent's replay actions with differentiable policy actions.
///
/// The centralized critic receives the other agents' recorded actions
/// as constants. Gradients through the returned joint actions reach
/// only the supplied policy action block.
/// @param replay Joint recorded actions [batch,total_action_width], floating and nonempty.
/// @param policy Replacement actions [batch,agent_action_width], matching dtype and device.
/// @param first_column Zero-based start of the agent's block.
/// @return Hybrid action rows with the same shape as replay.
/// @throws std::invalid_argument Incompatible geometry, dtype, device or offset.
/// @par Example
/// @code
/// const auto actions=dm::rl::maddpg_replay_actions(dm::nn::zeros({2,3}),dm::nn::ones({2,1}),1);
/// std::print("{:.0f}\n",actions.sum().item()); // 2
/// @endcode
/// @par References
/// - Lowe et al. (2017), Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.
///   https://arxiv.org/abs/1706.02275
inline nn::Tensor maddpg_replay_actions(const nn::Tensor& replay,const nn::Tensor& policy,
                                       std::int64_t first_column) {
  namespace n=nn;
  if (!replay.defined() || !policy.defined() || replay.dim()!=2 || policy.dim()!=2 ||
      replay.numel()==0 || policy.numel()==0 || policy.size(0)!=replay.size(0) ||
      !n::is_floating(replay.dtype()) || policy.dtype()!=replay.dtype() ||
      replay.device().backend!=policy.device().backend || replay.device().index!=policy.device().index ||
      first_column<0 || first_column>replay.size(1)-policy.size(1)) {
    throw std::invalid_argument("maddpg_replay_actions: incompatible inputs");
  }
  const auto batch=replay.size(0),end=first_column+policy.size(1);
  const auto fixed=replay.detach();
  std::vector<n::Tensor> columns;
  if (first_column>0) columns.push_back(n::slice(fixed,{0,0},{batch,first_column}));
  columns.push_back(policy);
  if (end<replay.size(1)) columns.push_back(n::slice(fixed,{0,end},{batch,replay.size(1)-end}));
  return n::concat(columns,1);
}

} // namespace datamunge::rl
