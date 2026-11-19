/// @file
/// @brief Independent tabular updates for agents sharing a team reward.
#pragma once

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "datamunge/rl/tabular.hpp"

namespace datamunge::rl {

/// @brief Update independent private-state Q-tables from one joint transition.
///
/// Each agent bootstraps using its own successor and action values.
/// Targets use the pre-update tables. A terminal team transition
/// removes the bootstrap for all agents.
/// @param tables One mutable Q-table per agent.
/// @param states Current private state IDs.
/// @param actions Executed action IDs.
/// @param successors Next private state IDs.
/// @param reward Finite shared immediate reward.
/// @param terminal Whether the joint episode ended.
/// @param rate Learning rate in [0,1], default 0.2.
/// @param discount Discount in [0,1], default 0.95.
/// @throws std::invalid_argument Incompatible batch sizes, indices or update options.
/// @par Example
/// @code
/// namespace r=dm::rl;
/// std::array tables{r::foundations::QTable(2,1),r::foundations::QTable(2,1)};
/// const std::array from{0,0},action{0,0},next{1,1};
/// r::independent_q_update(tables,from,action,next,1.f,true);
/// std::print("{:.1f}\n",tables[0](0,0)); // 0.2
/// @endcode
inline void independent_q_update(std::span<foundations::QTable> tables,
    std::span<const int> states,std::span<const int> actions,std::span<const int> successors,
    float reward,bool terminal,float rate=.2f,float discount=.95f) {
  if (tables.empty() || states.size()!=tables.size() || actions.size()!=tables.size() ||
      successors.size()!=tables.size() || !std::isfinite(reward) ||
      !(rate>=0 && rate<=1) || !(discount>=0 && discount<=1)) {
    throw std::invalid_argument("independent_q_update: incompatible transition or options");
  }
  std::vector<float> updated(tables.size());
  for (std::size_t i=0;i<tables.size();++i) {
    if (states[i]<0 || states[i]>=tables[i].S || successors[i]<0 || successors[i]>=tables[i].S ||
        actions[i]<0 || actions[i]>=tables[i].A) {
      throw std::invalid_argument("independent_q_update: invalid state or action");
    }
    const float old=tables[i](states[i],actions[i]);
    const float target=reward+(terminal?0.f:discount*tables[i].Max(successors[i]));
    updated[i]=old+rate*(target-old);
    if (!std::isfinite(updated[i])) throw std::invalid_argument("independent_q_update: nonfinite update");
  }
  for (std::size_t i=0;i<tables.size();++i) tables[i](states[i],actions[i])=updated[i];
}

} // namespace datamunge::rl
