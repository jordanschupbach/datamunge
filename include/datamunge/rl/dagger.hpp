/// @file
/// @brief Expert labeling of learner-induced trajectories.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace datamunge::rl {

/// @brief Collect one batch of expert-labeled DAgger trajectories.
///
/// Query the expert at every visited nonterminal state and record the
/// label even when the learner supplies the action. The caller can
/// append those labels to previous data before retraining.
/// @param starts Initial state of each episode, in collection order.
/// @param goal Terminal state ID.
/// @param horizon Positive action budget per episode.
/// @param learner Learner action callback (state).
/// @param expert Expert action callback (state).
/// @param move Successor callback (state,action).
/// @param rng Generator used for action mixing.
/// @param expert_every Zero for learner-only rollouts, otherwise expert probability 1/expert_every.
/// @return Rows (episode,state,action,next,expert_label,used_expert), with zero-based episode IDs.
/// @throws std::invalid_argument Invalid horizon, episode count or absent callback.
/// @par Example
/// @code
/// std::mt19937 rng(1);
/// const std::array starts{0};
/// const auto visits=dm::rl::dagger_collect(starts,1,2,[](int){return 0;},
///     [](int){return 1;},[](int,int){return 1;},rng);
/// std::print("{} {}\n",visits[0][2],visits[0][4]); // 0 1
/// @endcode
/// @par References
/// - Ross et al. (2011), A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.
///   https://proceedings.mlr.press/v15/ross11a.html
inline std::vector<std::array<int,6>> dagger_collect(std::span<const int> starts,int goal,int horizon,
    const std::function<int(int)>& learner,const std::function<int(int)>& expert,
    const std::function<int(int,int)>& move,std::mt19937& rng,std::uint32_t expert_every=0) {
  if (horizon<1 || !learner || !expert || !move ||
      starts.size()>static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("dagger_collect: invalid rollout options");
  }
  std::vector<std::array<int,6>> trace;
  for (std::size_t episode=0;episode<starts.size();++episode) {
    int state=starts[episode];
    for (int step=0;step<horizon && state!=goal;++step) {
      const int label=expert(state);
      const bool use_expert=expert_every!=0 && rng()%expert_every==0;
      const int action=use_expert?label:learner(state);
      const int next=move(state,action);
      trace.push_back({static_cast<int>(episode),state,action,next,label,use_expert});
      state=next;
    }
  }
  return trace;
}

} // namespace datamunge::rl
