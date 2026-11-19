/// @file
/// @brief Future-goal relabeling for discrete hindsight replay.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>

namespace datamunge::rl {

/// @brief Relabel a transition with a sampled future achieved goal.
///
/// The eligible window includes the current transition's successor and
/// ends at the last successor in the same episode. A candidate equal
/// to the current state falls back to the original goal.
/// @param achieved Goals achieved after each transition in the episode.
/// @param time Zero-based index of the transition being relabeled.
/// @param state Current state, before the transition.
/// @param original_goal Goal assigned when the episode was collected.
/// @param rng Generator advanced once, using modulo selection.
/// @return Future transition index and relabeled goal, or (-1,original_goal) on fallback.
/// @throws std::invalid_argument Invalid time or episode length outside the RNG range.
/// @par Example
/// @code
/// std::mt19937 rng(1);
/// const std::array achieved{4,5};
/// const auto [future,goal]=dm::rl::future_hindsight_goal(achieved,1,4,7,rng);
/// std::print("{} {}\n",future,goal); // 1 5
/// @endcode
/// @par References
/// - Andrychowicz et al. (2017), Hindsight Experience Replay.
///   https://arxiv.org/abs/1707.01495
inline std::pair<std::int64_t,int> future_hindsight_goal(std::span<const int> achieved,
    std::size_t time,int state,int original_goal,std::mt19937& rng) {
  if (time>=achieved.size() ||
      achieved.size()>std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("future_hindsight_goal: invalid episode or time");
  }
  const auto future=time+rng()%(achieved.size()-time);
  if (achieved[future]==state) return {-1,original_goal};
  return {static_cast<std::int64_t>(future),achieved[future]};
}

} // namespace datamunge::rl
