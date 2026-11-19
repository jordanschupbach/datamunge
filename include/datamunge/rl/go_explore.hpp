/// @file
/// @brief Deterministic Go-Explore archives and observed-graph recovery.
#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <queue>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::rl {

/// @brief Result of deterministic archive exploration.
/// @par Example
/// @code
/// dm::rl::GoExploreResult result;
/// std::print("{}\n",result.first_goal_round); // -1
/// @endcode
struct GoExploreResult {
  std::vector<std::optional<std::vector<int>>> paths; ///< Shortest discovered return path per cell.
  std::vector<std::vector<int>> edges; ///< Observed successor per cell and action, or -1.
  std::vector<std::vector<int>> trace; ///< Selected cell followed by burst actions and -1 padding.
  std::vector<int> coverage;          ///< Occupied archive cells after each burst.
  int first_goal_round=-1;           ///< First goal discovery, zero based, or -1.
};

/// @brief Revisit archived cells in a deterministic model and explore from them.
///
/// Sample an archived nonterminal cell with weight inverse to one plus
/// its number of selections. Replay its path from the start, explore
/// uniformly for up to the burst length, and retain shorter paths.
/// Only transitions in the exploration bursts enter the edge table.
/// @param states Positive number of discrete cells, at least two.
/// @param actions Positive number of actions per cell.
/// @param start Valid initial cell.
/// @param goal Valid terminal cell distinct from start.
/// @param rounds Positive number of archive selections.
/// @param burst Positive number of exploratory actions per selection.
/// @param rng Generator advanced by archive sampling and actions.
/// @param move Deterministic successor function (cell,action), returning a valid cell.
/// @return Paths, learned edges, selection trace, coverage and first goal discovery.
/// @throws std::invalid_argument Invalid geometry, missing transition function, or invalid successor.
/// @throws std::logic_error Replaying an archived path fails to return to its cell.
/// @par Example
/// @code
/// std::mt19937 rng(1);
/// const auto found=dm::rl::go_explore(2,1,0,1,1,1,rng,[](int,int){return 1;});
/// std::print("{}\n",found.first_goal_round); // 0
/// @endcode
/// @par References
/// - Ecoffet et al. (2021), First return, then explore.
///   doi:10.1038/s41586-020-03157-9
/// @see observed_recovery_actions()
inline GoExploreResult go_explore(int states,int actions,int start,int goal,int rounds,int burst,
                                 std::mt19937& rng,const std::function<int(int,int)>& move) {
  if (states<2 || actions<1 || start<0 || start>=states || goal<0 || goal>=states ||
      start==goal || rounds<1 || burst<1 || !move) {
    throw std::invalid_argument("go_explore: invalid options");
  }
  GoExploreResult result;
  result.paths.resize(states);
  result.paths[start].emplace();
  result.edges.assign(states,std::vector<int>(actions,-1));
  std::vector<int> selections(states);
  int known=1;
  const auto advance=[&](int state,int action) {
    const int next=move(state,action);
    if (next<0 || next>=states) throw std::invalid_argument("go_explore: invalid successor");
    return next;
  };
  for (int round=0;round<rounds;++round) {
    std::vector<double> weights(states);
    for (int cell=0;cell<states;++cell) {
      if (cell!=goal && result.paths[cell]) weights[cell]=1./(1.+selections[cell]);
    }
    const int selected=std::discrete_distribution<int>(weights.begin(),weights.end())(rng);
    ++selections[selected];
    auto path=*result.paths[selected];
    int cell=start;
    for (const auto action:path) cell=advance(cell,action);
    if (cell!=selected) throw std::logic_error("go_explore: replay returned to a different cell");
    std::vector<int> row(static_cast<std::size_t>(burst)+1,-1);
    row[0]=selected;
    for (int step=0;step<burst && cell!=goal;++step) {
      const int action=rng()%actions;
      row[step+1]=action;
      const int next=advance(cell,action);
      result.edges[cell][action]=next;
      cell=next;
      path.push_back(action);
      auto& incumbent=result.paths[cell];
      if (!incumbent) {
        ++known;
        if (cell==goal) result.first_goal_round=round;
      }
      if (!incumbent || path.size()<incumbent->size()) incumbent=path;
    }
    result.trace.push_back(std::move(row));
    result.coverage.push_back(known);
  }
  return result;
}

/// @brief Recover shortest-path action labels using only observed transitions.
///
/// Reverse breadth-first search starts at the goal. Among predecessors
/// of the same successor, lower state and action indices come first.
/// @param edges Successor by state and action. -1 denotes an unobserved edge.
/// @param goal Valid goal state index.
/// @return One shortest-path action per state, or -1 at unreachable states and the goal.
/// @throws std::invalid_argument Invalid goal or successor indices.
/// @par Example
/// @code
/// const std::vector<std::vector<int>> edges{{-1,1},{-1,-1}};
/// const auto actions=dm::rl::observed_recovery_actions(edges,1);
/// std::print("{}\n",actions[0]); // 1
/// @endcode
/// @see go_explore()
inline std::vector<int> observed_recovery_actions(std::span<const std::vector<int>> edges,int goal) {
  if (goal<0 || static_cast<std::size_t>(goal)>=edges.size()) {
    throw std::invalid_argument("observed_recovery_actions: invalid goal");
  }
  std::vector<std::vector<std::pair<std::size_t,int>>> predecessors(edges.size());
  for (std::size_t state=0;state<edges.size();++state) {
    for (std::size_t action=0;action<edges[state].size();++action) {
      const int next=edges[state][action];
      if (next==-1) continue;
      if (next<0 || static_cast<std::size_t>(next)>=edges.size()) {
        throw std::invalid_argument("observed_recovery_actions: invalid successor");
      }
      predecessors[next].emplace_back(state,static_cast<int>(action));
    }
  }
  std::vector<int> labels(edges.size(),-1);
  std::queue<std::size_t> pending;
  pending.push(goal);
  while (!pending.empty()) {
    const auto next=pending.front();
    pending.pop();
    for (const auto& [state,action]:predecessors[next]) {
      if (state==static_cast<std::size_t>(goal) || labels[state]!=-1) continue;
      labels[state]=action;
      pending.push(state);
    }
  }
  return labels;
}

} // namespace datamunge::rl
