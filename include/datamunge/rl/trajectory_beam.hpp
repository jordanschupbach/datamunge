/// @file
/// @brief Reward-ranked autoregressive planning over trajectory tokens.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::rl {

/// @brief Plan action–reward–state triples using an autoregressive token predictor.
///
/// Expand each action in the beam and greedily complete its reward and
/// successor tokens. Rank by discounted predicted reward, breaking ties
/// by the sum of token log likelihoods normalized within each token type.
/// Exact ties preserve expansion order.
/// @param history Nonempty observed token prefix.
/// @param actions Permitted action token IDs.
/// @param states Permitted successor token IDs.
/// @param rewards Consecutive reward magnitudes, indexed from reward_offset.
/// @param reward_offset Token ID of the first reward magnitude.
/// @param predict Callback returning next-token logits indexed by token ID.
/// @param horizon Positive number of planned transitions.
/// @param width Positive beam width.
/// @param discount Finite discount in [0,1], default 0.95.
/// @return Planned token suffix, beginning with the chosen action token.
/// @throws std::invalid_argument Invalid planner options or invalid candidate logits.
/// @par Example
/// @code
/// const std::array history{0},states{0};
/// const std::array actions{1,2};
/// const std::array rewards{0.,1.};
/// const auto plan=dm::rl::trajectory_beam_plan(history,actions,states,rewards,3,
///     [](auto){return std::vector<float>{0,1,0,0,1};},1,2);
/// std::print("{}\n",plan.front()); // 1
/// @endcode
/// @par References
/// - Janner, Li and Levine (2021), Offline Reinforcement Learning as One Big Sequence Modeling Problem.
///   https://arxiv.org/abs/2106.02039
inline std::vector<int> trajectory_beam_plan(std::span<const int> history,std::span<const int> actions,
    std::span<const int> states,std::span<const double> rewards,int reward_offset,
    const std::function<std::vector<float>(std::span<const int>)>& predict,
    int horizon,std::size_t width,double discount=.95) {
  if (history.empty() || actions.empty() || states.empty() || rewards.empty() ||
      !predict || horizon<1 || width==0 || !(discount>=0 && discount<=1) || reward_offset<0 ||
      rewards.size()-1>static_cast<std::size_t>(std::numeric_limits<int>::max()-reward_offset) ||
      !std::all_of(rewards.begin(),rewards.end(),[](double r){return std::isfinite(r);})) {
    throw std::invalid_argument("trajectory_beam_plan: invalid planner options");
  }
  struct Candidate {
    std::vector<int> tokens;  ///< Observed and generated token sequence.
    double value=0;           ///< Discounted predicted return.
    double likelihood=0;      ///< Type-conditioned log likelihood of generated tokens.
  };
  std::vector<int> reward_ids(rewards.size());
  for (std::size_t i=0;i<rewards.size();++i) reward_ids[i]=reward_offset+static_cast<int>(i);
  const auto append=[&](Candidate& candidate,std::span<const int> allowed,int selected=-1) {
    const auto logits=predict(candidate.tokens);
    for (const int token:allowed) {
      if (token<0 || static_cast<std::size_t>(token)>=logits.size() || !std::isfinite(logits[token])) {
        throw std::invalid_argument("trajectory_beam_plan: invalid candidate logit");
      }
    }
    const int best=*std::max_element(allowed.begin(),allowed.end(),[&](int a,int b){return logits[a]<logits[b];});
    double mass=0;
    for (const int token:allowed) mass+=std::exp(logits[token]-logits[best]);
    const int chosen=selected<0?best:selected;
    candidate.likelihood+=logits[chosen]-logits[best]-std::log(mass);
    candidate.tokens.push_back(chosen);
    return chosen;
  };
  std::vector<Candidate> beam{{std::vector<int>(history.begin(),history.end())}};
  for (int step=0;step<horizon;++step) {
    std::vector<Candidate> next;
    for (const auto& parent:beam) {
      for (const int action:actions) {
        auto child=parent;
        append(child,actions,action);
        const int reward=append(child,reward_ids);
        child.value+=std::pow(discount,step)*rewards[reward-reward_offset];
        append(child,states);
        next.push_back(std::move(child));
      }
    }
    std::stable_sort(next.begin(),next.end(),[](const auto& a,const auto& b) {
      return a.value==b.value?a.likelihood>b.likelihood:a.value>b.value;
    });
    next.resize(std::min(width,next.size()));
    beam=std::move(next);
  }
  auto best=std::move(beam.front().tokens);
  best.erase(best.begin(),best.begin()+history.size());
  return best;
}

} // namespace datamunge::rl
