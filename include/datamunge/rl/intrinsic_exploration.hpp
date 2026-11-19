/// @file
/// @brief Sparse grid exploration and random network distillation.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/optim.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/rl/tabular.hpp"

namespace datamunge::rl {

/// @brief Learn to predict the embedding of a fixed random network.
///
/// Both networks have one ReLU hidden layer and a linear output. The
/// target is initialized independently and only predictor parameters
/// receive Adam updates.
/// @par Example
/// @code
/// dm::rl::RandomNetworkDistillation rnd(2,4,3);
/// const auto [reward,prediction,target]=rnd.evaluate(dm::nn::zeros({1,2}));
/// std::print("{}\n",reward.size(0)); // 1
/// @endcode
/// @par References
/// - Burda et al. (2019), Exploration by Random Network Distillation.
///   https://arxiv.org/abs/1810.12894
class RandomNetworkDistillation {
 public:
  /// @brief Initialize independent predictor and target networks.
  /// @param inputs Positive observation width.
  /// @param hidden Positive hidden width.
  /// @param features Positive embedding width.
  /// @param predictor_seed Predictor initializer seed, default 43701.
  /// @param target_seed Distinct target initializer seed, default 43702.
  /// @param rate Positive finite Adam learning rate, default 0.003.
  /// @throws std::invalid_argument Invalid widths, equal seeds or invalid rate.
  /// @par Example
  /// @code
  /// dm::rl::RandomNetworkDistillation rnd(2,4,3);
  /// @endcode
  RandomNetworkDistillation(std::int64_t inputs,std::int64_t hidden,std::int64_t features,
                            std::uint64_t predictor_seed=43701,std::uint64_t target_seed=43702,
                            double rate=.003)
      : inputs_(inputs),hidden_(hidden),features_(features),predictor_(predictor_seed),target_(target_seed) {
    if (inputs<1 || hidden<1 || features<1 || predictor_seed==target_seed ||
        !(rate>0 && std::isfinite(rate))) throw std::invalid_argument("RND: invalid configuration");
    const auto probe=nn::zeros({1,inputs});
    (void)encode(predictor_,probe);
    (void)encode(target_,probe);
    optimizer_.emplace(predictor_,rate,nn::OptimizerConfig{.kind=nn::OptimizerKind::Adam});
  }

  /// @brief Compute detached per-observation novelty and both embeddings.
  /// @param observations Float32 CPU [batch,inputs] with positive batch size and finite entries.
  /// @return Half squared embedding distance [batch], prediction and target [batch,features].
  /// @throws std::invalid_argument Invalid observation tensor.
  /// @par Example
  /// @code
  /// dm::rl::RandomNetworkDistillation rnd(2,4,3);
  /// const auto result=rnd.evaluate(dm::nn::ones({2,2}));
  /// std::print("{}\n",result[2].size(1)); // 3
  /// @endcode
  std::array<nn::Tensor,3> evaluate(const nn::Tensor& observations) {
    check(observations);
    const nn::NoGradGuard guard;
    const auto prediction=encode(predictor_,observations);
    const auto target=encode(target_,observations);
    return {.5*nn::sum(nn::square(prediction-target),{1}),prediction,target};
  }

  /// @brief Take one Adam step on mean novelty without updating the target.
  /// @param observations Float32 CPU [batch,inputs] with positive batch size and finite entries.
  /// @return Mean novelty of that batch before the update.
  /// @throws std::invalid_argument Invalid observation tensor.
  /// @par Example
  /// @code
  /// dm::rl::RandomNetworkDistillation rnd(2,4,3);
  /// const double loss=rnd.train(dm::nn::ones({2,2}));
  /// @endcode
  double train(const nn::Tensor& observations) {
    check(observations);
    const auto target=[&] {
      const nn::NoGradGuard guard;
      return encode(target_,observations);
    }();
    const auto loss=.5*nn::mean(nn::sum(nn::square(encode(predictor_,observations.detach())-target),{1}));
    const double value=loss.item();
    optimizer_->step(nn::grad(loss,optimizer_->parameters()));
    return value;
  }

 private:
  /// @brief Apply the shared architecture with the selected parameters.
  /// @param parameters Parameter store for one network.
  /// @param observations Batched input matrix.
  /// @return Network embeddings.
  nn::Tensor encode(nn::ParameterStore& parameters,const nn::Tensor& observations) const {
    return nn::dense(parameters,"embedding",
        nn::dense(parameters,"hidden",observations,hidden_,nn::Activation::Relu),features_);
  }

  /// @brief Validate observation storage, geometry and finiteness.
  /// @param observations Input tensor to check.
  /// @throws std::invalid_argument Invalid observations.
  void check(const nn::Tensor& observations) const {
    if (!observations.defined() || observations.dim()!=2 || observations.size(0)<1 ||
        observations.size(1)!=inputs_ || observations.dtype()!=nn::DType::Float32 ||
        observations.device().backend!=compute::Backend::CPU) {
      throw std::invalid_argument("RND: invalid observation shape or storage");
    }
    const auto values=observations.to_vector<float>();
    if (!std::all_of(values.begin(),values.end(),[](float value){return std::isfinite(value);})) {
      throw std::invalid_argument("RND: observations must be finite");
    }
  }

  std::int64_t inputs_;                     ///< Observation width.
  std::int64_t hidden_;                     ///< Hidden layer width.
  std::int64_t features_;                   ///< Embedding width.
  nn::ParameterStore predictor_;             ///< Trainable predictor parameters.
  nn::ParameterStore target_;                ///< Fixed random target parameters.
  std::optional<nn::Optimizer> optimizer_;    ///< Adam initialized after predictor parameters.
};

/// @brief Move one step on the clipped seven-by-seven exploration grid.
/// @param state Row-major state ID from 0 to 48.
/// @param action Up (0), down (1), left (2), or right (3).
/// @return Row-major successor state ID.
/// @throws std::invalid_argument Invalid input ID.
/// @par Example
/// @code
/// std::print("{}\n",dm::rl::sparse_grid_move(0,1)); // 7
/// @endcode
inline int sparse_grid_move(int state,int action) {
  constexpr std::array dy{-1,1,0,0},dx{0,0,-1,1};
  if (state<0 || state>=49 || action<0 || action>=4) {
    throw std::invalid_argument("sparse_grid_move: invalid state or action");
  }
  return 7*std::clamp(state/7+dy[action],0,6)+std::clamp(state%7+dx[action],0,6);
}

/// @brief Run the sparse 7-by-7 exploration fixture with optional intrinsic rewards.
///
/// Task and behavior Q-tables see the same transitions. Only behavior
/// receives the intrinsic bonus, scaled by 0.5. Episodes end at the goal
/// or after 64 steps. Replay samples 32 transitions every 16 steps.
/// @param seed Behavior RNG seed. Replay uses an independent seed offset of 17.
/// @param curiosity Optional callback on (state, action, successor), returning finite nonnegative reward.
/// @param train Optional callback receiving the sampled transition triples.
/// @param reset Optional callback invoked at the start and at episode boundaries.
/// @return Transition and reward tensors, visited-cell count, goal count and task-only greedy success.
/// @throws std::invalid_argument Curiosity returned a negative or nonfinite reward.
/// @par Example
/// @code
/// const auto [path,rewards,visited,goals,success]=dm::rl::sparse_grid_experiment(43602);
/// std::print("{}\n",path.size(0)); // 2048
/// @endcode
/// @see RandomNetworkDistillation
inline auto sparse_grid_experiment(
    unsigned seed,const std::function<float(const std::array<int,3>&)>& curiosity={},
    const std::function<void(std::span<const std::array<int,3>>)>& train={},
    const std::function<void()>& reset={}) {
  constexpr int states=49, actions=4, budget=2048;
  std::mt19937 rng(seed), replay_rng(seed+17);
  foundations::QTable behavior(states,actions), task(states,actions);
  std::array<bool,states> seen{};
  seen[0]=true;
  std::vector<std::array<int,3>> replay;
  std::vector<int> path;
  std::vector<float> rewards;
  int cell=0, episode_step=0, hits=0;
  if (reset) reset();
  for (int i=0;i<budget;++i) {
    int action=behavior.Greedy(cell);
    if (rng()%4==0) action=rng()%actions;
    const int next=sparse_grid_move(cell,action);
    const bool terminal=next==states-1;
    replay.push_back({cell,action,next});
    const float bonus=curiosity?curiosity(replay.back()):0.f;
    if (!(bonus>=0 && std::isfinite(bonus))) throw std::invalid_argument("Invalid curiosity reward");
    const auto learn=[&](foundations::QTable& q,float reward) {
      const float after=terminal?0.f:.95f*q.Max(next);
      q(cell,action)+=.2f*(reward+after-q(cell,action));
    };
    learn(task,terminal);
    learn(behavior,float(terminal)+.5f*bonus);
    path.insert(path.end(),{cell,action,next});
    rewards.push_back(bonus);
    seen[next]=true;
    hits+=terminal;
    cell=next;
    if (++episode_step==64 || terminal) {
      cell=episode_step=0;
      if (reset) reset();
    }
    if (train && i%16==15) {
      std::array<std::array<int,3>,32> batch;
      for (auto& row:batch) row=replay[replay_rng()%replay.size()];
      train(batch);
    }
  }
  cell=0;
  for (int i=0;i<64 && cell!=states-1;++i) cell=sparse_grid_move(cell,task.Greedy(cell));
  return std::tuple{nn::from_vector(path,{budget,3}),nn::from_vector(rewards),
                    std::count(seen.begin(),seen.end(),true),hits,cell==states-1};
}

} // namespace datamunge::rl
