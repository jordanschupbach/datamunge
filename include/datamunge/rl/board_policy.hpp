/// @file
/// @brief Monte Carlo action-value policy for nine-cell board games.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/optim.hpp"
#include "datamunge/rl/environments/board_games.hpp"

namespace datamunge::rl {

/// @brief Predict board action values and fit outcomes from the moving player's perspective.
/// @par Example
/// @code
/// dm::rl::BoardValuePolicy policy;
/// dm::rl::environments::TicTacToe game;
/// const auto move=policy.greedy(game);
/// std::print("{}\n",move>=0 && move<9); // true
/// @endcode
class BoardValuePolicy {
 public:
  /// @brief Initialize the two-layer value network and its Adam optimizer.
  /// @param seed Parameter initializer seed, default 45301.
  /// @par Example
  /// @code
  /// dm::rl::BoardValuePolicy policy(1);
  /// @endcode
  explicit BoardValuePolicy(std::uint64_t seed=45301):params_(seed) {
    (void)forward(nn::zeros({1,9}));
    optimizer_.emplace(params_,.006,nn::OptimizerConfig{.kind=nn::OptimizerKind::Adam});
  }

  /// @brief Return the legal action with the greatest estimated outcome.
  /// @param game Live nine-cell game with a current-player-relative observation.
  /// @return Greedy legal action, breaking ties by the legal-action order.
  /// @throws std::invalid_argument Terminal or incompatible game.
  /// @par Example
  /// @code
  /// dm::rl::BoardValuePolicy policy;
  /// dm::rl::environments::TicTacToe game;
  /// game.Apply(policy.greedy(game));
  /// @endcode
  int greedy(const environments::Game& game) {
    if (game.Terminal() || game.num_actions()!=9) {
      throw std::invalid_argument("BoardValuePolicy: incompatible or terminal game");
    }
    const auto observation=game.Observation();
    const auto legal=game.LegalActions();
    if (observation.size()!=9 || legal.empty() ||
        !std::all_of(legal.begin(),legal.end(),[](int action){return action>=0 && action<9;})) {
      throw std::invalid_argument("BoardValuePolicy: invalid observation or legal actions");
    }
    const nn::NoGradGuard guard;
    const auto values=forward(nn::from_vector(observation,{1,9})).to_vector<float>();
    return *std::max_element(legal.begin(),legal.end(),[&](int a,int b){return values[a]<values[b];});
  }

  /// @brief Train one batch on squared error at the executed actions.
  /// @param boards CPU Float32 current-player-relative boards [batch,9].
  /// @param actions CPU Int32 or Int64 action IDs [batch] in [0,8].
  /// @param outcomes CPU Float32 moving-player outcomes [batch] in [-1,1].
  /// @return Mean squared error before the Adam update.
  /// @throws std::invalid_argument Invalid batch geometry, types or labels.
  /// @par Example
  /// @code
  /// dm::rl::BoardValuePolicy policy;
  /// const auto loss=policy.train(dm::nn::zeros({1,9}),dm::nn::from_vector<int>({4}),dm::nn::ones({1}));
  /// std::print("{:.0f}\n",loss); // 1
  /// @endcode
  double train(const nn::Tensor& boards,const nn::Tensor& actions,const nn::Tensor& outcomes) {
    namespace n=nn;
    if (!boards.defined() || !actions.defined() || !outcomes.defined() || boards.dim()!=2 ||
        boards.size(0)<1 || boards.size(1)!=9 || actions.shape()!=n::Shape{boards.size(0)} ||
        outcomes.shape()!=actions.shape() || boards.dtype()!=n::DType::Float32 ||
        outcomes.dtype()!=n::DType::Float32 ||
        (actions.dtype()!=n::DType::Int32 && actions.dtype()!=n::DType::Int64)) {
      throw std::invalid_argument("BoardValuePolicy: invalid batch");
    }
    for (const auto* t:{&boards,&actions,&outcomes}) {
      if (t->device().backend!=compute::Backend::CPU) throw std::invalid_argument("BoardValuePolicy: CPU required");
    }
    for (const auto label:actions.to_vector<std::int64_t>()) {
      if (label<0 || label>=9) throw std::invalid_argument("BoardValuePolicy: action out of range");
    }
    for (const auto label:outcomes.to_vector<float>()) {
      if (!(label>=-1 && label<=1)) throw std::invalid_argument("BoardValuePolicy: outcome out of range");
    }
    const auto chosen=n::sum(forward(boards.detach())*n::one_hot(actions,9),{1});
    const auto loss=n::mean(n::square(chosen-outcomes.detach()));
    const double value=loss.item();
    if (!std::isfinite(value)) throw std::invalid_argument("BoardValuePolicy: nonfinite batch loss");
    optimizer_->step(n::grad(loss,optimizer_->parameters()));
    return value;
  }

  /// @brief Capture independent copies of the policy parameters.
  /// @return Named parameter store with snapshot values.
  /// @par Example
  /// @code
  /// dm::rl::BoardValuePolicy policy;
  /// const auto checkpoint=policy.snapshot();
  /// @endcode
  [[nodiscard]] nn::ParameterStore snapshot() const { return params_.clone(); }

  /// @brief Replace policy values from a compatible snapshot.
  /// @param snapshot Checkpoint with the policy's parameter names and shapes.
  /// @throws std::invalid_argument Incompatible checkpoint.
  /// @par Example
  /// @code
  /// dm::rl::BoardValuePolicy player(1), opponent(2);
  /// opponent.load_snapshot(player.snapshot());
  /// @endcode
  void load_snapshot(const nn::ParameterStore& snapshot) {
    if (snapshot.names()!=params_.names() || snapshot.dtype()!=params_.dtype()) {
      throw std::invalid_argument("BoardValuePolicy: incompatible checkpoint");
    }
    for (const auto& name:params_.names()) {
      const auto value=snapshot.at(name);
      if (value.shape()!=params_.at(name).shape() || value.device().backend!=compute::Backend::CPU) {
        throw std::invalid_argument("BoardValuePolicy: incompatible checkpoint tensor");
      }
    }
    params_.copy_from(snapshot);
  }

 private:
  /// @brief Compute all nine bounded action values.
  /// @param boards Validated board feature rows.
  /// @return Current-player action scores [batch,9].
  nn::Tensor forward(const nn::Tensor& boards) {
    return nn::dense(params_,"values",nn::dense(params_,"hidden",boards,48,nn::Activation::Tanh),
                     9,nn::Activation::Tanh);
  }

  nn::ParameterStore params_;             ///< Board value network parameters.
  std::optional<nn::Optimizer> optimizer_; ///< Adam state for outcome training.
};

} // namespace datamunge::rl
