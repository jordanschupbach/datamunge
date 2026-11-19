/// @file
/// @brief Supervised categorical policies for imitation learning.
#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/losses.hpp"
#include "datamunge/nn/optim.hpp"

namespace datamunge::rl {

/// @brief Fit an action classifier to expert demonstrations with Adam.
/// @par Example
/// @code
/// dm::rl::ImitationPolicy policy(2,16,4);
/// std::print("{}\n",policy.predict(dm::nn::eye(2)).size(1)); // 4
/// @endcode
class ImitationPolicy {
 public:
  /// @brief Initialize one tanh hidden layer and a categorical action head.
  /// @param inputs Observation feature width.
  /// @param hidden Hidden layer width.
  /// @param actions Number of action labels.
  /// @param seed Parameter initialization seed, default 44501.
  /// @param rate Positive finite Adam learning rate, default 0.012.
  /// @throws std::invalid_argument Invalid widths or rate.
  /// @par Example
  /// @code
  /// dm::rl::ImitationPolicy policy(2,16,4,44501);
  /// @endcode
  ImitationPolicy(std::int64_t inputs,std::int64_t hidden,std::int64_t actions,
                  std::uint64_t seed=44501,double rate=.012)
      : inputs_(inputs),hidden_(hidden),actions_(actions),params_(seed) {
    if (inputs<1 || hidden<1 || actions<1 || !(rate>0 && std::isfinite(rate))) {
      throw std::invalid_argument("ImitationPolicy: invalid widths or rate");
    }
    (void)predict(nn::zeros({1,inputs}));
    optimizer_.emplace(params_,rate,nn::OptimizerConfig{.kind=nn::OptimizerKind::Adam});
  }

  /// @brief Predict detached logits for each observation.
  /// @param observations CPU Float32 rows [batch,inputs], with positive batch size.
  /// @return Float32 logits [batch,actions].
  /// @throws std::invalid_argument Invalid observation geometry or storage.
  /// @par Example
  /// @code
  /// dm::rl::ImitationPolicy policy(2,16,4);
  /// std::print("{}\n",policy.predict(dm::nn::eye(2)).size(0)); // 2
  /// @endcode
  nn::Tensor predict(const nn::Tensor& observations) {
    check(observations);
    const nn::NoGradGuard guard;
    return forward(observations);
  }

  /// @brief Update the policy with one batch of expert action labels.
  /// @param observations CPU Float32 rows [batch,inputs], with positive batch size.
  /// @param expert CPU Int32 or Int64 action labels [batch].
  /// @return Mean negative log-likelihood before updating the parameters.
  /// @throws std::invalid_argument Invalid observations or action labels.
  /// @par Example
  /// @code
  /// dm::rl::ImitationPolicy policy(2,16,4);
  /// const double loss=policy.train(dm::nn::eye(2),dm::nn::from_vector<int>({0,1}));
  /// @endcode
  double train(const nn::Tensor& observations,const nn::Tensor& expert) {
    check(observations);
    if (!expert.defined() || expert.shape()!=nn::Shape{observations.size(0)} ||
        expert.device().backend!=compute::Backend::CPU ||
        (expert.dtype()!=nn::DType::Int32 && expert.dtype()!=nn::DType::Int64)) {
      throw std::invalid_argument("ImitationPolicy: invalid labels");
    }
    for (const auto label:expert.to_vector<std::int64_t>()) {
      if (label<0 || label>=actions_) throw std::invalid_argument("ImitationPolicy: action out of range");
    }
    const auto loss=nn::sparse_softmax_cross_entropy(forward(observations.detach()),expert);
    const double value=loss.item();
    optimizer_->step(nn::grad(loss,optimizer_->parameters()));
    return value;
  }

 private:
  /// @brief Validate observation shape and storage.
  /// @param observations Input feature rows.
  /// @throws std::invalid_argument Invalid dimensions, type or device.
  void check(const nn::Tensor& observations) const {
    if (!observations.defined() || observations.dim()!=2 || observations.size(0)<1 ||
        observations.size(1)!=inputs_ || observations.dtype()!=nn::DType::Float32 ||
        observations.device().backend!=compute::Backend::CPU) {
      throw std::invalid_argument("ImitationPolicy: expected Float32 CPU observations");
    }
  }

  /// @brief Apply both classifier layers.
  /// @param observations Validated input features.
  /// @return Action logits.
  nn::Tensor forward(const nn::Tensor& observations) {
    return nn::dense(params_,"head",
        nn::dense(params_,"hidden",observations,hidden_,nn::Activation::Tanh),actions_);
  }

  std::int64_t inputs_;                    ///< Observation width.
  std::int64_t hidden_;                    ///< Tanh layer width.
  std::int64_t actions_;                   ///< Number of action classes.
  nn::ParameterStore params_;              ///< Weights and biases.
  std::optional<nn::Optimizer> optimizer_;  ///< Adam constructed after layer initialization.
};

} // namespace datamunge::rl
