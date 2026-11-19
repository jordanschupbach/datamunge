/// @file
/// @brief Inverse-dynamics features and forward prediction for discrete transitions.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/losses.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/optim.hpp"

namespace datamunge::rl {

/// @brief Learn discrete-state embeddings from forward and inverse dynamics.
///
/// A shared tanh encoder maps one-hot states into feature vectors.
/// The inverse head learns actions from pairs of features. The forward
/// head predicts successor features from current features and actions.
/// Its target is detached, while both features in the inverse head
/// contribute gradients to the shared encoder.
/// @par Example
/// @code
/// dm::rl::IntrinsicCuriosity model(3,2,4,5);
/// const auto output=model.evaluate(dm::nn::from_vector<int>({0,1,2},{1,3}));
/// std::print("{}\n",output[2].size(1)); // 4
/// @endcode
/// @par References
/// - Pathak et al. (2017), Curiosity-driven Exploration by Self-supervised Prediction.
///   https://proceedings.mlr.press/v70/pathak17a.html
class IntrinsicCuriosity {
 public:
  /// @brief Initialize the shared encoder and both prediction heads.
  /// @param states Positive count of discrete states.
  /// @param actions Positive count of discrete actions.
  /// @param features Positive embedding width.
  /// @param hidden Positive hidden width in each prediction head.
  /// @param forward_weight Forward loss weight in [0,1], default 0.2. Zero trains inverse dynamics alone.
  /// @param seed Parameter initializer seed, default 43601.
  /// @param rate Positive finite Adam learning rate, default 0.003.
  /// @throws std::invalid_argument Invalid network sizes, loss weight or learning rate.
  /// @par Example
  /// @code
  /// dm::rl::IntrinsicCuriosity model(49,4,12,24,0.);
  /// @endcode
  IntrinsicCuriosity(std::int64_t states,std::int64_t actions,std::int64_t features,
                     std::int64_t hidden,double forward_weight=.2,std::uint64_t seed=43601,
                     double rate=.003)
      : states_(states),actions_(actions),features_(features),hidden_(hidden),
        forward_weight_(forward_weight),params_(seed) {
    if (states<1 || actions<1 || features<1 || hidden<1 ||
        !(forward_weight>=0 && forward_weight<=1) || !(rate>0 && std::isfinite(rate))) {
      throw std::invalid_argument("IntrinsicCuriosity: invalid configuration");
    }
    (void)forward(nn::zeros({1,3},nn::DType::Int32));
    optimizer_.emplace(params_,rate,nn::OptimizerConfig{.kind=nn::OptimizerKind::Adam});
  }

  /// @brief Compute detached intrinsic rewards, predicted features, targets and inverse logits.
  /// @param transitions CPU Int32 or Int64 triples [batch,3] of (state,action,successor).
  /// @return Rewards [batch], forward predictions and target features [batch,features],
  /// and inverse action logits [batch,actions], all Float32.
  /// @throws std::invalid_argument Invalid transition shape, storage or IDs.
  /// @par Example
  /// @code
  /// dm::rl::IntrinsicCuriosity model(3,2,4,5);
  /// const auto out=model.evaluate(dm::nn::from_vector<int>({0,1,2},{1,3}));
  /// std::print("{}\n",out[3].size(1)); // 2
  /// @endcode
  std::array<nn::Tensor,4> evaluate(const nn::Tensor& transitions) {
    check(transitions);
    const nn::NoGradGuard guard;
    return forward(transitions);
  }

  /// @brief Encode state IDs using the current learned representation.
  /// @param ids Nonempty CPU Int32 or Int64 vector of valid state IDs.
  /// @return Detached Float32 feature rows [count,features].
  /// @throws std::invalid_argument Invalid IDs or storage.
  /// @par Example
  /// @code
  /// dm::rl::IntrinsicCuriosity model(3,2,4,5);
  /// const auto phi=model.embed(dm::nn::from_vector<int>({0,2}));
  /// std::print("{}\n",phi.size(0)); // 2
  /// @endcode
  nn::Tensor embed(const nn::Tensor& ids) {
    if (!valid_indices(ids) || ids.dim()!=1 || ids.size(0)<1) {
      throw std::invalid_argument("IntrinsicCuriosity: expected state ID vector");
    }
    const auto values=ids.to_vector<std::int64_t>();
    if (!std::all_of(values.begin(),values.end(),[&](auto id){return id>=0 && id<states_;})) {
      throw std::invalid_argument("IntrinsicCuriosity: state ID out of range");
    }
    const nn::NoGradGuard guard;
    return encode(ids);
  }

  /// @brief Take one Adam step on the weighted forward and inverse losses.
  /// @param transitions Nonempty CPU Int32 or Int64 transition triples [batch,3].
  /// @return Combined mean loss before the update.
  /// @throws std::invalid_argument Invalid transition shape, storage or IDs.
  /// @par Example
  /// @code
  /// dm::rl::IntrinsicCuriosity model(3,2,4,5);
  /// const auto loss=model.train(dm::nn::from_vector<int>({0,1,2},{1,3}));
  /// @endcode
  double train(const nn::Tensor& transitions) {
    check(transitions);
    const auto outputs=forward(transitions);
    const auto actions=column(transitions,1);
    const auto loss=forward_weight_*nn::mean(outputs[0])+
        (1.-forward_weight_)*nn::sparse_softmax_cross_entropy(outputs[3],actions);
    const double value=loss.item();
    optimizer_->step(nn::grad(loss,optimizer_->parameters()));
    return value;
  }

 private:
  /// @brief Check integer CPU storage.
  /// @param value Tensor to inspect.
  /// @return Whether the storage holds supported IDs.
  static bool valid_indices(const nn::Tensor& value) {
    return value.defined() && (value.dtype()==nn::DType::Int32 || value.dtype()==nn::DType::Int64) &&
        value.device().backend==compute::Backend::CPU;
  }

  /// @brief Validate transition geometry and all categorical IDs.
  /// @param rows Candidate transition tensor.
  /// @throws std::invalid_argument Invalid triples or out-of-range IDs.
  void check(const nn::Tensor& rows) const {
    if (!valid_indices(rows) || rows.dim()!=2 || rows.size(0)<1 || rows.size(1)!=3) {
      throw std::invalid_argument("IntrinsicCuriosity: expected transition triples");
    }
    const auto data=rows.to_vector<std::int64_t>();
    for (std::size_t i=0;i<data.size();i+=3) {
      if (data[i]<0 || data[i]>=states_ || data[i+1]<0 || data[i+1]>=actions_ ||
          data[i+2]<0 || data[i+2]>=states_) {
        throw std::invalid_argument("IntrinsicCuriosity: transition ID out of range");
      }
    }
  }

  /// @brief Extract one categorical column as a vector.
  /// @param rows Transition matrix.
  /// @param index Column index.
  /// @return Vector of corresponding IDs.
  static nn::Tensor column(const nn::Tensor& rows,std::int64_t index) {
    return nn::slice(rows,{0,index},{rows.size(0),1}).reshape({rows.size(0)});
  }

  /// @brief Apply the shared encoder to validated state IDs.
  /// @param ids State IDs.
  /// @return State features.
  nn::Tensor encode(const nn::Tensor& ids) {
    return nn::dense(params_,"encode",nn::one_hot(ids,states_),features_,nn::Activation::Tanh);
  }

  /// @brief Apply both prediction heads to validated triples.
  /// @param rows Transition triples.
  /// @return Bonus, predicted successor features, encoded successor features and inverse logits.
  std::array<nn::Tensor,4> forward(const nn::Tensor& rows) {
    const auto current=encode(column(rows,0));
    const auto next=encode(column(rows,2));
    const auto actions=nn::one_hot(column(rows,1),actions_);
    const auto forward_hidden=nn::dense(params_,"forward/hidden",nn::concat({current,actions},1),
                                       hidden_,nn::Activation::Relu);
    const auto predicted=nn::dense(params_,"forward/out",forward_hidden,features_);
    const auto inverse_hidden=nn::dense(params_,"inverse/hidden",nn::concat({current,next},1),
                                       hidden_,nn::Activation::Relu);
    const auto logits=nn::dense(params_,"inverse/out",inverse_hidden,actions_);
    return {.5*nn::sum(nn::square(predicted-next.detach()),{1}),predicted,next,logits};
  }

  std::int64_t states_;                     ///< Categorical state count.
  std::int64_t actions_;                    ///< Categorical action count.
  std::int64_t features_;                   ///< State embedding width.
  std::int64_t hidden_;                     ///< Hidden width for both heads.
  double forward_weight_;                  ///< Relative weight of forward prediction.
  nn::ParameterStore params_;              ///< Shared encoder and prediction head weights.
  std::optional<nn::Optimizer> optimizer_;  ///< Adam initialized after all layers.
};

} // namespace datamunge::rl
