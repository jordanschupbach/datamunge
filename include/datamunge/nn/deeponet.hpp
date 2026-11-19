#pragma once

/// @file deeponet.hpp
/// @brief Branch and trunk networks for a scalar-valued learned operator.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

namespace datamunge::nn {

/// @brief Branch and trunk embeddings and their paired scalar output.
/// @par Example
/// @code
/// dm::nn::DeepONetOutput output{};
/// @endcode
struct DeepONetOutput {
  Tensor prediction; ///< Scalar result per paired input row, shaped [batch,1].
  Tensor branch; ///< Initial function embeddings [batch,code].
  Tensor trunk; ///< Query embeddings [batch,code].
};

/// @brief Evaluate a scalar DeepONet using paired sensor and query rows.
/// Both towers have one tanh layer. Their embeddings are combined by a
/// row-wise inner product and one learned scalar bias.
/// @param model Store for both towers and their shared output bias.
/// @param sensors Floating [batch,sensors] sampled function values.
/// @param query Floating [batch,dimensions] query coordinates.
/// @param width Positive hidden width, default 32.
/// @param code Positive embedding width, default 16.
/// @return Predictions and both embeddings with live parameter gradients.
/// @throws std::invalid_argument For incompatible shape, data type, device or width.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto result=dm::nn::deeponet(model,dm::nn::zeros({2,8}),dm::nn::zeros({2,2}));
/// @endcode
/// @par References
/// - Lu et al., Learning nonlinear operators via DeepONet, https://arxiv.org/abs/1910.03193.
inline DeepONetOutput deeponet(ParameterStore& model,const Tensor& sensors,const Tensor& query,
                               std::int64_t width=32,std::int64_t code=16) {
  if (!sensors.defined() || !query.defined() || sensors.dim()!=2 || query.dim()!=2 ||
      !sensors.numel() || !query.numel() || sensors.size(0)!=query.size(0) ||
      sensors.dtype()!=model.dtype() || query.dtype()!=model.dtype() ||
      sensors.device().backend!=compute::Backend::CPU || query.device().backend!=compute::Backend::CPU ||
      width<=0 || code<=0)
    throw std::invalid_argument("deeponet: expected aligned CPU sensor and query matrices");
  const auto embed=[&](const std::string& name,const Tensor& input) {
    return dense(model,name+"/code",dense(model,name+"/hidden",input,width,Activation::Tanh),code);
  };
  auto branch=embed("branch",sensors);
  auto trunk=embed("trunk",query);
  auto prediction=sum(branch*trunk,{1},true)+model.get("bias",{},init::zeros());
  return {prediction,branch,trunk};
}

} // namespace datamunge::nn
