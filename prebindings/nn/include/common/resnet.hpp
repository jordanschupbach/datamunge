#pragma once
// Compatibility names for canonical tensor ResNet definitions.
#include <datamunge/nn/resnet_models.hpp>
namespace resnet {
namespace nn = datamunge::nn;
using ConvNormConfig = datamunge::nn::models::ResNetNormConfig;
using datamunge::nn::models::ResNet;
using datamunge::nn::models::ResNetBlock;
using datamunge::nn::models::ResNetBlockCounts;
using datamunge::nn::models::ResNetBlockOutputs;
using datamunge::nn::models::ResNetConfig;
using datamunge::nn::models::ResNetConv;
using datamunge::nn::models::ResNetOutputs;
/// @copydoc datamunge::nn::models::ResNetNormalizedConv
inline nn::Tensor NormalizedConv(nn::ParameterStore &p, nn::ParameterStore &state,
                                 const std::string &name, const nn::Tensor &input,
                                 std::int64_t out, std::int64_t kernel,
                                 const ConvNormConfig &cfg, std::int64_t stride,
                                 std::int64_t pad, bool relu) {
  return nn::models::ResNetNormalizedConv(p, state, name, input, out, kernel, cfg, stride, pad,
                                          relu);
}
} // namespace resnet
