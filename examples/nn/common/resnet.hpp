#pragma once
// Compatibility names for canonical tensor ResNet definitions.
#include <datamunge/nn/resnet_models.hpp>
namespace resnet {
namespace nn = datamunge::nn;
using ConvNormConfig = nn::models::ResNetNormConfig;
using nn::models::ResNet;
using nn::models::ResNetBlock;
using nn::models::ResNetBlockCounts;
using nn::models::ResNetBlockOutputs;
using nn::models::ResNetConfig;
using nn::models::ResNetConv;
using nn::models::ResNetOutputs;
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
