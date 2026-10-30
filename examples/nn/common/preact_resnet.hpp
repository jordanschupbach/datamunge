#pragma once
// Compatibility names for canonical preactivation tensor definitions.
#include "common/resnet.hpp"
#include <datamunge/nn/preactivation_densenet_models.hpp>
namespace preact {
namespace nn = datamunge::nn;
using NormConfig = nn::models::PreactivationNormConfig;
using nn::models::PreActBottleneck;
using nn::models::PreActivate;
using nn::models::PreActResNet;
using nn::models::PreActResNetConfig;
using nn::models::PreActResNetOutputs;
using nn::models::RawResidualConv;
/// @copydoc datamunge::nn::models::PreactivationNormalize
inline nn::Tensor Normalize(nn::ParameterStore &p, nn::ParameterStore &state,
                            const std::string &name, const nn::Tensor &input,
                            const NormConfig &cfg) {
  return nn::models::PreactivationNormalize(p, state, name, input, cfg);
}
} // namespace preact
