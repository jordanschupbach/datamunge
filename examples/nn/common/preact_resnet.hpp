#pragma once
// Compatibility names for canonical preactivation tensor definitions.
#include "common/resnet.hpp"
#include <datamunge/nn/preactivation_densenet_models.hpp>
namespace preact {
namespace nn = datamunge::nn;
using NormConfig = datamunge::nn::models::PreactivationNormConfig;
using datamunge::nn::models::PreActBottleneck;
using datamunge::nn::models::PreActivate;
using datamunge::nn::models::PreActResNet;
using datamunge::nn::models::PreActResNetConfig;
using datamunge::nn::models::PreActResNetOutputs;
using datamunge::nn::models::RawResidualConv;
/// @copydoc datamunge::nn::models::PreactivationNormalize
inline nn::Tensor Normalize(nn::ParameterStore &p, nn::ParameterStore &state,
                            const std::string &name, const nn::Tensor &input,
                            const NormConfig &cfg) {
  return nn::models::PreactivationNormalize(p, state, name, input, cfg);
}
} // namespace preact
