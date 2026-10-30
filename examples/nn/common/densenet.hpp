#pragma once
// Compatibility names for canonical dense-concatenation tensor definitions.
#include "common/preact_resnet.hpp"
#include <datamunge/nn/preactivation_densenet_models.hpp>
namespace densenet {
namespace nn = datamunge::nn;
using nn::models::DenseGrowthLayer;
using nn::models::DenseGrowthOutputs;
using nn::models::DenseNet;
using nn::models::DenseNetConfig;
using nn::models::DenseNetOutputs;
using nn::models::DenseTransition;
} // namespace densenet
