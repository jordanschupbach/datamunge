#pragma once
// Compatibility names for canonical dense-concatenation tensor definitions.
#include "common/preact_resnet.hpp"
#include <datamunge/nn/preactivation_densenet_models.hpp>
namespace densenet {
namespace nn = datamunge::nn;
using datamunge::nn::models::DenseGrowthLayer;
using datamunge::nn::models::DenseGrowthOutputs;
using datamunge::nn::models::DenseNet;
using datamunge::nn::models::DenseNetConfig;
using datamunge::nn::models::DenseNetOutputs;
using datamunge::nn::models::DenseTransition;
} // namespace densenet
