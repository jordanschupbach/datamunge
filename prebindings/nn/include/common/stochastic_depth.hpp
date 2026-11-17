#pragma once
// Compatibility names for canonical DropPath tensor models.
#include <datamunge/nn/stochastic_residual_models.hpp>
#include "common/resnet.hpp"
namespace stochastic_depth {
namespace nn=datamunge::nn;
using datamunge::nn::models::DropPathScaling;
using datamunge::nn::models::SurvivalProbabilities;
using datamunge::nn::models::DropPathOutputs;
using datamunge::nn::models::DropPath;
using datamunge::nn::models::CifarIdentityShortcut;
using datamunge::nn::models::StochasticCifarConfig;
using datamunge::nn::models::StochasticCifarOutputs;
using datamunge::nn::models::StochasticCifarResNet;
}
