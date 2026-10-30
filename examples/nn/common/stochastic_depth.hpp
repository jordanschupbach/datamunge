#pragma once
// Compatibility names for canonical DropPath tensor models.
#include <datamunge/nn/stochastic_residual_models.hpp>
#include "common/resnet.hpp"
namespace stochastic_depth {
namespace nn=datamunge::nn;
using nn::models::DropPathScaling;
using nn::models::SurvivalProbabilities;
using nn::models::DropPathOutputs;
using nn::models::DropPath;
using nn::models::CifarIdentityShortcut;
using nn::models::StochasticCifarConfig;
using nn::models::StochasticCifarOutputs;
using nn::models::StochasticCifarResNet;
}
