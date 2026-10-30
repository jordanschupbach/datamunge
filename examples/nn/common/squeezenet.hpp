#pragma once
// Compatibility names for canonical Fire and SqueezeNet tensor models.
#include <datamunge/nn/squeezenet_models.hpp>
namespace squeezenet {
namespace nn=datamunge::nn;
using nn::models::FireOutputs;
using nn::models::Fire;
using nn::models::CeilMaxPool;
using nn::models::SqueezeNetConfig;
using nn::models::SqueezeNetOutputs;
using nn::models::SqueezeNet;
}
