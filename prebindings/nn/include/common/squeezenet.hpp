#pragma once
// Compatibility names for canonical Fire and SqueezeNet tensor models.
#include <datamunge/nn/squeezenet_models.hpp>
namespace squeezenet {
namespace nn=datamunge::nn;
using datamunge::nn::models::FireOutputs;
using datamunge::nn::models::Fire;
using datamunge::nn::models::CeilMaxPool;
using datamunge::nn::models::SqueezeNetConfig;
using datamunge::nn::models::SqueezeNetOutputs;
using datamunge::nn::models::SqueezeNet;
}
