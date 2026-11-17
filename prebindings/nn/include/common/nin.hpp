#pragma once
// Compatibility names for canonical tensor architecture definitions.
#include <datamunge/nn/classic_cnn_models.hpp>
namespace nin {
namespace nn=datamunge::nn;
using datamunge::nn::models::MlpConv;
using datamunge::nn::models::NinConfig;
using datamunge::nn::models::NinOutputs;
using datamunge::nn::models::NetworkInNetwork;
}
