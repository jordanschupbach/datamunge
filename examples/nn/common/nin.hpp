#pragma once
// Compatibility names for canonical tensor architecture definitions.
#include <datamunge/nn/classic_cnn_models.hpp>
namespace nin {
namespace nn=datamunge::nn;
using nn::models::MlpConv;
using nn::models::NinConfig;
using nn::models::NinOutputs;
using nn::models::NetworkInNetwork;
}
