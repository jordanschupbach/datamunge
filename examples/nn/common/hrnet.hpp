#pragma once
#include "common/resnet.hpp"
#include "datamunge/nn/hrnet_models.hpp"
namespace hrnet {
namespace nn = datamunge::nn;
using nn::models::HRFuse;
using nn::models::HRNet;
using nn::models::HRNetConfig;
using nn::models::HRNetOutputs;
using nn::models::HRResize;
using resnet::ConvNormConfig;
} // namespace hrnet
