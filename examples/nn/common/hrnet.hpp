#pragma once
#include "common/resnet.hpp"
#include "datamunge/nn/hrnet_models.hpp"
namespace hrnet {
namespace nn = datamunge::nn;
using datamunge::nn::models::HRFuse;
using datamunge::nn::models::HRNet;
using datamunge::nn::models::HRNetConfig;
using datamunge::nn::models::HRNetOutputs;
using datamunge::nn::models::HRResize;
using resnet::ConvNormConfig;
} // namespace hrnet
