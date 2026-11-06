#pragma once
// Compatibility names; implementation belongs to the canonical tensor library.
#include <datamunge/nn/zfnet_xception.hpp>
#include "common/normalized_conv.hpp"
namespace xception {
namespace nn=datamunge::nn;
using datamunge::nn::models::ConvNormConfig;
using datamunge::nn::models::XceptionConfig;
using datamunge::nn::models::XceptionOutputs;
using datamunge::nn::models::XceptionSeparable;
using datamunge::nn::models::Xception;
}
