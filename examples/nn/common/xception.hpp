#pragma once
// Compatibility names; implementation belongs to the canonical tensor library.
#include <datamunge/nn/zfnet_xception.hpp>
#include "common/normalized_conv.hpp"
namespace xception {
namespace nn=datamunge::nn;
using nn::models::ConvNormConfig;
using nn::models::XceptionConfig;
using nn::models::XceptionOutputs;
using nn::models::XceptionSeparable;
using nn::models::Xception;
}
