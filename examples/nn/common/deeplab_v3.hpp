#pragma once
#include "common/deeplab.hpp"
#include "common/normalized_conv.hpp"
#include "datamunge/nn/deeplab_v3_models.hpp"
namespace deeplab {
using nn::models::AtrousSeparableBlock;
using nn::models::DeepLabV3;
using nn::models::DeepLabV3Config;
using nn::models::DeepLabV3Outputs;
using nn::models::ImagePoolBranch;
using normalized_conv::ConvNormConfig;
using normalized_conv::NormalizedConv;
} // namespace deeplab
