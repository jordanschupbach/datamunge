#pragma once
#include "common/deeplab.hpp"
#include "common/normalized_conv.hpp"
#include "datamunge/nn/deeplab_v3_models.hpp"
namespace deeplab {
using datamunge::nn::models::AtrousSeparableBlock;
using datamunge::nn::models::DeepLabV3;
using datamunge::nn::models::DeepLabV3Config;
using datamunge::nn::models::DeepLabV3Outputs;
using datamunge::nn::models::ImagePoolBranch;
using normalized_conv::ConvNormConfig;
using normalized_conv::NormalizedConv;
} // namespace deeplab
