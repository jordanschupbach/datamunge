#pragma once
// Compatibility names for canonical tensor mobile inverted bottlenecks.
#include <datamunge/nn/mbconv_models.hpp>
#include "common/mobilenet.hpp"
namespace mbconv {
namespace nn=datamunge::nn;
using datamunge::nn::models::ConvBN;
using datamunge::nn::models::MakeDivisible;
using datamunge::nn::models::MobileContext;
using datamunge::nn::models::MobileConv;
using datamunge::nn::models::MobileNetOutputs;
using datamunge::nn::models::BlockArgs;
using datamunge::nn::models::MBConvOutputs;
using datamunge::nn::models::MnasNetConfig;
using datamunge::nn::models::EfficientNetScaling;
using datamunge::nn::models::EfficientNetConfig;
using datamunge::nn::models::EfficientNetV2Variant;
using datamunge::nn::models::EfficientNetV2Config;
using datamunge::nn::models::DecodeBlock;
using datamunge::nn::models::EncodeBlock;
using datamunge::nn::models::RoundFilters;
using datamunge::nn::models::Swish;
using datamunge::nn::models::DropConnect;
using datamunge::nn::models::MBConv;
using datamunge::nn::models::MnasNetBlocks;
using datamunge::nn::models::MnasNet;
using datamunge::nn::models::EfficientNetCoefficients;
using datamunge::nn::models::EfficientNetBlocks;
using datamunge::nn::models::EfficientNet;
using datamunge::nn::models::EfficientNetV2Spec;
using datamunge::nn::models::EfficientNetV2Blocks;
using datamunge::nn::models::EfficientNetV2;
}
