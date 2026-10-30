#pragma once
// Compatibility names for canonical tensor mobile inverted bottlenecks.
#include <datamunge/nn/mbconv_models.hpp>
#include "common/mobilenet.hpp"
namespace mbconv {
namespace nn=datamunge::nn;
using nn::models::ConvBN;
using nn::models::MakeDivisible;
using nn::models::MobileContext;
using nn::models::MobileConv;
using nn::models::MobileNetOutputs;
using nn::models::BlockArgs;
using nn::models::MBConvOutputs;
using nn::models::MnasNetConfig;
using nn::models::EfficientNetScaling;
using nn::models::EfficientNetConfig;
using nn::models::EfficientNetV2Variant;
using nn::models::EfficientNetV2Config;
using nn::models::DecodeBlock;
using nn::models::EncodeBlock;
using nn::models::RoundFilters;
using nn::models::Swish;
using nn::models::DropConnect;
using nn::models::MBConv;
using nn::models::MnasNetBlocks;
using nn::models::MnasNet;
using nn::models::EfficientNetCoefficients;
using nn::models::EfficientNetBlocks;
using nn::models::EfficientNet;
using nn::models::EfficientNetV2Spec;
using nn::models::EfficientNetV2Blocks;
using nn::models::EfficientNetV2;
}
