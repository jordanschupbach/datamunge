#pragma once
// Compatibility names for canonical MobileNet tensor models.
#include <datamunge/nn/mobilenet_models.hpp>
namespace mobilenet {
namespace nn=datamunge::nn;
using datamunge::nn::models::MobileLayer;
using datamunge::nn::models::MobileContext;
using datamunge::nn::models::ContextBatchNorm;
using datamunge::nn::models::MobileConv;
using datamunge::nn::models::ConvBN;
using datamunge::nn::models::MobileNetV1Config;
using datamunge::nn::models::MobileNetOutputs;
using datamunge::nn::models::MobileNetV1;
using datamunge::nn::models::MakeDivisible;
using datamunge::nn::models::InvertedResidualOutputs;
using datamunge::nn::models::InvertedResidual;
using datamunge::nn::models::MobileNetV2Config;
using datamunge::nn::models::MobileNetV2;
using datamunge::nn::models::HardSigmoid;
using datamunge::nn::models::HardSwish;
using datamunge::nn::models::SqueezeExcitation;
using datamunge::nn::models::SqueezeExcite;
using datamunge::nn::models::MobileNetV3Block;
using datamunge::nn::models::MobileNetV3Blocks;
using datamunge::nn::models::MobileNetV3Config;
using datamunge::nn::models::MobileNetV3;
}
