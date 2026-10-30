#pragma once
// Compatibility names for canonical MobileNet tensor models.
#include <datamunge/nn/mobilenet_models.hpp>
namespace mobilenet {
namespace nn=datamunge::nn;
using nn::models::MobileLayer;
using nn::models::MobileContext;
using nn::models::ContextBatchNorm;
using nn::models::MobileConv;
using nn::models::ConvBN;
using nn::models::MobileNetV1Config;
using nn::models::MobileNetOutputs;
using nn::models::MobileNetV1;
using nn::models::MakeDivisible;
using nn::models::InvertedResidualOutputs;
using nn::models::InvertedResidual;
using nn::models::MobileNetV2Config;
using nn::models::MobileNetV2;
using nn::models::HardSigmoid;
using nn::models::HardSwish;
using nn::models::SqueezeExcitation;
using nn::models::SqueezeExcite;
using nn::models::MobileNetV3Block;
using nn::models::MobileNetV3Blocks;
using nn::models::MobileNetV3Config;
using nn::models::MobileNetV3;
}
