#pragma once
// Compatibility names for canonical GhostNet tensor models.
#include <datamunge/nn/ghostnet_models.hpp>
#include "common/mobilenet.hpp"
namespace ghostnet {
namespace nn=datamunge::nn;
using datamunge::nn::models::HardSigmoid;
using datamunge::nn::models::MakeDivisible;
using datamunge::nn::models::MobileContext;
using datamunge::nn::models::MobileNetOutputs;
using datamunge::nn::models::GhostConvBN;
using datamunge::nn::models::GhostModuleOutputs;
using datamunge::nn::models::GhostModule;
using datamunge::nn::models::GhostBlock;
using datamunge::nn::models::GhostNetBlocks;
using datamunge::nn::models::GhostBottleneck;
using datamunge::nn::models::GhostNetConfig;
using datamunge::nn::models::GhostNet;
/// @copydoc nn::models::GhostPadSpatial
inline nn::Tensor PadSpatial(const nn::Tensor& x,int64_t pad){return nn::models::GhostPadSpatial(x,pad);}
}
