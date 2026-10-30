#pragma once
// Compatibility names for canonical GhostNet tensor models.
#include <datamunge/nn/ghostnet_models.hpp>
#include "common/mobilenet.hpp"
namespace ghostnet {
namespace nn=datamunge::nn;
using nn::models::HardSigmoid;
using nn::models::MakeDivisible;
using nn::models::MobileContext;
using nn::models::MobileNetOutputs;
using nn::models::GhostConvBN;
using nn::models::GhostModuleOutputs;
using nn::models::GhostModule;
using nn::models::GhostBlock;
using nn::models::GhostNetBlocks;
using nn::models::GhostBottleneck;
using nn::models::GhostNetConfig;
using nn::models::GhostNet;
/// @copydoc nn::models::GhostPadSpatial
inline nn::Tensor PadSpatial(const nn::Tensor& x,int64_t pad){return nn::models::GhostPadSpatial(x,pad);}
}
