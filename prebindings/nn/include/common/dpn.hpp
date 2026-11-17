#pragma once
// Compatibility names for canonical dual-path tensor models.
#include "common/preact_resnet.hpp"
#include "common/resnet.hpp"
#include <datamunge/nn/dual_path_models.hpp>
namespace dpn {
namespace nn = datamunge::nn;
using datamunge::nn::models::DPN;
using datamunge::nn::models::DPNBlock;
using datamunge::nn::models::DPNBlockOutputs;
using datamunge::nn::models::DPNConfig;
using datamunge::nn::models::DPNJoin;
using datamunge::nn::models::DPNLayout;
using datamunge::nn::models::DPNLayoutFor;
using datamunge::nn::models::DPNOutputs;
using datamunge::nn::models::DPNPaths;
using datamunge::nn::models::DPNPreActConv;
/// @copydoc datamunge::nn::models::DPNChannelSlice
inline nn::Tensor ChannelSlice(const nn::Tensor &x, std::int64_t start, std::int64_t count) {
  return nn::models::DPNChannelSlice(x, start, count);
}
} // namespace dpn
