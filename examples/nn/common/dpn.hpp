#pragma once
// Compatibility names for canonical dual-path tensor models.
#include "common/preact_resnet.hpp"
#include "common/resnet.hpp"
#include <datamunge/nn/dual_path_models.hpp>
namespace dpn {
namespace nn = datamunge::nn;
using nn::models::DPN;
using nn::models::DPNBlock;
using nn::models::DPNBlockOutputs;
using nn::models::DPNConfig;
using nn::models::DPNJoin;
using nn::models::DPNLayout;
using nn::models::DPNLayoutFor;
using nn::models::DPNOutputs;
using nn::models::DPNPaths;
using nn::models::DPNPreActConv;
/// @copydoc datamunge::nn::models::DPNChannelSlice
inline nn::Tensor ChannelSlice(const nn::Tensor &x, std::int64_t start, std::int64_t count) {
  return nn::models::DPNChannelSlice(x, start, count);
}
} // namespace dpn
