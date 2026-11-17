#pragma once
// Compatibility names for canonical ShuffleNet tensor models.
#include <datamunge/nn/shufflenet_models.hpp>
#include "common/grouped.hpp"
namespace shufflenet {
namespace nn=datamunge::nn;
using datamunge::nn::models::ReshapeChannels;
using datamunge::nn::models::ChannelShuffle;
using datamunge::nn::models::PaperChannelShuffle;
using datamunge::nn::models::PadSpatial;
using datamunge::nn::models::ShuffleContext;
using datamunge::nn::models::ShuffleConv;
using datamunge::nn::models::AveragePoolWithPadding;
using datamunge::nn::models::ShuffleV1Unit;
using datamunge::nn::models::ShuffleV2Unit;
using datamunge::nn::models::ShuffleNetConfig;
using datamunge::nn::models::ShuffleNetOutputs;
using datamunge::nn::models::ShuffleNet;
}
