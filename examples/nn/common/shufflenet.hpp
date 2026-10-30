#pragma once
// Compatibility names for canonical ShuffleNet tensor models.
#include <datamunge/nn/shufflenet_models.hpp>
#include "common/grouped.hpp"
namespace shufflenet {
namespace nn=datamunge::nn;
using nn::models::ReshapeChannels;
using nn::models::ChannelShuffle;
using nn::models::PaperChannelShuffle;
using nn::models::PadSpatial;
using nn::models::ShuffleContext;
using nn::models::ShuffleConv;
using nn::models::AveragePoolWithPadding;
using nn::models::ShuffleV1Unit;
using nn::models::ShuffleV2Unit;
using nn::models::ShuffleNetConfig;
using nn::models::ShuffleNetOutputs;
using nn::models::ShuffleNet;
}
