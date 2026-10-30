#pragma once
// Compatibility names for canonical coupled-gate highway tensor layers.
#include <datamunge/nn/highway_models.hpp>
namespace highway {
namespace nn=datamunge::nn;
using nn::models::HighwayOutputs;
using nn::models::HighwayMix;
using nn::models::HighwayDense;
using nn::models::HighwayConv2D;
}
