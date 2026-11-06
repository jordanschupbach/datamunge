#pragma once
// Compatibility names for canonical coupled-gate highway tensor layers.
#include <datamunge/nn/highway_models.hpp>
namespace highway {
namespace nn=datamunge::nn;
using datamunge::nn::models::HighwayOutputs;
using datamunge::nn::models::HighwayMix;
using datamunge::nn::models::HighwayDense;
using datamunge::nn::models::HighwayConv2D;
}
