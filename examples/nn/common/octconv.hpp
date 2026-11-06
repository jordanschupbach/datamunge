#pragma once
// Compatibility names for canonical tensor octave convolution models.
#include <datamunge/nn/octave_models.hpp>
#include "common/resnet.hpp"
namespace octconv {
namespace nn=datamunge::nn;
using datamunge::nn::models::OctaveMaps;
using datamunge::nn::models::OctaveCost;
using datamunge::nn::models::OctResNetConfig;
using datamunge::nn::models::OctResNetOutputs;
using datamunge::nn::models::LowChannels;
using datamunge::nn::models::OctaveConv;
using datamunge::nn::models::OctaveNorm;
using datamunge::nn::models::OctResNet;
}
