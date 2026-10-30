#pragma once
// Compatibility names for canonical tensor octave convolution models.
#include <datamunge/nn/octave_models.hpp>
#include "common/resnet.hpp"
namespace octconv {
namespace nn=datamunge::nn;
using nn::models::OctaveMaps;
using nn::models::OctaveCost;
using nn::models::OctResNetConfig;
using nn::models::OctResNetOutputs;
using nn::models::LowChannels;
using nn::models::OctaveConv;
using nn::models::OctaveNorm;
using nn::models::OctResNet;
}
