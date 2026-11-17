#pragma once
// Compatibility names for canonical tensor vision modules.
#include <datamunge/nn/deformable_models.hpp>
namespace deformable {
namespace nn=datamunge::nn;
using datamunge::nn::models::DeformableOutputs;
using datamunge::nn::models::DeformableConv2D;
}
