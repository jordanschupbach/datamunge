#pragma once
// Compatibility names for canonical tensor vision modules.
#include <datamunge/nn/deformable_models.hpp>
namespace deformable {
namespace nn=datamunge::nn;
using nn::models::DeformableOutputs;
using nn::models::DeformableConv2D;
}
