#pragma once
// Compatibility names; canonical tensor FPN lives in the library.
#include <datamunge/nn/feature_pyramid_models.hpp>
namespace fpn {
using FPNOutputs=datamunge::nn::models::FeaturePyramidOutputs;
using datamunge::nn::models::FeaturePyramid;
}
