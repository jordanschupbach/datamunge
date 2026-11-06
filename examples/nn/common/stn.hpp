#pragma once
// Compatibility names for canonical tensor affine localization and sampling.
#include <datamunge/nn/spatial_transformer_models.hpp>
namespace stn {
namespace nn=datamunge::nn;
using datamunge::nn::models::SpatialTransformerOutputs;
using datamunge::nn::models::AffineSample;
using datamunge::nn::models::SpatialTransformer;
using datamunge::nn::models::DistortedMnistOutputs;
using datamunge::nn::models::DistortMnist;
using datamunge::nn::models::StnClassifierOutputs;
using datamunge::nn::models::StnMnistClassifier;
}
