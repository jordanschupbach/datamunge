#pragma once
// Compatibility names for canonical tensor affine localization and sampling.
#include <datamunge/nn/spatial_transformer_models.hpp>
namespace stn {
namespace nn=datamunge::nn;
using nn::models::SpatialTransformerOutputs;
using nn::models::AffineSample;
using nn::models::SpatialTransformer;
using nn::models::DistortedMnistOutputs;
using nn::models::DistortMnist;
using nn::models::StnClassifierOutputs;
using nn::models::StnMnistClassifier;
}
