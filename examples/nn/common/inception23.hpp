#pragma once
// Compatibility names for canonical tensor Inception definitions.
#include <datamunge/nn/inception_models.hpp>
#include "common/normalized_conv.hpp"
namespace inception23 {
namespace nn=datamunge::nn;
using datamunge::nn::models::Inception23Config;
using datamunge::nn::models::Inception23Outputs;
using datamunge::nn::models::ConvFeature;
using datamunge::nn::models::InceptionBuilder;
using datamunge::nn::models::InceptionClassifier;
using datamunge::nn::models::InceptionAuxiliary;
using datamunge::nn::models::Inception23;
}
