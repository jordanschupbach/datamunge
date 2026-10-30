#pragma once
// Compatibility names for canonical tensor Inception definitions.
#include <datamunge/nn/inception_models.hpp>
#include "common/normalized_conv.hpp"
namespace inception23 {
namespace nn=datamunge::nn;
using nn::models::Inception23Config;
using nn::models::Inception23Outputs;
using nn::models::ConvFeature;
using nn::models::InceptionBuilder;
using nn::models::InceptionClassifier;
using nn::models::InceptionAuxiliary;
using nn::models::Inception23;
}
