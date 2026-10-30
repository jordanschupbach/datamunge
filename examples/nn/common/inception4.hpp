#pragma once
// Compatibility names for canonical tensor Inception definitions.
#include <datamunge/nn/inception_models.hpp>
#include "common/inception23.hpp"
namespace inception4 {
namespace nn=datamunge::nn;
using nn::models::ConvFeature;
using nn::models::Inception23Config;
using nn::models::Inception23Outputs;
using nn::models::InceptionBuilder;
using nn::models::Inception4Config;
using nn::models::Inception4B;
using nn::models::Inception4C;
using nn::models::InceptionResidual;
using nn::models::Inception4;
}
