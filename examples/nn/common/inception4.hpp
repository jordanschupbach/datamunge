#pragma once
// Compatibility names for canonical tensor Inception definitions.
#include <datamunge/nn/inception_models.hpp>
#include "common/inception23.hpp"
namespace inception4 {
namespace nn=datamunge::nn;
using datamunge::nn::models::ConvFeature;
using datamunge::nn::models::Inception23Config;
using datamunge::nn::models::Inception23Outputs;
using datamunge::nn::models::InceptionBuilder;
using datamunge::nn::models::Inception4Config;
using datamunge::nn::models::Inception4B;
using datamunge::nn::models::Inception4C;
using datamunge::nn::models::InceptionResidual;
using datamunge::nn::models::Inception4;
}
