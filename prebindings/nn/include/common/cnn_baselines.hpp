#pragma once
// Compatibility names for the canonical tensor model definition.
#include <datamunge/nn/cnn_models.hpp>
namespace cnn_baselines {
namespace nn = datamunge::nn;
using datamunge::nn::models::MnistCnnOutputs;
using datamunge::nn::models::MnistCnn;
}
