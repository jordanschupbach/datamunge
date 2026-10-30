#pragma once
// Compatibility names for the canonical tensor model definition.
#include <datamunge/nn/cnn_models.hpp>
namespace cnn_baselines {
namespace nn = datamunge::nn;
using nn::models::MnistCnnOutputs;
using nn::models::MnistCnn;
}
