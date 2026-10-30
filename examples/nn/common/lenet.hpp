#pragma once
// Compatibility names for the canonical sparse LeNet-5 definition.
#include <datamunge/nn/cnn_models.hpp>
namespace lenet {
namespace nn = datamunge::nn;
using nn::models::LeNetC3Connections;
using nn::models::LeNetTanh;
using nn::models::LeNetSubsample;
using nn::models::LeNetC3;
using nn::models::LeNetDigitCodes;
using nn::models::LeNetRbfLogits;
using nn::models::LeNetOutputs;
using nn::models::LeNet5;
}
