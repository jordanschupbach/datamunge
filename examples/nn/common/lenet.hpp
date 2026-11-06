#pragma once
// Compatibility names for the canonical sparse LeNet-5 definition.
#include <datamunge/nn/cnn_models.hpp>
namespace lenet {
namespace nn = datamunge::nn;
using datamunge::nn::models::LeNetC3Connections;
using datamunge::nn::models::LeNetTanh;
using datamunge::nn::models::LeNetSubsample;
using datamunge::nn::models::LeNetC3;
using datamunge::nn::models::LeNetDigitCodes;
using datamunge::nn::models::LeNetRbfLogits;
using datamunge::nn::models::LeNetOutputs;
using datamunge::nn::models::LeNet5;
}
