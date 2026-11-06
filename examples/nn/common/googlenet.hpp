#pragma once
// Compatibility names; canonical tensor definitions own the implementation.
#include <datamunge/nn/alexnet_googlenet.hpp>
#include "common/inception.hpp"
#include "common/lrn.hpp"
namespace googlenet {
namespace nn=datamunge::nn;
using datamunge::nn::models::GoogLeNetWidths;
using datamunge::nn::models::GoogLeNetConfig;
using datamunge::nn::models::GoogLeNetOutputs;
using datamunge::nn::models::GoogLeNetAuxiliary;
using datamunge::nn::models::GoogLeNet;
}
