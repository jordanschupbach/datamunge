#pragma once
// Compatibility names; canonical tensor definitions own the implementation.
#include <datamunge/nn/alexnet_googlenet.hpp>
#include "common/inception.hpp"
#include "common/lrn.hpp"
namespace googlenet {
namespace nn=datamunge::nn;
using nn::models::GoogLeNetWidths;
using nn::models::GoogLeNetConfig;
using nn::models::GoogLeNetOutputs;
using nn::models::GoogLeNetAuxiliary;
using nn::models::GoogLeNet;
}
