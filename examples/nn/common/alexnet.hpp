#pragma once
// Compatibility names; canonical tensor definitions own the implementation.
#include <datamunge/nn/alexnet_googlenet.hpp>
#include "common/grouped.hpp"
#include "common/lrn.hpp"
namespace alexnet {
namespace nn=datamunge::nn;
using datamunge::nn::models::AlexNetConfig;
using datamunge::nn::models::AlexNetOutputs;
using datamunge::nn::models::AlexNet;
using datamunge::nn::models::AlexNetLrn;
using datamunge::nn::models::KeepDropout;
}
