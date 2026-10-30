#pragma once
// Compatibility names; canonical tensor definitions own the implementation.
#include <datamunge/nn/alexnet_googlenet.hpp>
#include "common/grouped.hpp"
#include "common/lrn.hpp"
namespace alexnet {
namespace nn=datamunge::nn;
using nn::models::AlexNetConfig;
using nn::models::AlexNetOutputs;
using nn::models::AlexNet;
using nn::models::AlexNetLrn;
using nn::models::KeepDropout;
}
