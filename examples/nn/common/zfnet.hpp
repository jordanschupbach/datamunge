#pragma once
// Compatibility names; implementation belongs to the canonical tensor library.
#include <datamunge/nn/zfnet_xception.hpp>
#include "common/alexnet.hpp"
#include "common/deconvnet.hpp"
namespace zfnet {
namespace nn=datamunge::nn;
using nn::models::ZFNetConfig;
using nn::models::ZFNetOutputs;
using nn::models::ZFNet;
using nn::models::ZFNetDeconvolution;
}
