#pragma once
// Compatibility names; implementation belongs to the canonical tensor library.
#include <datamunge/nn/zfnet_xception.hpp>
#include "common/alexnet.hpp"
#include "common/deconvnet.hpp"
namespace zfnet {
namespace nn=datamunge::nn;
using datamunge::nn::models::ZFNetConfig;
using datamunge::nn::models::ZFNetOutputs;
using datamunge::nn::models::ZFNet;
using datamunge::nn::models::ZFNetDeconvolution;
}
