#pragma once
// Compatibility aliases to the canonical tensor implementation.
#include "datamunge/nn/bifpn_models.hpp"
namespace bifpn {
namespace nn=datamunge::nn;
using datamunge::nn::models::Swish;
using datamunge::nn::models::FastFeatureFusion;
using datamunge::nn::models::DetSeparable;
using datamunge::nn::models::DetBatchNorm;
using datamunge::nn::models::BiFpnCell;
}
