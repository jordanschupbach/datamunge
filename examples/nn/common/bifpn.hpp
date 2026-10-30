#pragma once
// Compatibility aliases to the canonical tensor implementation.
#include "datamunge/nn/bifpn_models.hpp"
namespace bifpn {
namespace nn=datamunge::nn;
using nn::models::Swish;
using nn::models::FastFeatureFusion;
using nn::models::DetSeparable;
using nn::models::DetBatchNorm;
using nn::models::BiFpnCell;
}
