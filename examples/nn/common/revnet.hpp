#pragma once
// Compatibility names for canonical reversible residual tensor models.
#include <datamunge/nn/reversible_residual_models.hpp>
namespace revnet {
namespace nn=datamunge::nn;
using datamunge::nn::models::RevNetConfig;
using datamunge::nn::models::RevContext;
using datamunge::nn::models::RevPair;
using datamunge::nn::models::RevConv;
using datamunge::nn::models::RevNormRelu;
using datamunge::nn::models::RevResidual;
using datamunge::nn::models::RevShortcut;
using datamunge::nn::models::RevBlockForward;
using datamunge::nn::models::RevBlockInverse;
using datamunge::nn::models::RevCoupling;
using datamunge::nn::models::RevNetHead;
using datamunge::nn::models::RevNetUnit;
using datamunge::nn::models::RevNetOutputs;
using datamunge::nn::models::RevNetUnits;
using datamunge::nn::models::RevNet;
using Mode=datamunge::nn::models::RevNetMode;
}
