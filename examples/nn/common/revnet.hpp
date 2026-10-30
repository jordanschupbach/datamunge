#pragma once
// Compatibility names for canonical reversible residual tensor models.
#include <datamunge/nn/reversible_residual_models.hpp>
namespace revnet {
namespace nn=datamunge::nn;
using nn::models::RevNetConfig;
using nn::models::RevContext;
using nn::models::RevPair;
using nn::models::RevConv;
using nn::models::RevNormRelu;
using nn::models::RevResidual;
using nn::models::RevShortcut;
using nn::models::RevBlockForward;
using nn::models::RevBlockInverse;
using nn::models::RevCoupling;
using nn::models::RevNetHead;
using nn::models::RevNetUnit;
using nn::models::RevNetOutputs;
using nn::models::RevNetUnits;
using nn::models::RevNet;
using Mode=nn::models::RevNetMode;
}
