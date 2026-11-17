#pragma once
// Compatibility aliases for canonical tensor forward models.
#include <datamunge/nn/grouped_wide_residual_models.hpp>
#include "common/preact_resnet.hpp"
#include "common/resnet.hpp"
namespace wide_resnet {
namespace nn=datamunge::nn;
using datamunge::nn::models::WideBlockOutputs;
using datamunge::nn::models::WideBasicBlock;
using datamunge::nn::models::WideResNetConfig;
using datamunge::nn::models::WideResNetOutputs;
using datamunge::nn::models::WideResNet;
}
