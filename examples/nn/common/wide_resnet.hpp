#pragma once
// Compatibility aliases for canonical tensor forward models.
#include <datamunge/nn/grouped_wide_residual_models.hpp>
#include "common/preact_resnet.hpp"
#include "common/resnet.hpp"
namespace wide_resnet {
namespace nn=datamunge::nn;
using nn::models::WideBlockOutputs;
using nn::models::WideBasicBlock;
using nn::models::WideResNetConfig;
using nn::models::WideResNetOutputs;
using nn::models::WideResNet;
}
