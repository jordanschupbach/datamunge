#pragma once
// Compatibility aliases for canonical tensor forward models.
#include <datamunge/nn/grouped_wide_residual_models.hpp>
#include "common/preact_resnet.hpp"
#include "common/resnet.hpp"
namespace resnext {
namespace nn=datamunge::nn;
using nn::models::ResNeXtBlock;
using nn::models::ResNeXtConfig;
using nn::models::ResNeXtOutputs;
using nn::models::ResNeXt;
}
