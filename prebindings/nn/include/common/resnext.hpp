#pragma once
// Compatibility aliases for canonical tensor forward models.
#include <datamunge/nn/grouped_wide_residual_models.hpp>
#include "common/preact_resnet.hpp"
#include "common/resnet.hpp"
namespace resnext {
namespace nn=datamunge::nn;
using datamunge::nn::models::ResNeXtBlock;
using datamunge::nn::models::ResNeXtConfig;
using datamunge::nn::models::ResNeXtOutputs;
using datamunge::nn::models::ResNeXt;
}
