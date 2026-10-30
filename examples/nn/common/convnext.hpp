#pragma once
// Compatibility names for canonical tensor ConvNeXt V1/V2.
#include <datamunge/nn/convnext_models.hpp>
#include "common/mbconv.hpp"
namespace convnext {
namespace nn=datamunge::nn;
using nn::models::ConvNeXtSize;
using nn::models::GrnOutputs;
using nn::models::ConvNeXtBlockOutputs;
using nn::models::ConvNeXtConfig;
using nn::models::ConvNeXtOutputs;
using nn::models::ConvNeXtLayout;
using nn::models::ConvNeXtConv;
using nn::models::GlobalResponseNorm;
using nn::models::ConvNeXtBlock;
using nn::models::ConvNeXt;
using nn::models::ConvNeXtMacs;
using nn::models::ConvNeXtParameters;
}
