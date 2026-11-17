#pragma once
// Compatibility names for canonical tensor ConvNeXt V1/V2.
#include <datamunge/nn/convnext_models.hpp>
#include "common/mbconv.hpp"
namespace convnext {
namespace nn=datamunge::nn;
using datamunge::nn::models::ConvNeXtSize;
using datamunge::nn::models::GrnOutputs;
using datamunge::nn::models::ConvNeXtBlockOutputs;
using datamunge::nn::models::ConvNeXtConfig;
using datamunge::nn::models::ConvNeXtOutputs;
using datamunge::nn::models::ConvNeXtLayout;
using datamunge::nn::models::ConvNeXtConv;
using datamunge::nn::models::GlobalResponseNorm;
using datamunge::nn::models::ConvNeXtBlock;
using datamunge::nn::models::ConvNeXt;
using datamunge::nn::models::ConvNeXtMacs;
using datamunge::nn::models::ConvNeXtParameters;
}
