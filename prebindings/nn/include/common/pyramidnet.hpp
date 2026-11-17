#pragma once
// Compatibility names for canonical tensor models.
#include <datamunge/nn/pyramidal_multiscale_models.hpp>
#include "common/preact_resnet.hpp"
#include "common/resnet.hpp"
namespace pyramidnet {
namespace nn=datamunge::nn;
using datamunge::nn::models::PyramidShortcut;
using datamunge::nn::models::PyramidBlock;
using datamunge::nn::models::PyramidNetConfig;
using datamunge::nn::models::PyramidNetOutputs;
using datamunge::nn::models::PyramidNet;
}
