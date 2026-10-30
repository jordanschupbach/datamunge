#pragma once
// Compatibility names for canonical tensor models.
#include <datamunge/nn/pyramidal_multiscale_models.hpp>
#include "common/preact_resnet.hpp"
#include "common/resnet.hpp"
namespace pyramidnet {
namespace nn=datamunge::nn;
using nn::models::PyramidShortcut;
using nn::models::PyramidBlock;
using nn::models::PyramidNetConfig;
using nn::models::PyramidNetOutputs;
using nn::models::PyramidNet;
}
