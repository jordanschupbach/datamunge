#pragma once
#include "common/deeplab.hpp"
#include "common/normalized_conv.hpp"
#include "datamunge/nn/pyramid_pool_models.hpp"
namespace pspnet {
namespace nn = datamunge::nn;
using nn::models::AdaptiveAveragePool;
using nn::models::PSPNet;
using nn::models::PSPNetConfig;
using nn::models::PSPNetOutputs;
} // namespace pspnet
