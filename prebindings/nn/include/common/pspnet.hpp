#pragma once
#include "common/deeplab.hpp"
#include "common/normalized_conv.hpp"
#include "datamunge/nn/pyramid_pool_models.hpp"
namespace pspnet {
namespace nn = datamunge::nn;
using datamunge::nn::models::AdaptiveAveragePool;
using datamunge::nn::models::PSPNet;
using datamunge::nn::models::PSPNetConfig;
using datamunge::nn::models::PSPNetOutputs;
} // namespace pspnet
