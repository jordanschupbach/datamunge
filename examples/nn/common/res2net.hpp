#pragma once
// Compatibility names for canonical tensor models.
#include <datamunge/nn/pyramidal_multiscale_models.hpp>
#include "common/preact_resnet.hpp"
#include "common/resnet.hpp"
namespace res2net {
namespace nn=datamunge::nn;
using nn::models::Res2ScaleOutputs;
using nn::models::Res2ScaleTransform;
using nn::models::Res2NetBlock;
using nn::models::Res2NetConfig;
using nn::models::Res2Net;
}
