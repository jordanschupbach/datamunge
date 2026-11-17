#pragma once
#include <datamunge/nn/guided_backprop.hpp>
#include "common/saliency.hpp"
namespace guided{namespace nn=datamunge::nn;using nn::Tensor;
using datamunge::nn::models::GuidedReluBackward;
using datamunge::nn::models::GuidedReluCount;
using datamunge::nn::models::GuidedReluFunction;
using datamunge::nn::models::GuidedRelu;
using datamunge::nn::models::GuidedSaliencyCnn;
}
