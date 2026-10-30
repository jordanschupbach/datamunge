#pragma once
#include <datamunge/nn/guided_backprop.hpp>
#include "common/saliency.hpp"
namespace guided{namespace nn=datamunge::nn;using nn::Tensor;
using nn::models::GuidedReluBackward;
using nn::models::GuidedReluCount;
using nn::models::GuidedReluFunction;
using nn::models::GuidedRelu;
using nn::models::GuidedSaliencyCnn;
}
