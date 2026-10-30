#pragma once
// Compatibility names for canonical tensor architecture definitions.
#include <datamunge/nn/classic_cnn_models.hpp>
namespace vgg {
namespace nn=datamunge::nn;
using nn::models::VggBlockDepths;
using nn::models::VggConfig;
using nn::models::VggOutputs;
using nn::models::Vgg;
using nn::models::VggConvolutionalHead;
}
