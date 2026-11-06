#pragma once
// Compatibility names for canonical tensor architecture definitions.
#include <datamunge/nn/classic_cnn_models.hpp>
namespace vgg {
namespace nn=datamunge::nn;
using datamunge::nn::models::VggBlockDepths;
using datamunge::nn::models::VggConfig;
using datamunge::nn::models::VggOutputs;
using datamunge::nn::models::Vgg;
using datamunge::nn::models::VggConvolutionalHead;
}
