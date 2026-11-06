#pragma once
#include <datamunge/nn/smoothgrad.hpp>
#include "common/autoencoder.hpp"
#include "common/integrated_gradients.hpp"
namespace smooth{namespace nn=datamunge::nn;using nn::Tensor;using ig::InputGradientFunction;using datamunge::nn::models::SmoothGradResult;using datamunge::nn::models::SmoothGrad;inline Tensor GaussianNoise(const nn::Shape&shape,int samples,std::uint32_t seed){return nn::models::SmoothGradNoise(shape,samples,seed);}}
