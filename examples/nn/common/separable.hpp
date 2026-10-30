#pragma once
// Compatibility entry point for the canonical differentiable kernel composition.
#include <datamunge/nn/convolution_components.hpp>
#include <stdexcept>
namespace separable {
namespace nn = datamunge::nn;
inline nn::Tensor ComposeSeparableKernel(const nn::Tensor& depthwise,const nn::Tensor& pointwise,
                                         int kernel,int in,int multiplier,int out) {
  if(kernel<=0 || in<=0 || multiplier<=0 || out<=0 ||
     depthwise.shape()!=nn::Shape{kernel,kernel,in,multiplier} ||
     pointwise.shape()!=nn::Shape{1,1,in*multiplier,out})
    throw std::invalid_argument("invalid separable filter dimensions");
  return nn::compose_separable_kernel(depthwise,pointwise);
}
} // namespace separable
