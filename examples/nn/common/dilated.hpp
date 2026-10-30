#pragma once
// Compatibility entry points; the differentiable filter expansion is canonical.
#include <datamunge/nn/convolution_components.hpp>
#include <string>
namespace dilated {
namespace nn = datamunge::nn;
inline nn::Tensor ExpandDilatedKernel(const nn::Tensor& kernel, int dilation) {
  return nn::expand_dilated_kernel(kernel, dilation);
}
inline nn::Tensor PortableDilatedConv2D(const nn::Tensor& input, const nn::Tensor& kernel,
                                       int dilation, const std::string& padding = "SAME") {
  return nn::conv2d(input, ExpandDilatedKernel(kernel,dilation),
                    nn::conv_options(1,padding=="SAME"?nn::Padding::Same:nn::Padding::Valid));
}
} // namespace dilated
