#pragma once

/// @file digit_backbone.hpp
/// @brief Convolutional digit image features for transfer learning.

#include <stdexcept>
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Extract a 32-dimensional descriptor for each digit image.
/// Two convolution and max-pooling blocks feed a tanh projection.
/// The store contains only backbone parameters when called with an empty store.
/// @param model Parameter store initialized on first use.
/// @param image Floating [batch,28,28,1] pixels in the store dtype.
/// @return Descriptor tensor [batch,32].
/// @throws std::invalid_argument For incompatible images or parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// auto features=dm::nn::digit_backbone(model,dm::nn::zeros({2,28,28,1}));
/// @endcode
/// @see conv2d(), dense()
inline Tensor digit_backbone(ParameterStore& model,const Tensor& image) {
  if (!image.defined() || image.dim()!=4 || image.size(0)<1 ||
      image.shape()!=Shape{image.size(0),28,28,1} || image.dtype()!=model.dtype())
    throw std::invalid_argument("Invalid digit backbone image batch");
  const ConvLayerOptions options{.activation=Activation::Relu};
  const auto first=max_pool2d(conv2d(model,"conv1",image,8,3,options),2,2);
  const auto second=max_pool2d(conv2d(model,"conv2",first,12,3,options),2,2);
  return dense(model,"projection",reshape(second,{image.size(0),588}),32,Activation::Tanh);
}

}  // namespace datamunge::nn
