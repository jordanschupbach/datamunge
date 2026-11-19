#pragma once

/// @file tabular_image_fusion.hpp
/// @brief Image and metadata fusion with multiplicative interactions.

#include <stdexcept>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Predict a binary logit from paired image and metadata rows.
/// A convolution and global spatial average encode the image. The fused
/// features concatenate the two encodings and their elementwise product.
/// @param model Parameter store whose weights initialize on first use.
/// @param images Floating NHWC image batch with positive dimensions.
/// @param metadata Floating [batch,fields] with positive field count.
/// Both inputs must match the store dtype.
/// @return Binary logits [batch,1].
/// @throws std::invalid_argument For incompatible shapes, dtypes or parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// auto logits=dm::nn::tabular_image_logits(model,dm::nn::zeros({2,8,8,1}),
///                                        dm::nn::zeros({2,3}));
/// @endcode
/// @see global_avg_pool2d(), sigmoid_cross_entropy()
inline Tensor tabular_image_logits(ParameterStore& model,const Tensor& images,
                                    const Tensor& metadata) {
  if (!images.defined() || !metadata.defined() || images.dim()!=4 ||
      images.size(0)<1 || images.size(1)<1 || images.size(2)<1 || images.size(3)<1 ||
      metadata.dim()!=2 || metadata.size(0)!=images.size(0) || metadata.size(1)<1 ||
      images.dtype()!=model.dtype() || metadata.dtype()!=model.dtype()) {
    throw std::invalid_argument("Invalid paired image and metadata inputs");
  }
  const auto features=conv2d(model,"image_conv",images,12,3,{.activation=Activation::Relu});
  const auto image=dense(model,"image_code",global_avg_pool2d(features),12,Activation::Tanh);
  const auto meta=dense(model,"metadata",metadata,12,Activation::Tanh);
  const auto fused=concat({image,meta,image*meta},1);
  return dense(model,"head",dense(model,"fusion",fused,16,Activation::Tanh),1);
}

}  // namespace datamunge::nn
