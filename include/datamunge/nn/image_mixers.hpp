/// @file
/// @brief Attention-free patch classifiers for 28-by-28 grayscale images.
#pragma once

#include <stdexcept>
#include <string>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Forward-pass diagnostics from an attention-free image classifier.
struct ImageMixerOutput {
  Tensor logits;   ///< Class scores [batch,10].
  Tensor patches;  ///< Row-major 7-by-7 patches [batch,16,49].
  Tensor embedded; ///< Patch embeddings [batch,16,24] or [batch,4,4,24].
};

/// @brief Pack nonoverlapping 7-by-7 patches in raster order.
/// @param images Grayscale images [batch,28,28,1].
/// @return Flattened patches [batch,16,49].
/// @throws std::invalid_argument Incompatible input geometry.
inline Tensor image_mixer_patches(const Tensor& images) {
  if (!images.defined() || images.dim()!=4 || images.numel()==0 ||
      images.size(1)!=28 || images.size(2)!=28 || images.size(3)!=1) {
    throw std::invalid_argument("image_mixer_patches: expected nonempty [batch,28,28,1]");
  }
  return transpose(images.reshape({-1,4,7,4,7}),{0,1,3,2,4}).reshape({-1,16,49});
}

/// @brief Classify digit images with MLP-Mixer token and channel mixing.
/// @param params Shared trainable store.
/// @param images Grayscale images [batch,28,28,1].
/// @return Scores and patch representations with live gradients.
inline ImageMixerOutput mlp_image_mixer(ParameterStore& params,const Tensor& images) {
  const auto patches=image_mixer_patches(images);
  const auto embedded=dense(params,"patch",patches,24);
  auto state=embedded;
  for (int i=0;i<2;++i) {
    const auto name="mixer"+std::to_string(i);
    const auto tokens=transpose(layer_norm(params,name+"/token_norm",state),{0,2,1});
    const auto spatial=dense(params,name+"/token_up",tokens,32,Activation::GeluTanh);
    state=state+transpose(dense(params,name+"/token_down",spatial,16),{0,2,1});
    const auto channels=layer_norm(params,name+"/channel_norm",state);
    const auto widened=dense(params,name+"/channel_up",channels,48,Activation::GeluTanh);
    state=state+dense(params,name+"/channel_down",widened,24);
  }
  return {dense(params,"head",mean(layer_norm(params,"final_norm",state),{1}),10),patches,embedded};
}

/// @brief Classify digit images with depthwise spatial mixing and pointwise channel mixing.
/// @param params Shared trainable store.
/// @param statistics Separate store for batch normalization running estimates.
/// @param images Grayscale images [batch,28,28,1].
/// @param training Whether to use batch statistics and update running estimates.
/// @return Scores and patch representations with live gradients.
inline ImageMixerOutput conv_image_mixer(ParameterStore& params,ParameterStore& statistics,
    const Tensor& images,bool training) {
  const auto patches=image_mixer_patches(images);
  const auto norm=[&](const std::string& name,const Tensor& x) {
    return batch_norm(params,statistics,name,x,training,{.momentum=.9,.eps=1e-3});
  };
  const auto embedded=norm("stem_norm",conv2d(params,"stem",images,24,7,
      {.stride=7,.padding=Padding::Valid,.activation=Activation::GeluTanh}));
  auto state=embedded;
  const ConvLayerOptions spatial{.activation=Activation::GeluTanh};
  for (int i=0;i<2;++i) {
    const auto name="mixer"+std::to_string(i);
    const auto residual=state+norm(name+"/depth_norm",
        depthwise_conv2d(params,name+"/depthwise",state,1,3,spatial));
    state=norm(name+"/point_norm",conv2d(params,name+"/pointwise",residual,24,1,spatial));
  }
  return {dense(params,"head",mean(state,{1,2}),10),patches,embedded};
}

} // namespace datamunge::nn
