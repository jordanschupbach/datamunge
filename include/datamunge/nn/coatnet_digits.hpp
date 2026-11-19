/// @file
/// @brief Convolution and relative attention for digit classification.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/image_mixers.hpp"

namespace datamunge::nn {

/// @brief Scores and intermediate features of the digit CoAtNet.
struct CoatnetDigitOutput {
  Tensor logits; ///< Class scores [batch,10].
  Tensor patches; ///< Raster-order input patches [batch,16,49].
  Tensor embedded; ///< Final convolutional grid [batch,4,4,24].
  std::array<Tensor,2> attention; ///< Fine and coarse relative attention maps.
};

/// @brief Evaluate a CoAtNet with two convolutional and two attention stages.
/// @param params Shared trainable parameter store.
/// @param images Grayscale images [batch,28,28,1].
/// @return Class scores, embeddings and attention diagnostics.
inline CoatnetDigitOutput coatnet_digits(ParameterStore& params,const Tensor& images) {
  const auto patches=image_mixer_patches(images);
  const auto bottleneck=[&](const Tensor& x,const std::string& name,int width,int next) {
    const auto norm=layer_norm(params,name+"/norm",x);
    const auto wide=conv2d(params,name+"/expand",norm,2*width,1,
                           {.stride=2,.activation=Activation::GeluTanh});
    const auto spatial=depthwise_conv2d(params,name+"/depth",wide,1,3,
                                        {.activation=Activation::GeluTanh});
    const auto reduced=dense(params,name+"/squeeze",mean(spatial,{1,2}),width/2,Activation::GeluTanh);
    const auto gate=dense(params,name+"/excite",reduced,2*width,Activation::Sigmoid);
    const auto skip=conv2d(params,name+"/skip",max_pool2d(x,2,2,Padding::Same),next,1);
    return skip+conv2d(params,name+"/project",spatial*gate.reshape({-1,1,1,2*width}),next,1);
  };
  const auto attend=[&](const Tensor& x,const std::string& name,int side,int width) {
    const int tokens=side*side,span=2*side-1;
    const auto sequence=x.reshape({-1,tokens,width});
    std::vector<int> offsets;
    for (int q=0;q<tokens;++q) for (int k=0;k<tokens;++k)
      offsets.push_back(span*(q/side-k/side+side-1)+q%side-k%side+side-1);
    const auto relative=params.get(name+"/relative",{span*span,3},init::normal(.02));
    const auto bias=transpose(gather(relative,from_vector(offsets)).reshape({tokens,tokens,3}),{2,0,1});
    const auto norm=layer_norm(params,name+"/ln1",sequence);
    const auto mixed=multi_head_attention(params,name+"/attention",norm,norm,3,
                                          {.mask=bias,.need_weights=true});
    const auto residual=sequence+mixed.output;
    const auto ff=dense(params,name+"/up",layer_norm(params,name+"/ln2",residual),
                        2*width,Activation::GeluTanh);
    return Attended{residual+dense(params,name+"/down",ff,width),mixed.weights};
  };
  const auto stem=conv2d(params,"stem",images,12,3,{.stride=2,.activation=Activation::GeluTanh});
  const auto fine=bottleneck(stem,"mb1",12,18);
  const auto grid=bottleneck(fine,"mb2",18,24);
  const auto global=attend(grid,"global",4,24);
  const auto pool=max_pool2d(global.output.reshape({-1,4,4,24}),2,2);
  const auto coarse=attend(conv2d(params,"merge",pool,36,1),"coarse",2,36);
  const auto pooled=mean(layer_norm(params,"final_norm",coarse.output),{1});
  return {dense(params,"head",pooled,10),patches,grid,{global.weights,coarse.weights}};
}

} // namespace datamunge::nn
