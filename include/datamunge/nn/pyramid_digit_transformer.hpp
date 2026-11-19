/// @file
/// @brief Three-stage spatial-reduction transformer for digit classification.
#pragma once

#include <array>
#include <string>

#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/image_mixers.hpp"

namespace datamunge::nn {

/// @brief Digit classifier scores and spatial-reduction diagnostics.
struct PyramidDigitOutput {
  Tensor logits; ///< Class logits [batch,10].
  Tensor patches; ///< Flattened input patches [batch,16,49].
  Tensor embedded; ///< Position-aware first-stage embeddings [batch,16,24].
  std::array<Tensor,3> attention; ///< Attention maps with query/key sizes 16/4, 4/1 and 2/2.
};

/// @brief Run three stages of a Pyramid Vision Transformer on grayscale digits.
/// @param params Shared trainable parameter store.
/// @param images Float32 images [batch,28,28,1].
/// @return Logits, patches, embeddings and per-stage attention maps.
inline PyramidDigitOutput pyramid_digit_transformer(ParameterStore& params,const Tensor& images) {
  const auto neighborhoods=[](const Tensor& x,int side,int width) {
    return transpose(x.reshape({-1,side/2,2,side/2,2,width}),{0,1,3,2,4,5})
        .reshape({-1,side*side/4,4*width});
  };
  PyramidDigitOutput out;
  out.patches=image_mixer_patches(images);
  auto state=dense(params,"patch",out.patches,24);
  for (int level=0;level<3;++level) {
    const int side=4>>level,width=24+12*level,count=side*side+(level==2);
    const auto name="stage"+std::to_string(level);
    if (level>0) {
      state=dense(params,name+"/patch",neighborhoods(state,2*side,width-12),width);
    }
    if (level==2) {
      const auto cls=params.get("cls",{1,1,width},init::normal(.02));
      state=concat({broadcast_to(cls,{images.size(0),1,width}),state},1);
    }
    state=state+params.get(name+"/position",{1,count,width},init::normal(.02));
    if (level==0) out.embedded=state;
    const auto query=layer_norm(params,name+"/ln1",state);
    auto memory=query;
    if (side>1) {
      memory=layer_norm(params,name+"/reduction_norm",
          dense(params,name+"/reduction",neighborhoods(query,side,width),width));
    }
    const auto read=multi_head_attention(params,name+"/attention",query,memory,3,{.need_weights=true});
    state=state+read.output;
    out.attention[level]=read.weights;
    const auto ff=dense(params,name+"/up",layer_norm(params,name+"/ln2",state),
                        2*width,Activation::GeluTanh);
    state=state+dense(params,name+"/down",ff,width);
  }
  out.logits=dense(params,"head",layer_norm(params,"final_norm",select(state,1,0)),10);
  return out;
}

} // namespace datamunge::nn
