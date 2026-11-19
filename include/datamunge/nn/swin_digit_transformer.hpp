/// @file
/// @brief Shifted-window attention and patch merging for digit classification.
#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/image_mixers.hpp"

namespace datamunge::nn {

/// @brief Digit logits and window attention diagnostics.
struct SwinDigitOutput {
  Tensor logits; ///< Class scores [batch,10].
  Tensor patches; ///< Flattened 7-by-7 patches [batch,16,49].
  Tensor embedded; ///< Patch embeddings [batch,16,24].
  std::array<Tensor,3> attention; ///< Regular, shifted and coarse attention maps.
};

/// @brief Run local and shifted window attention, merge patches, then classify.
/// @param params Shared trainable parameter store.
/// @param images Float32 digit images [batch,28,28,1].
/// @return Class scores, patch representations and attention maps.
inline SwinDigitOutput swin_digit_transformer(ParameterStore& params,const Tensor& images) {
  const auto windows=[](const Tensor& x,int side,int width) {
    return transpose(x.reshape({-1,side/2,2,side/2,2,width}),{0,1,3,2,4,5}).reshape({-1,4,width});
  };
  const auto shift_grid=[](const Tensor& x,int side,int shift) {
    if (shift==0) return x;
    std::vector<int> order;
    for (int y=0;y<side;++y) for (int col=0;col<side;++col)
      order.push_back(((y+shift+side)%side)*side+(col+shift+side)%side);
    return gather(x,from_vector(order),1);
  };
  const auto block=[&](const Tensor& grid,const std::string& name,int side,int width,int heads,int shift) {
    const int count=side*side/4;
    const auto packed=windows(shift_grid(grid,side,shift),side,width);
    const auto norm=layer_norm(params,name+"/ln1",packed);
    std::vector<int> offsets;
    for (int q=0;q<4;++q) for (int k=0;k<4;++k) {
      offsets.push_back(3*(q/2-k/2+1)+q%2-k%2+1);
    }
    const auto relative=params.get(name+"/relative",{9,heads},init::normal(.02));
    const auto bias=transpose(gather(relative,from_vector(offsets)).reshape({4,4,heads}),{2,0,1});
    std::vector<float> mask;
    for (int i=0;i<count;++i) {
      const auto wrapped=[=](int position) {
        return std::pair{2*(i/(side/2))+position/2+shift>=side,
                         2*(i%(side/2))+position%2+shift>=side};
      };
      for (int q=0;q<4;++q) for (int k=0;k<4;++k)
        mask.push_back(wrapped(q)==wrapped(k) ? 0.f : -1e9f);
    }
    const auto blocked=tile(from_vector(mask,{count,1,4,4}),{grid.size(0),1,1,1});
    const auto mixed=multi_head_attention(params,name+"/attention",norm,norm,heads,
                                          {.mask=blocked+bias,.need_weights=true});
    const auto residual=packed+mixed.output;
    const auto wide=dense(params,name+"/up",layer_norm(params,name+"/ln2",residual),
                          2*width,Activation::GeluTanh);
    const auto local=residual+dense(params,name+"/down",wide,width);
    const auto restored=transpose(local.reshape({-1,side/2,side/2,2,2,width}),{0,1,3,2,4,5})
        .reshape({-1,side*side,width});
    return std::pair{shift_grid(restored,side,-shift),mixed.weights};
  };
  SwinDigitOutput out;
  out.patches=image_mixer_patches(images);
  out.embedded=dense(params,"patch",out.patches,24);
  const auto [regular,a]=block(out.embedded,"regular",4,24,3,0);
  const auto [shifted,b]=block(regular,"shifted",4,24,3,1);
  const auto merged=layer_norm(params,"merge_norm",windows(shifted,4,24).reshape({-1,4,96}));
  const auto compact=dense(params,"merge",merged,48);
  const auto [coarse,c]=block(compact,"coarse",2,48,6,0);
  out.logits=dense(params,"head",mean(layer_norm(params,"final_norm",coarse),{1}),10);
  out.attention={a,b,c};
  return out;
}

} // namespace datamunge::nn
