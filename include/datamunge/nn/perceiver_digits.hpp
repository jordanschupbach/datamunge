/// @file
/// @brief Latent-array attention and coordinate queries for digit images.
#pragma once

#include <array>
#include <string>

#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/image_mixers.hpp"

namespace datamunge::nn {

/// @brief Classifier outputs and latent-attention diagnostics.
struct PerceiverDigitOutput {
  Tensor logits; ///< Class scores [batch,10].
  Tensor patches; ///< Raster-order image patches [batch,16,49].
  Tensor embedded; ///< Position-aware patches [batch,16,24].
  std::array<Tensor,2> read; ///< Latent-to-input attention for the two passes.
  std::array<Tensor,2> process; ///< Latent self-attention for the two passes.
  Tensor decode; ///< Output-to-latent attention for Perceiver IO, otherwise undefined.
  Tensor ink; ///< Predicted mean intensity per patch for Perceiver IO, otherwise undefined.
};

/// @brief Evaluate a six-latent Perceiver with a pooled or query-based readout.
/// @param params Shared trainable parameter store.
/// @param images Grayscale images [batch,28,28,1].
/// @param io Use a class query and sixteen coordinate queries instead of latent pooling.
/// @return Scores, intermediate features and attention diagnostics.
inline PerceiverDigitOutput perceiver_digits(ParameterStore& params,const Tensor& images,bool io=false) {
  constexpr int width=24;
  const auto block=[&](const std::string& name,const Tensor& query,const Tensor& memory) {
    const auto q=layer_norm(params,name+"/query_norm",query);
    const auto kv=layer_norm(params,name+"/memory_norm",memory);
    const auto read=multi_head_attention(params,name+"/attention",q,kv,3,{.need_weights=true});
    const auto residual=query+read.output;
    const auto ff=dense(params,name+"/up",layer_norm(params,name+"/ff_norm",residual),
                        2*width,Activation::GeluTanh);
    return Attended{residual+dense(params,name+"/down",ff,width),read.weights};
  };
  PerceiverDigitOutput out;
  out.patches=image_mixer_patches(images);
  out.embedded=dense(params,"patch",out.patches,width)+
               params.get("position",{1,16,width},init::normal(.02));
  const auto seed=[&](const std::string& name,int count) {
    return broadcast_to(params.get(name,{1,count,width},init::normal(.02)),
                        {images.size(0),count,width});
  };
  auto state=seed("latent_seed",6);
  for (int pass=0;pass<2;++pass) {
    const auto read=block("read",state,out.embedded);
    const auto process=block("process",read.output,read.output);
    state=process.output;
    out.read[pass]=read.weights;
    out.process[pass]=process.weights;
  }
  Tensor pooled;
  if (io) {
    const auto decode=block("decode",seed("output_query",17),state);
    const auto norm=layer_norm(params,"final_norm",decode.output);
    pooled=select(norm,1,0);
    out.decode=decode.weights;
    out.ink=squeeze(dense(params,"ink",slice_axis(norm,1,1,17),1,Activation::Sigmoid),{2});
  } else {
    pooled=mean(layer_norm(params,"final_norm",state),{1});
  }
  out.logits=dense(params,"head",pooled,10);
  return out;
}

} // namespace datamunge::nn
