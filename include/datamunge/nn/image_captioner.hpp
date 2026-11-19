#pragma once

/// @file image_captioner.hpp
/// @brief Global and spatial-attention recurrent image captioning.

#include <stdexcept>
#include <vector>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Caption decoder outputs.
/// @par Example
/// @code
/// dm::nn::ImageCaptionResult result;
/// @endcode
struct ImageCaptionResult {
  Tensor logits; ///< Per-token logits [batch,3,8].
  Tensor tokens; ///< Greedy token choices at each step [batch,3].
  Tensor attention; ///< Spatial attention [batch,3,36], undefined in global mode.
};

/// @brief Predict color, shape and EOS with a global or attentive image captioner.
/// @param model Trainable store. Use a separate store for each mode.
/// @param images Floating [batch,12,12,3] images in the store dtype.
/// @param attentive Whether to read spatial regions at each decoder step.
/// @param teacher Optional reference IDs [batch,3] in [0,7] for teacher forcing.
/// @return Caption logits, argmax tokens and optional attention maps.
/// @throws std::invalid_argument For invalid inputs or parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// auto caption=dm::nn::image_caption(model,dm::nn::zeros({1,12,12,3}),true);
/// @endcode
/// @par References
/// - Vinyals et al. (2015), Show and Tell. https://arxiv.org/abs/1411.4555
/// - Xu et al. (2015), Show, Attend and Tell. https://arxiv.org/abs/1502.03044
/// @see lstm_step(), global_avg_pool2d()
inline ImageCaptionResult image_caption(ParameterStore& model,const Tensor& images,
                                        bool attentive,const Tensor& teacher=Tensor()) {
  if (!images.defined() || images.dim()!=4 || images.size(0)<1 ||
      images.shape()!=Shape{images.size(0),12,12,3} || images.dtype()!=model.dtype())
    throw std::invalid_argument("Invalid caption image batch");
  const auto batch=images.size(0);
  if (teacher.defined() && (teacher.shape()!=Shape{batch,3} ||
      (teacher.dtype()!=DType::Int32 && teacher.dtype()!=DType::Int64) ||
      min(teacher).item()<0 || max(teacher).item()>7))
    throw std::invalid_argument("Invalid reference caption tokens");
  const auto w=model.get("cnn/filters",{3,3,3,16},init::uniform(.22));
  const auto b=model.get("cnn/bias",{16},init::zeros());
  const auto spatial=avg_pool2d(relu(bias_add(conv2d(images,w),b)),2,2);
  const auto descriptor=dense(model,"image_projection",global_avg_pool2d(spatial),24,Activation::Tanh);
  const auto zero=zeros_like(descriptor);
  const auto regions=reshape(spatial,{batch,36,16});
  const auto keys=attentive ? dense(model,"attention_keys",regions,24) : Tensor();
  auto state=attentive ? LstmState{descriptor,zero}
                       : lstm_step(model,"lstm",descriptor,{zero,zero});
  const auto table=model.get("token_embedding",{8,24},init::uniform(.25));
  auto previous=zeros({batch},DType::Int64);
  std::vector<Tensor> logits,tokens,maps;
  for (int t=0;t<3;++t) {
    auto input=gather(table,previous);
    if (attentive) {
      const auto query=expand_dims(dense(model,"attention_query",state.h,24),1);
      const auto mass=softmax(squeeze(dense(model,"attention_score",tanh(keys+query),1),{2}));
      maps.push_back(mass);
      input=concat({input,sum(regions*expand_dims(mass,2),{1})},1);
    }
    state=lstm_step(model,"lstm",input,state);
    logits.push_back(dense(model,"token_head",state.h,8));
    tokens.push_back(argmax(logits.back()));
    previous=teacher.defined() ? select(teacher,1,t).to(DType::Int64) : tokens.back();
  }
  return {stack(logits,1),stack(tokens,1),attentive ? stack(maps,1) : Tensor()};
}

}  // namespace datamunge::nn
