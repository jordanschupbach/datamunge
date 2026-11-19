#pragma once

/// @file tacotron2.hpp
/// @brief Compact location-sensitive text-to-mel model.

#include <array>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/losses.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Text-to-mel predictions and decoding diagnostics.
/// @par Example
/// @code
/// dm::nn::Tacotron2Result result;
/// @endcode
struct Tacotron2Result {
  Tensor raw; ///< Decoder mel frames [batch,28,24].
  Tensor refined; ///< Post-net mel frames [batch,28,24].
  Tensor stop; ///< Stop logits [batch,14].
  Tensor attention; ///< Per-group alignment [batch,14,2].
  Tensor feedback; ///< Consumed mel groups [batch,14,48].
};

/// @brief Decode one or two digit tokens into fourteen two-frame mel groups.
/// The optional teacher tensor supplies decoder feedback shifted by one
/// group. Without it, decoding feeds back its own raw mel predictions.
/// @param model Parameter store initialized on first use.
/// @param text Integer digit IDs [batch,2]. The second ID may be 10 for padding.
/// @param teacher Optional normalized mel targets [batch,28,24] in model dtype.
/// @return Raw and refined frames, stop logits, attention and feedback.
/// @throws std::invalid_argument For invalid input dimensions, symbols or parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(4);
/// auto mel=dm::nn::tacotron2(model,dm::nn::zeros({1,2},dm::nn::DType::Int32));
/// @endcode
/// @par References
/// - Shen et al. (2018), Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions.
///   https://arxiv.org/abs/1712.05884
/// @see tacotron2_loss()
inline Tacotron2Result tacotron2(ParameterStore& model,const Tensor& text,
                                const Tensor& teacher=Tensor()) {
  if (!text.defined() || text.dim()!=2 || text.size(0)<1 || text.size(1)!=2 ||
      (text.dtype()!=DType::Int32 && text.dtype()!=DType::Int64) ||
      min(text).item()<0 || max(text).item()>10 || max(select(text,1,0)).item()>9) {
    throw std::invalid_argument("Invalid Tacotron digit prompt");
  }
  const auto batch=text.size(0);
  const auto dtype=model.dtype();
  if (teacher.defined() && (teacher.shape()!=Shape{batch,28,24} || teacher.dtype()!=dtype))
    throw std::invalid_argument("Invalid Tacotron teacher frames");
  const auto guide=teacher.defined() ? reshape(teacher,{batch,14,48}) : Tensor();
  const auto embedded=dense(model,"text",one_hot(text,11,1,0,dtype),24);
  const auto tokens=time_steps(conv1d(model,"encoder_conv",embedded,24,3,{.activation=Activation::Relu}));
  auto f=lstm_zero_state(batch,12,dtype), b=f;
  std::array<Tensor,2> forward,backward;
  for (int t=0;t<2;++t) {
    f=lstm_step(model,"encoder_forward",tokens[t],f);
    b=lstm_step(model,"encoder_backward",tokens[1-t],b);
    forward[t]=f.h;
    backward[1-t]=b.h;
  }
  const auto memory=concat({stack({forward[0],forward[1]},1),stack({backward[0],backward[1]},1)},2);
  const auto keys=dense(model,"keys",memory,24);
  const auto pad=equal(slice_axis(text,1,1,2),10).to(dtype);
  const auto mask=concat({zeros({batch,1},dtype),-1e9*pad},1);
  auto state=lstm_zero_state(batch,24,dtype);
  auto previous=zeros({batch,48},dtype);
  auto last=zeros({batch,2},dtype), cumulative=last;
  std::vector<Tensor> frames,stops,attention,feedback;
  for (int t=0;t<14;++t) {
    if (guide.defined() && t>0) previous=select(guide,1,t-1);
    feedback.push_back(previous);
    const auto prenet=dense(model,"prenet2",dense(model,"prenet1",previous,24,Activation::Relu),
                            24,Activation::Relu);
    const auto query=expand_dims(dense(model,"query",state.h,24),1);
    const auto local=conv1d(model,"location",stack({last,cumulative},2),24,3);
    const auto energy=squeeze(dense(model,"energy",tanh(keys+query+local),1),{2});
    last=softmax(energy+mask);
    cumulative=cumulative+last;
    attention.push_back(last);
    const auto context=sum(memory*expand_dims(last,2),{1});
    state=lstm_step(model,"decoder",concat({prenet,context},1),state);
    const auto features=concat({state.h,context},1);
    previous=dense(model,"mel",features,48);
    frames.push_back(previous);
    stops.push_back(dense(model,"stop",features,1));
  }
  const auto raw=reshape(stack(frames,1),{batch,28,24});
  const auto hidden=conv1d(model,"post1",raw,24,5,{.activation=Activation::Tanh});
  return {raw,raw+conv1d(model,"post2",hidden,24,5),concat(stops,1),
          stack(attention,1),stack(feedback,1)};
}

/// @brief Combine raw and post-net reconstruction error with stop-token loss.
/// @param output Model predictions from tacotron2().
/// @param mel Reference normalized frames with the same shape as the predictions.
/// @param stop Binary reference stop indicators matching the stop logits.
/// @return Scalar sum of two mean squared errors and mean binary cross-entropy.
/// @throws std::invalid_argument For incompatible prediction or target shapes.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(4);
/// auto y=dm::nn::zeros({1,28,24});
/// auto out=dm::nn::tacotron2(model,dm::nn::zeros({1,2},dm::nn::DType::Int32),y);
/// auto loss=dm::nn::tacotron2_loss(out,y,dm::nn::zeros({1,14}));
/// @endcode
/// @see tacotron2()
inline Tensor tacotron2_loss(const Tacotron2Result& output,const Tensor& mel,const Tensor& stop) {
  if (!mel.defined() || !stop.defined() || !output.raw.defined() ||
      !output.refined.defined() || !output.stop.defined() ||
      mel.shape()!=output.raw.shape() || mel.shape()!=output.refined.shape() ||
      stop.shape()!=output.stop.shape()) {
    throw std::invalid_argument("Invalid Tacotron targets");
  }
  return mean(square(output.raw-mel))+mean(square(output.refined-mel))+
      sigmoid_cross_entropy(output.stop,stop);
}

}  // namespace datamunge::nn
