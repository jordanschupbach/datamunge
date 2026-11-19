#pragma once

/// @file listen_attend_spell.hpp
/// @brief Pyramidal recurrent listener and attention-based digit speller.

#include <stdexcept>
#include <string>
#include <vector>

#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Logits, attention and actual feedback tokens for a three-step decode.
/// @par Example
/// @code
/// dm::nn::ListenAttendSpellResult result;
/// @endcode
struct ListenAttendSpellResult {
  Tensor logits; ///< [batch,3,11], digit and end-of-sequence scores.
  Tensor attention; ///< [batch,3,frames/2], normalized attention on the listener frames.
  Tensor feedback; ///< [batch,3], BOS followed by two teacher or predicted token IDs.
};

/// @brief Decode two digits and EOS with a bidirectional listener and LSTM speller.
/// Pairs adjacent listener outputs before the second recurrent layer. Each
/// decoder step attends to the listener using the previous decoder state.
/// @param model Trainable store with parameters created on first use.
/// @param audio Log-mel features [batch, even frames, bands] in the store dtype.
/// @param teacher Optional integer digit IDs [batch,2]. An undefined tensor uses greedy feedback.
/// @return Three-step scores, attention probabilities and consumed token IDs.
/// @throws std::invalid_argument For incompatible inputs or stored parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(4);
/// auto decoded = dm::nn::listen_attend_spell(model, dm::nn::zeros({1,28,24}));
/// @endcode
/// @par References
/// - Chan et al. (2016), Listen, Attend and Spell. https://arxiv.org/abs/1508.01211
/// @see additive_attention(), lstm_step()
inline ListenAttendSpellResult listen_attend_spell(ParameterStore& model, const Tensor& audio,
                                                   const Tensor& teacher = Tensor()) {
  if (!audio.defined() || audio.dim()!=3 || audio.size(0)<1 || audio.size(1)<2 ||
      audio.size(1)%2!=0 || audio.size(2)<1 || audio.dtype()!=model.dtype()) {
    throw std::invalid_argument("Invalid Listen, Attend and Spell acoustic input");
  }
  const auto batch=audio.size(0), frames=audio.size(1);
  if (teacher.defined() && (teacher.shape()!=Shape{batch,2} ||
      (teacher.dtype()!=DType::Int32 && teacher.dtype()!=DType::Int64) ||
      min(teacher).item()<0 || max(teacher).item()>9)) {
    throw std::invalid_argument("Expected two teacher digit IDs per clip");
  }
  constexpr int units=16;
  /// Encode a sequence in both temporal directions, then join aligned states.
  const auto listener=[&](const std::string& prefix,const Tensor& input) {
    const auto time=input.size(1);
    auto f=lstm_zero_state(batch,units,audio.dtype()),b=f;
    std::vector<Tensor> forward(time),backward(time);
    for (std::int64_t i=0;i<time;++i) {
      const auto j=time-1-i;
      f=lstm_step(model,prefix+"/forward",select(input,1,i),f);
      b=lstm_step(model,prefix+"/backward",select(input,1,j),b);
      forward[i]=f.h;
      backward[j]=b.h;
    }
    return concat({stack(forward,1),stack(backward,1)},2);
  };
  const auto first=listener("listener1",(audio+6)*.2);
  const auto memory=listener("listener2",reshape(first,{batch,frames/2,4*units}));
  auto state=lstm_zero_state(batch,units,audio.dtype());
  const auto table=model.get("tokens",{12,units},init::normal(.15));
  auto previous=full({batch},11,DType::Int64);
  std::vector<Tensor> logits,attention,feedback;
  for (int i=0;i<3;++i) {
    feedback.push_back(previous);
    const auto attended=additive_attention(model,"attention",state.h,memory,memory,units);
    const auto embedded=gather(table,previous);
    state=lstm_step(model,"speller",concat({embedded,attended.output},1),state);
    const auto score=dense(model,"head",concat({state.h,attended.output},1),11);
    logits.push_back(score);
    attention.push_back(attended.weights);
    if (i<2) previous=teacher.defined() ? select(teacher,1,i).to(DType::Int64) : argmax(score);
  }
  return {stack(logits,1),stack(attention,1),stack(feedback,1)};
}

}  // namespace datamunge::nn
