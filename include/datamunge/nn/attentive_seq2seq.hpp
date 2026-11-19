/// @file
/// @brief Additive and global dot attention for an LSTM encoder–decoder.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/completion_graphs.hpp"

namespace datamunge::nn::models::seq2seq {

/// @brief Timing and scoring rule for encoder–decoder attention.
enum class DecoderAttention {
  Bahdanau, ///< Additive attention before the LSTM update.
  Luong     ///< Global dot attention after the LSTM update.
};

/// @brief Run one attentive decoder step.
/// @param p Shared parameter store.
/// @param embedded Input token embeddings [batch,8].
/// @param previous Previous decoder state.
/// @param memory Encoder output [batch,6,8].
/// @param lengths Valid source lengths [batch].
/// @param vocab Number of target tokens.
/// @param method Attention scoring rule.
/// @return Logits, next decoder state, and masked source probabilities.
inline DecoderStep AttentiveStep(ParameterStore& p,const Tensor& embedded,const LstmState& previous,
    const Tensor& memory,const Tensor& lengths,std::int64_t vocab,DecoderAttention method) {
  const auto valid=ValidPositions(lengths);
  if (method==DecoderAttention::Bahdanau) {
    const auto attended=additive_attention(p,"attention",previous.h,memory,memory,kUnits,
                                           where(valid,zeros_like(sum(memory,{2})),-1e9));
    const auto context=where(greater(lengths,0).reshape({-1,1}),attended.output,0.0);
    const auto state=lstm_step(p,"decoder",concat({embedded,context},1),previous);
    return {Head(p,state.h,vocab),state.h,state.c,where(valid,attended.weights,0.0)};
  }
  const auto state=lstm_step(p,"decoder",embedded,previous);
  const auto scores=sum(memory*expand_dims(state.h,1),{2});
  const auto weights=where(valid,softmax(where(valid,scores,-1e9),1),0.0);
  const auto context=sum(memory*expand_dims(weights,2),{1});
  const auto combined=tanh(dense(p,"attention/combine",concat({state.h,context},1),kUnits));
  return {Head(p,combined,vocab),state.h,state.c,weights};
}

/// @brief Run teacher forcing with additive or global dot attention.
/// @param p Shared parameter store.
/// @param input Source one-hot tokens [batch,6,vocab].
/// @param teacher Teacher one-hot tokens [batch,6,vocab].
/// @param lengths Valid source lengths [batch].
/// @param target_lengths Valid output lengths [batch].
/// @param vocab Number of tokens in the shared vocabulary.
/// @param method Attention scoring rule.
/// @return Pair of logits [batch,6,vocab] and alignments [batch,6,6], zeroed at padded output steps.
inline std::pair<Tensor,Tensor> Attentive(ParameterStore& p,const Tensor& input,const Tensor& teacher,
    const Tensor& lengths,const Tensor& target_lengths,std::int64_t vocab,DecoderAttention method) {
  const auto zero=zeros({input.size(0),kUnits});
  const auto source=Encode(p,Embedding(p,input,vocab),lengths,zero,zero);
  const auto tokens=time_steps(Embedding(p,teacher,vocab));
  LstmState state{source.h.back(),source.c.back()};
  std::vector<Tensor> logits, alignments;
  for (int t=0;t<kSteps;++t) {
    const auto next=AttentiveStep(p,tokens[t],state,source.memory,lengths,vocab,method);
    const auto valid=ValidAt(target_lengths,t);
    state={where(valid,next.h,state.h),where(valid,next.c,state.c)};
    logits.push_back(where(valid,next.logits,0.0));
    alignments.push_back(where(valid,next.attention,0.0));
  }
  return {stack_time(logits),stack_time(alignments)};
}

} // namespace datamunge::nn::models::seq2seq
