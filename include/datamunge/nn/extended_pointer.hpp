/// @file
/// @brief Three-step pointer generation with local out-of-vocabulary token IDs.
#pragma once

#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/layers.hpp>
#include <array>

namespace datamunge::nn {

/// @brief Decode using a learned mixture of vocabulary and source-position probabilities.
///
/// IDs 12 and 13 identify local unknown words. They use the shared unknown
/// embedding but remain distinct in the copy map. Repeated source positions
/// add their mass. Without targets, decoding feeds back greedy predictions
/// and keeps emitting EOS once reached.
///
/// @param model Owns shared embeddings and the recurrent and attention parameters.
/// @param source Integral [N,6] source token IDs in [0,14), N positive.
/// @param lengths Integral [N] valid source lengths in [1,6].
/// @param targets Optional integral [N,3] teacher tokens in [0,14).
/// @return In order, mixed probabilities [N,3,14], base probabilities [N,3,12],
/// attention [N,3,6], generator gates [N,3,1], and fed-back IDs [N,3].
/// @throws std::invalid_argument For invalid token geometry, IDs or lengths.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto ids = dm::nn::DType::Int64;
/// const auto out = dm::nn::extended_pointer_generator(model,
///     dm::nn::zeros({2,6},ids), dm::nn::full({2},6.,ids));
/// @endcode
/// @par References
/// - See et al. (2017), https://aclanthology.org/P17-1099/
/// @see additive_attention()
inline std::array<Tensor, 5> extended_pointer_generator(ParameterStore &model,
    const Tensor &source, const Tensor &lengths, const Tensor &targets = {}) {
  if (source.dim() != 2 || source.shape()[0] < 1 || source.shape()[1] != 6 ||
      lengths.shape() != Shape{source.shape()[0]} || !is_integral(lengths.dtype()) ||
      !is_integral(source.dtype()) || min(source).item() < 0 || max(source).item() >= 14 ||
      min(lengths).item() < 1 || max(lengths).item() > 6 ||
      (targets.defined() && (targets.shape() != Shape{source.shape()[0],3} ||
          !is_integral(targets.dtype()) || min(targets).item() < 0 || max(targets).item() >= 14)))
    throw std::invalid_argument("extended_pointer_generator: invalid tokens or lengths");
  const auto batch = source.shape()[0];
  const auto type = model.dtype();
  const auto table = model.get("embedding/W",{12,8},init::normal(.1));
  const auto embed = [&](const Tensor &ids) { return gather(table, where(less(ids,12.), ids, 3.)); };
  auto state = lstm_zero_state(batch,8,type);
  const auto seen = embed(source);
  std::vector<Tensor> hidden;
  for (int t = 0; t < 6; ++t) {
    const auto next = lstm_step(model,"encoder",select(seen,1,t),state);
    const auto valid = greater(lengths,t).reshape({batch,1});
    state.h = where(valid,next.h,state.h);
    state.c = where(valid,next.c,state.c);
    hidden.push_back(state.h);
  }
  const auto memory = stack(hidden,1);
  const auto valid = less(arange(0,6,1,DType::Int64), lengths.reshape({batch,1}));
  const auto mask = where(valid, scalar(0.,type), scalar(-1e9,type));
  const auto map = one_hot(source,14,1.,0.,type);
  auto token = full({batch},1.,DType::Int64);
  std::array<std::vector<Tensor>,5> trace;
  for (int t = 0; t < 3; ++t) {
    const auto word = embed(token);
    state = lstm_step(model,"decoder",word,state);
    const auto attended = additive_attention(model,"attention",state.h,memory,memory,8,mask);
    const auto features = concat({state.h,attended.output},1);
    const auto gate = sigmoid(dense(model,"gate",concat({features,word},1),1));
    const auto base = softmax(dense(model,"generate",features,12),1);
    const auto copy = sum(map * expand_dims(attended.weights,2),{1});
    const auto prob = gate * concat({base,zeros({batch,2},type)},1) + (1. - gate) * copy;
    const auto chosen = argmax(prob,1);
    token = targets.defined() ? select(targets,1,t) : where(equal(token,2.),2.,chosen);
    const std::array values{prob,base,attended.weights,gate,token};
    for (std::size_t i=0;i<values.size();++i)
      trace[i].push_back(values[i]);
  }
  return {stack(trace[0],1),stack(trace[1],1),stack(trace[2],1),stack(trace[3],1),stack(trace[4],1)};
}


/// @brief Point to distinct source slots with a recurrent additive-attention decoder.
///
/// Slot zero is STOP and stays available. In greedy mode it is absorbing.
/// Teacher mode feeds back the supplied indices. Both paths mask padded
/// slots and previously selected nonzero slots with -1e9.
///
/// @param model Owns the encoder, decoder and three attention tensors.
/// @param points Floating [N,7,3] source slots, including STOP at index zero.
/// @param lengths Integral [N] valid source lengths in [1,7].
/// @param targets Optional integral [N,7] teacher indices in [0,7).
/// @return Logits [N,7,7] and selected indices [N,7], in that order.
/// @throws std::invalid_argument For incompatible geometry, IDs or lengths.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(7);
/// const auto result = dm::nn::point_sequence_pointer(p, dm::nn::zeros({2,7,3}),
///     dm::nn::ones({2},dm::nn::DType::Int64));
/// @endcode
/// @par References
/// - Vinyals et al. (2015), https://arxiv.org/abs/1506.03134
/// @see extended_pointer_generator()
inline std::pair<Tensor,Tensor> point_sequence_pointer(ParameterStore &model, const Tensor &points,
    const Tensor &lengths, const Tensor &targets = {}) {
  if (points.dim()!=3 || points.shape()[0]<1 || points.shape()[1]!=7 || points.shape()[2]!=3 ||
      points.dtype()!=model.dtype() || lengths.shape()!=Shape{points.shape()[0]} ||
      !is_integral(lengths.dtype()) || min(lengths).item()<1 || max(lengths).item()>7 ||
      (targets.defined() && (targets.shape()!=Shape{points.shape()[0],7} ||
       !is_integral(targets.dtype()) || min(targets).item()<0 || max(targets).item()>=7)))
    throw std::invalid_argument("point_sequence_pointer: invalid geometry");
  const auto batch=points.shape()[0];
  const auto type=points.dtype();
  const auto slots=arange(0,7,1,DType::Int64);
  auto state=lstm_zero_state(batch,12,type);
  std::vector<Tensor> hidden;
  for (int t=0;t<7;++t) {
    const auto next=lstm_step(model,"encoder",select(points,1,t),state);
    const auto keep=greater(lengths,t).reshape({batch,1});
    state.h=where(keep,next.h,state.h);
    state.c=where(keep,next.c,state.c);
    hidden.push_back(state.h);
  }
  const auto memory=stack(hidden,1);
  const auto q=model.get("pointer/query",{12,12},init::normal(.2));
  const auto k=model.get("pointer/key",{12,12},init::normal(.2));
  const auto v=model.get("pointer/vector",{12,1},init::normal(.2));
  const auto projected=matmul(memory,k);
  auto available=less(slots,lengths.reshape({batch,1}));
  auto finished=equal(lengths,0.);
  auto feedback=zeros({batch,12},type);
  std::vector<Tensor> scores, selected;
  for(int t=0;t<7;++t) {
    state=lstm_step(model,"decoder",feedback,state);
    const auto raw=matmul(tanh(expand_dims(matmul(state.h,q),1)+projected),v).reshape({batch,7});
    const auto logits=where(available,raw,-1e9);
    const auto chosen=targets.defined()?select(targets,1,t):where(finished,0.,argmax(logits,1));
    scores.push_back(logits);
    selected.push_back(chosen);
    finished=logical_or(finished,equal(chosen,0.));
    const auto index=equal(chosen.reshape({batch,1}),slots);
    available=logical_and(available,logical_or(equal(slots,0.),equal(index,0.)));
    feedback=sum(memory*expand_dims(index.to(type),2),{1});
  }
  return {stack(scores,1),stack(selected,1)};
}

}  // namespace datamunge::nn
