/// @file
/// @brief Masked response fitting and prompt-only greedy character decoding.
#pragma once

#include <datamunge/nn/autoregressive_components.hpp>
#include <datamunge/nn/losses.hpp>

namespace datamunge::nn {

/// @brief Average token cross-entropy over selected response positions.
///
/// Ignored targets may contain arbitrary integer sentinels. The mask
/// and labels are treated as constants.
///
/// @param logits Floating scores [N,T,C], with C positive.
/// @param targets Integral token IDs [N,T], valid at selected positions.
/// @param mask Boolean [N,T] selecting at least one supervised position.
/// @return Differentiable scalar response NLL.
/// @throws std::invalid_argument For invalid geometry, selected labels or an empty mask.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto loss=n::response_token_nll(n::zeros({1,2,3}),
///     n::zeros({1,2},n::DType::Int64),n::ones({1,2},n::DType::Bool));
/// @endcode
/// @see greedy_character_response()
inline Tensor response_token_nll(const Tensor &logits,const Tensor &targets,const Tensor &mask) {
  if(logits.dim()!=3 || logits.shape()[2]<1 || !is_floating(logits.dtype()) ||
     targets.shape()!=Shape{logits.shape()[0],logits.shape()[1]} ||
     !is_integral(targets.dtype()) || mask.shape()!=targets.shape() || mask.dtype()!=DType::Bool)
    throw std::invalid_argument("response_token_nll: invalid tensor geometry");
  const auto count=count_nonzero(mask).item<std::int64_t>();
  if(count==0)
    throw std::invalid_argument("response_token_nll: no supervised tokens");
  const auto ids=where(mask,targets,0.);
  if(min(ids).item()<0 || max(ids).item()>=logits.shape()[2])
    throw std::invalid_argument("response_token_nll: selected token outside vocabulary");
  const auto losses=sparse_softmax_cross_entropy(logits,ids,Reduction::None);
  return sum(where(mask,losses,0.))/double(count);
}

/// @brief Generate greedy responses for a batch of variable-length prompts.
///
/// Uses the default 64-position, 65-character causal decoder. Entries
/// beyond each prompt length are ignored, including any supplied response.
/// The stop token is included and all remaining output slots are -1.
///
/// @param model Owns or lazily initializes the decoder parameters.
/// @param input Integral token array [N,64].
/// @param prompt_lengths Integral [N] positive prompt lengths leaving room for max_tokens.
/// @param stop Stop token ID in [0,65).
/// @param max_tokens Positive generation budget, default 4.
/// @return Int32 response tokens [N,max_tokens] padded with -1.
/// @throws std::invalid_argument For invalid geometry, prompt IDs or generation settings.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// n::ParameterStore model(7);
/// const auto answer=n::greedy_character_response(model,n::zeros({1,64},n::DType::Int64),
///     n::ones({1},n::DType::Int64),0);
/// @endcode
/// @see models::DecoderTransformer(), response_token_nll()
inline Tensor greedy_character_response(ParameterStore &model,const Tensor &input,
    const Tensor &prompt_lengths,int stop,int max_tokens=4) {
  if(input.dim()!=2 || input.shape()[0]<1 || input.shape()[1]!=64 ||
     !is_integral(input.dtype()) || prompt_lengths.shape()!=Shape{input.shape()[0]} ||
     !is_integral(prompt_lengths.dtype()) || stop<0 || stop>=65 || max_tokens<1 || max_tokens>=64 ||
     min(prompt_lengths).item()<1 || max(prompt_lengths).item()>64-max_tokens)
    throw std::invalid_argument("greedy_character_response: invalid prompt or budget");
  NoGradGuard guard;
  const auto batch=input.shape()[0];
  const auto lengths=prompt_lengths.to_vector<int>();
  const auto supplied=input.to_vector<std::int64_t>();
  std::vector<int> context(batch*64), response(batch*max_tokens,-1);
  std::vector<bool> finished(batch,false);
  for(std::int64_t row=0;row<batch;++row)
    for(int t=0;t<lengths[row];++t) {
      const auto token=supplied[row*64+t];
      if(token<0 || token>=65)
        throw std::invalid_argument("greedy_character_response: prompt ID outside vocabulary");
      context[row*64+t]=static_cast<int>(token);
    }
  for(int step=0;step<max_tokens;++step) {
    const auto ids=from_vector(context,{batch,64});
    const auto path=models::DecoderTransformer(model,one_hot(ids,65,1.,0.,model.dtype()));
    const auto next=argmax(path.logits,2).to_vector<int>();
    for(std::int64_t row=0;row<batch;++row)
      if(!finished[row]) {
        const auto pos=row*64+lengths[row]+step;
        const auto token=next[pos-1];
        response[row*max_tokens+step]=context[pos]=token;
        finished[row]=token==stop;
      }
  }
  return from_vector(response,{batch,max_tokens});
}

}  // namespace datamunge::nn
