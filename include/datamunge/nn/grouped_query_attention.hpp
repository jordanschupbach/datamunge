/// @file
/// @brief Attention over projected queries with shared key/value heads.
#pragma once

#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/ops.hpp>

namespace datamunge::nn {

/// @brief Apply scaled dot-product attention with contiguous groups of query heads.
///
/// Each key/value head serves H/G contiguous query heads. Gradients from
/// those query heads accumulate into the shared key and value tensors.
/// Use the causal option for right-aligned prefix decoding.
///
/// @param q Floating queries [B,H,Tq,D].
/// @param k Floating keys [B,G,Tk,D], with G dividing H.
/// @param v Floating values [B,G,Tk,Dv].
/// @param options Attention settings. A mask must broadcast to [B,H,Tq,Tk].
/// @return Output [B,H,Tq,Dv] and optional weights [B,H,Tq,Tk].
/// @throws std::invalid_argument For incompatible tensors or mask geometry.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto y=n::grouped_query_attention(n::ones({1,4,2,3}),n::ones({1,2,2,3}),
///                                        n::ones({1,2,2,5}),{.causal=true});
/// @endcode
/// @par References
/// - Ainslie et al. (2023), https://aclanthology.org/2023.emnlp-main.298/
/// @see multi_head_attention(), scaled_dot_product_attention()
inline Attended grouped_query_attention(const Tensor &q,const Tensor &k,const Tensor &v,
                                          AttentionOptions options={}) {
  if(q.dim()!=4 || k.dim()!=4 || v.dim()!=4 || q.shape()[0]<1 || q.shape()[1]<1 ||
     k.shape()[1]<1 || q.shape()[1]%k.shape()[1] || q.shape()[0]!=k.shape()[0] ||
     k.shape()[0]!=v.shape()[0] || k.shape()[1]!=v.shape()[1] ||
     q.shape()[3]!=k.shape()[3] || k.shape()[2]!=v.shape()[2])
    throw std::invalid_argument("grouped_query_attention: incompatible query or KV geometry");
  const auto b=q.shape()[0], h=q.shape()[1], g=k.shape()[1], tq=q.shape()[2], tk=k.shape()[2];
  if(options.mask.defined() && options.mask.dim()>2)
    options.mask=broadcast_to(options.mask,{b,h,tq,tk}).reshape({b,g,h/g,tq,tk});
  const auto out=scaled_dot_product_attention(q.reshape({b,g,h/g,tq,q.shape()[3]}),
      expand_dims(k,2),expand_dims(v,2),options);
  return {out.output.reshape({b,h,tq,v.shape()[3]}),
          out.weights.defined()?out.weights.reshape({b,h,tq,tk}):Tensor{}};
}

}  // namespace datamunge::nn
