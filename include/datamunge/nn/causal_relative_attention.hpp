/// @file
/// @brief Causal self-attention with query-dependent relative positions.

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Add query–relative-position dot products to causal self-attention.
///
/// For key position j <= i, the relative contribution is the dot product
/// of query i and the learned embedding of distance i-j. Both content
/// and relative dots are scaled by the inverse square root of key width.
/// Scores are gathered from a query-by-distance matrix without expanding
/// embeddings over all position pairs. Gradients reach the relative table.
/// @param q CPU floating queries [batch,heads,time,key_width] with positive extents.
/// @param k Keys with the same shape and dtype as q.
/// @param v Values [batch,heads,time,value_width], same dtype as q.
/// @param relative Shared distance table [time,key_width], same dtype as q.
/// @return Attention output and causal attention weights.
/// @throws std::invalid_argument Undefined inputs or incompatible shapes, devices or dtypes.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::ones({1,1,2,1});
/// const auto result = n::causal_relative_attention(x,x,x,n::zeros({2,1}));
/// std::print("{:.0f}\n", result.weights.to_vector<float>()[1]); // 0
/// @endcode
/// @par References
/// - Huang et al. (2019), Music Transformer. https://arxiv.org/abs/1809.04281
/// @see scaled_dot_product_attention()
inline Attended causal_relative_attention(
    const Tensor& q,const Tensor& k,const Tensor& v,const Tensor& relative) {
  for (const auto* input : {&q,&k,&v,&relative}) {
    if (!input->defined() || input->device().backend!=compute::Backend::CPU ||
        input->dtype()!=q.dtype()) throw std::invalid_argument("relative attention: expected matching CPU tensors");
  }
  if (!is_floating(q.dtype()) || q.dim()!=4 || q.numel()==0 || k.shape()!=q.shape() ||
      v.dim()!=4 || v.numel()==0 || v.size(0)!=q.size(0) || v.size(1)!=q.size(1) ||
      v.size(2)!=q.size(2) || relative.shape()!=Shape{q.size(2),q.size(3)}) {
    throw std::invalid_argument("relative attention: incompatible tensor geometry");
  }
  const auto time=q.size(2);
  const auto by_distance=matmul(q,relative,false,true).reshape({q.size(0),q.size(1),time*time});
  std::vector<std::int64_t> ids;
  for (std::int64_t i=0;i<time;++i) {
    for (std::int64_t j=0;j<time;++j) ids.push_back(i*time+std::max<std::int64_t>(0,i-j));
  }
  AttentionOptions options;
  options.causal=true;
  options.need_weights=true;
  options.mask=gather(by_distance,from_vector(ids,{time,time}),2)/std::sqrt(double(q.size(3)));
  return scaled_dot_product_attention(q,k,v,options);
}

} // namespace datamunge::nn
