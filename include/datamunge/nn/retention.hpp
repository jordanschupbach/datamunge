/// @file
/// @brief Causal multi-scale retention with optional chunkwise execution.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Apply causal exponentially decayed query-key products to values.
///
/// Computes the sum of (query[t] dot key[j]) times decay^(t-j) times
/// value[j] for j <= t. There is no softmax or implicit key scaling.
/// Chunked execution carries a discounted key-value outer-product sum
/// between chunks and combines it with parallel retention within each
/// chunk. A chunk size of one gives token-by-token recurrence.
/// @param query CPU Float32 or Float64 queries [batch, heads, time, key_width].
/// @param key Keys with the same shape and dtype as query.
/// @param value CPU values [batch, heads, time, value_width] with matching dtype.
/// @param decay One finite coefficient in [0,1] per head.
/// @param chunk_size Zero for the full parallel form, or a positive chunk length.
/// @return Differentiable retained values [batch, heads, time, value_width].
/// @throws std::invalid_argument Incompatible shapes, devices, dtypes, or decay.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::ones({1,1,2,1});
/// const std::array decay{0.5};
/// std::print("{:.1f}\n", n::causal_retention(x,x,x,decay,1).to_vector<float>().back()); // 1.5
/// @endcode
/// @par References
/// - Sun et al. (2023), Retentive Network. https://arxiv.org/abs/2307.08621
/// @see matmul(), apply_rotary()
inline Tensor causal_retention(const Tensor& query, const Tensor& key, const Tensor& value,
                               std::span<const double> decay, std::int64_t chunk_size = 0) {
  if (!query.defined() || !key.defined() || !value.defined() || query.dim() != 4 ||
      key.shape() != query.shape() || value.dim() != 4 ||
      value.size(0) != query.size(0) || value.size(1) != query.size(1) ||
      value.size(2) != query.size(2) || query.numel() == 0 || value.numel() == 0 ||
      !is_floating(query.dtype()) || key.dtype() != query.dtype() || value.dtype() != query.dtype() ||
      query.device().backend != compute::Backend::CPU || key.device().backend != compute::Backend::CPU ||
      value.device().backend != compute::Backend::CPU || chunk_size < 0 ||
      decay.size() != static_cast<std::size_t>(query.size(1)) ||
      std::any_of(decay.begin(),decay.end(),[](double d){return !(d>=0 && d<=1);})) {
    throw std::invalid_argument("causal_retention: incompatible inputs");
  }
  const auto [batch, heads, time, width] = std::array{
      query.size(0),query.size(1),query.size(2),query.size(3)};
  const auto tensor = [&](const std::vector<double>& data,const Shape& shape) {
    return from_vector(data,shape).to(query.dtype());
  };
  const auto within = [&](const Tensor& q,const Tensor& k,const Tensor& v) {
    const auto count = q.size(2);
    std::vector<double> weights;
    for (double d : decay) {
      for (std::int64_t i=0;i<count;++i) {
        for (std::int64_t j=0;j<count;++j) weights.push_back(j>i ? 0.0 : std::pow(d,i-j));
      }
    }
    return matmul(matmul(q,k,false,true)*tensor(weights,{heads,count,count}),v);
  };
  if (chunk_size == 0 || chunk_size >= time) return within(query,key,value);
  auto state = zeros({batch,heads,width,value.size(3)},query.dtype());
  std::vector<Tensor> outputs;
  for (std::int64_t start=0;start<time;) {
    const auto count=std::min(chunk_size,time-start);
    const auto cut=[&](const Tensor& x) {
      return slice(x,{0,0,start,0},{batch,heads,count,x.size(3)});
    };
    const auto q=cut(query), k=cut(key), v=cut(value);
    std::vector<double> prefix,suffix,carry;
    for (double d : decay) {
      carry.push_back(std::pow(d,count));
      for (std::int64_t i=0;i<count;++i) {
        prefix.push_back(std::pow(d,i+1));
        suffix.push_back(std::pow(d,count-i-1));
      }
    }
    outputs.push_back(within(q,k,v)+matmul(q*tensor(prefix,{heads,count,1}),state));
    state=state*tensor(carry,{heads,1,1})+matmul(k*tensor(suffix,{heads,count,1}),v,true,false);
    start+=count;
  }
  return concat(outputs,2);
}

} // namespace datamunge::nn
