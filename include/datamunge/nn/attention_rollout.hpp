#pragma once

/// @file attention_rollout.hpp
/// @brief Residual attention rollout across self-attention layers.

#include <stdexcept>
#include <vector>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Residual transitions and their cumulative products.
/// @par Example
/// @code
/// dm::nn::AttentionRollout result;
/// @endcode
struct AttentionRollout {
  std::vector<Tensor> transitions; ///< Normalized [batch,tokens,tokens] map for each layer.
  std::vector<Tensor> cumulative; ///< Transition products through each layer, in input order.
};

/// @brief Trace attention through residual self-attention layers.
/// Averages heads, adds the identity and normalizes over input tokens.
/// Each layer's transition multiplies the preceding cumulative map on the left.
/// @param layers Nonempty earlier-to-later sequence of nonnegative attention
/// weights [batch,heads,tokens,tokens] with shared batch, token count and floating dtype.
/// @return Each residual transition and cumulative rollout, retaining the batch axis.
/// @throws std::invalid_argument For incompatible or empty shapes or dtypes.
/// @par Example
/// @code
/// auto result=dm::nn::attention_rollout({dm::nn::full({1,2,3,3},1./3)});
/// @endcode
/// @par References
/// - Abnar and Zuidema (2020), Quantifying Attention Flow in Transformers.
///   https://aclanthology.org/2020.acl-main.385/
/// @see multi_head_attention()
inline AttentionRollout attention_rollout(const std::vector<Tensor>& layers) {
  if (layers.empty() || !layers.front().defined() || layers.front().dim()!=4)
    throw std::invalid_argument("Expected a nonempty sequence of attention layers");
  const auto batch=layers.front().size(0), tokens=layers.front().size(2);
  const auto dtype=layers.front().dtype();
  if (batch<1 || tokens<1 || (dtype!=DType::Float32 && dtype!=DType::Float64))
    throw std::invalid_argument("Expected positive attention dimensions and floating weights");
  const auto identity=eye(tokens,dtype);
  AttentionRollout out;
  for (const auto& layer:layers) {
    if (!layer.defined() || layer.dim()!=4 || layer.size(0)!=batch || layer.size(1)<1 ||
        layer.size(2)!=tokens || layer.size(3)!=tokens || layer.dtype()!=dtype)
      throw std::invalid_argument("Incompatible attention weights");
    const auto with_skip=mean(layer,{1})+identity;
    const auto transition=with_skip/sum(with_skip,{2},true);
    out.transitions.push_back(transition);
    out.cumulative.push_back(out.cumulative.empty()?transition:matmul(transition,out.cumulative.back()));
  }
  return out;
}

}  // namespace datamunge::nn
