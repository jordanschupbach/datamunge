#pragma once

/// @file ft_transformer.hpp
/// @brief Feature-tokenizing transformer for tabular regression.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Live outputs from a tabular feature transformer.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(42);
/// auto out = n::ft_transformer_regression(p,"ft",n::zeros({2,3}),{4});
/// auto token_count = out.tokens.size(1); // 4
/// @endcode
/// @see ft_transformer_regression()
struct FtTransformerRegression {
  Tensor prediction; ///< Scalar responses [N,1].
  Tensor tokens; ///< Feature tokens followed by aggregate token [N,D+1,W].
  std::vector<Tensor> attention; ///< One attention tensor [N,H,D+1,D+1] per block.
};

/// @brief Apply an FT-Transformer with affine numeric tokens and categorical embeddings.
///
/// Numeric fields come first. Categorical values are integer IDs represented
/// in the same tensor and have a field-specific vocabulary. Each block uses
/// self-attention, residual connections, and a ReGLU feed-forward layer.
/// The first attention sublayer skips input normalization. A normalized
/// aggregate token drives the scalar prediction head. Reuse the same
/// parameter prefix only with the same architecture.
/// @param parameters Store owning the trainable weights.
/// @param name Parameter prefix.
/// @param input Nonempty [N,D] floating features with categories after numeric fields.
/// @param vocabularies Positive vocabulary size for each trailing categorical field. Empty for all-numeric.
/// @param width Positive token width divisible by heads.
/// @param heads Positive self-attention head count.
/// @param blocks Positive transformer block count.
/// @param hidden Positive ReGLU hidden width.
/// @return Differentiable prediction, feature tokens and attention weights.
/// @throws std::invalid_argument Invalid architecture, geometry or categorical ID.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(42);
/// auto out = n::ft_transformer_regression(p,"ft",n::zeros({2,3}),{4},8,2,1,12);
/// auto prediction = out.prediction; // Shape [2,1]
/// @endcode
/// @par References
/// - Gorishniy et al. (2021), Revisiting Deep Learning Models for Tabular Data.
///   https://arxiv.org/abs/2106.11959
/// @see multi_head_attention(), FtTransformerRegression
inline FtTransformerRegression ft_transformer_regression(
    ParameterStore& parameters, const std::string& name, const Tensor& input,
    const std::vector<std::int64_t>& vocabularies = {}, int width = 16, int heads = 2,
    int blocks = 2, int hidden = 24) {
  if (!input.defined() || input.dim() != 2 || input.size(0) == 0 || input.size(1) == 0 ||
      vocabularies.size() > static_cast<std::size_t>(input.size(1)) ||
      width <= 0 || heads <= 0 || width % heads != 0 || blocks <= 0 || hidden <= 0)
    throw std::invalid_argument("invalid FT-Transformer geometry");
  const auto rows = input.size(0), fields = input.size(1);
  const auto numeric = fields - static_cast<std::int64_t>(vocabularies.size());
  std::vector<Tensor> tokens;
  for (std::int64_t i = 0; i < fields; ++i) {
    const auto x = slice(input,{0,i},{rows,1});
    if (i < numeric) {
      tokens.push_back(dense(parameters,name+"/numeric_"+std::to_string(i),x,width));
    } else {
      const auto prefix = name+"/category_"+std::to_string(i-numeric);
      const auto vocab = vocabularies[i-numeric];
      const auto labels = x.reshape({rows});
      const auto valid = logical_and(greater_equal(labels,0.),less(labels,vocab));
      if (vocab <= 0 || !all(logical_and(valid,equal(labels,floor(labels)))).item<bool>())
        throw std::invalid_argument("invalid FT-Transformer category IDs");
      const auto table = embedding(parameters,prefix,labels.to(DType::Int64),vocab,width,.2);
      tokens.push_back(table+parameters.get(prefix+"/bias",{width},init::zeros()));
    }
  }
  const auto cls = parameters.get(name+"/cls",{1,width},init::normal(.15));
  tokens.push_back(broadcast_to(cls,{rows,width}));
  FtTransformerRegression out;
  out.tokens = stack(tokens,1);
  auto state = out.tokens;
  MultiHeadOptions options;
  options.need_weights = true;
  for (int block = 0; block < blocks; ++block) {
    const auto prefix = name+"/block_"+std::to_string(block);
    const auto input = block == 0 ? state : layer_norm(parameters,prefix+"/attention_norm",state);
    auto attended = multi_head_attention(parameters,prefix+"/attention",input,input,heads,options);
    out.attention.push_back(attended.weights);
    const auto residual = state+attended.output;
    const auto norm = layer_norm(parameters,prefix+"/ff_norm",residual);
    const auto wide = dense(parameters,prefix+"/expand",norm,2LL*hidden);
    const auto left = slice(wide,{0,0,0},{rows,fields+1,hidden});
    const auto right = slice(wide,{0,0,hidden},{rows,fields+1,hidden});
    state = residual+dense(parameters,prefix+"/contract",left*relu(right),width);
  }
  const auto aggregate = slice(state,{0,fields,0},{rows,1,width}).reshape({rows,width});
  out.prediction = dense(parameters,name+"/response",
                         relu(layer_norm(parameters,name+"/final_norm",aggregate)),1);
  return out;
}
}  // namespace datamunge::nn
