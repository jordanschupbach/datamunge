#pragma once

/// @file wav2vec2.hpp
/// @brief Compact masked audio contrastive pretraining with product quantization.

#include <stdexcept>

#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Training objectives and diagnostics for the compact wav2vec model.
/// @par Example
/// @code
/// dm::nn::Wav2Vec2Result result;
/// @endcode
struct Wav2Vec2Result {
  Tensor loss; ///< Contrastive loss plus 0.1 times diversity penalty.
  Tensor contrastive; ///< Mean positive-pair cross-entropy over masked frames.
  Tensor diversity; ///< One minus normalized mean codebook perplexity.
  Tensor latent; ///< Acoustic encoder output [batch,8,12].
  Tensor masked; ///< Context input after replacing selected frames [batch,8,12].
  Tensor assignment_logits; ///< Raw code scores [batch*8*2,8].
  Tensor ids; ///< Discrete code indices [batch,8,2].
  Tensor assignment; ///< Straight-through assignments [batch*8*2,8].
  Tensor product; ///< Concatenated codebook vectors [batch,8,12].
  Tensor query; ///< Context predictions [batch,8,12].
  Tensor target; ///< Projected quantized targets [batch,8,12].
  Tensor logits; ///< Duplicate-suppressed contrastive scores [batch,8,8].
};

/// @brief Fit masked acoustic context against straight-through quantized targets.
/// Frames are nonoverlapping groups of twelve samples. The context network
/// sees learned replacements at masked positions. Within each clip, each
/// masked frame is contrasted against other frames with different code pairs.
/// @param model Trainable parameter store initialized on first use.
/// @param waveform Floating [batch,96] input in store dtype.
/// @param mask Binary floating [batch,8,1] with at least one selected frame.
/// @param noise Additive Gumbel draws [batch*8*2,8], or zeros for deterministic evaluation.
/// @return Differentiable objectives and forward diagnostics.
/// @throws std::invalid_argument For invalid shapes, dtypes, masks or parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// auto result = dm::nn::wav2vec2_pretraining(model,dm::nn::zeros({1,96}),
///     dm::nn::ones({1,8,1}),dm::nn::zeros({16,8}));
/// @endcode
/// @par References
/// - Baevski et al. (2020), wav2vec 2.0. https://arxiv.org/abs/2006.11477
/// @see multi_head_attention()
inline Wav2Vec2Result wav2vec2_pretraining(ParameterStore& model, const Tensor& waveform,
                                          const Tensor& mask,const Tensor& noise) {
  if (!waveform.defined() || !mask.defined() || !noise.defined() ||
      waveform.dim()!=2 || waveform.size(0)<1 || waveform.size(1)!=96 ||
      mask.shape()!=Shape{waveform.size(0),8,1} || noise.shape()!=Shape{waveform.size(0)*16,8} ||
      waveform.dtype()!=model.dtype() || mask.dtype()!=model.dtype() || noise.dtype()!=model.dtype() ||
      sum(equal(mask,0).to(DType::Int32)+equal(mask,1).to(DType::Int32)).item()!=mask.numel() || sum(mask).item()<=0) {
    throw std::invalid_argument("Invalid wav2vec pretraining batch");
  }
  const auto batch=waveform.size(0);
  const auto latent=dense(model,"feature",reshape(waveform,{batch,8,12}),12,Activation::Tanh);
  const auto token=model.get("mask_token",{12},init::uniform(.15));
  const auto position=model.get("position",{8,12},init::uniform(.12));
  const auto masked=latent*(1-mask)+token*mask;
  const auto context=masked+position;
  const auto h=layer_norm(model,"attention_norm",
      context+multi_head_attention(model,"attention",context,context,2).output);
  const auto ff=dense(model,"ff",h,24,Activation::Relu);
  const auto query=dense(model,"query",layer_norm(model,"ff_norm",h+dense(model,"ff_out",ff,12)),12);
  const auto grouped=reshape(dense(model,"assignment",latent,16),{-1,8});
  const auto noisy=grouped+noise;
  const auto soft=softmax(noisy/.7);
  const auto ids=reshape(argmax(noisy),{batch,8,2});
  const auto assignment=one_hot(reshape(ids,{-1}),8,1,0,waveform.dtype())+(soft-soft.detach());
  const auto table=model.get("codebook",{2,8,6},init::uniform(.35));
  const auto by_group=transpose(reshape(assignment,{-1,2,8}),{1,0,2});
  const auto product=reshape(transpose(matmul(by_group,table),{1,0,2}),{batch,8,12});
  const auto target=dense(model,"target",product,12);
  /// Normalize embeddings for cosine comparisons with a squared-norm floor.
  const auto unit=[](const Tensor& x) { return x/sqrt(sum(square(x),{2},true)+1e-8); };
  const auto similarity=matmul(unit(query),transpose(unit(target),{0,2,1}))/.2;
  const auto duplicate=equal(sum(equal(expand_dims(ids,1),expand_dims(ids,2)).to(DType::Int32),{3}),2);
  const auto diagonal=eye(8,waveform.dtype());
  const auto suppress=equal(duplicate.to(DType::Int32)+equal(diagonal,0).to(DType::Int32),2);
  const auto logits=where(suppress,-1e9,similarity);
  const auto contrastive=-sum(sum(diagonal*log_softmax(logits),{2},true)*mask)/sum(mask);
  const auto average=mean(reshape(softmax(grouped),{-1,2,8}),{0});
  const auto entropy=-sum(average*log(average+1e-8),{1});
  const auto diversity=1-mean(exp(entropy))/8;
  return {contrastive+.1*diversity,contrastive,diversity,latent,masked,grouped,ids,
          assignment,product,query,target,logits};
}

}  // namespace datamunge::nn
