#pragma once
/// @file retrieval_models.hpp
/// @brief Learned generalized-mean pooling and VGG-style unit-descriptor encoding,without hidden fitting.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <cmath>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Pool spatial feature activations by a differentiable generalized mean.
///
/// Spatial H/W are reduced equally. Values below the epsilon clamp have
/// zero input gradient;the equality boundary follows core maximum ties.
/// p=1 gives the clamped spatial mean. No parameters,mutation or fitting.
///
/// @param input Float32/64 positive[B,H,W,C] geometry,signed activations allowed.
/// @param exponent Same-dtype finite positive singleton pooling power.
/// @param epsilon Finite positive activation clamp,default1e-6.
/// @return Same-type[B,C] mean(max(input,epsilon)^p)^(1/p),with input and p gradients.
/// @throws std::invalid_argument For incompatible rank,shape,dtype,nonpositive exponent,epsilon or model dimensions.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto x = n::ones({2, 2, 3, 4}).requires_grad_();
/// auto p = n::full({}, 3.).requires_grad_();
/// const auto pooled = n::models::GeneralizedMean(x, p);
/// const auto gradient = n::grad(n::sum(pooled), {x, p});
/// @endcode
/// @par References
/// - Radenovic,Tolias and Chum(2018),Fine-tuning CNN Image Retrieval with No Human Annotation, https://arxiv.org/abs/1711.02512;author implementation https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/functional.py. From-scratch VGG-style compact CIFAR category retrieval and p=1+softplus(rho) are explicit teaching adaptations.
/// @see GeneralizedMean(), RetrievalEncoder()
inline nn::Tensor GeneralizedMean(const nn::Tensor &input, const nn::Tensor &exponent,
                                  float epsilon = 1e-6f) {
  if (input.dim() != 4 || input.shape()[0] < 1 || input.shape()[1] < 1 ||
      input.shape()[2] < 1 || input.shape()[3] < 1 ||
      (input.dtype() != nn::DType::Float32 && input.dtype() != nn::DType::Float64) ||
      exponent.numel() != 1 || exponent.dtype() != input.dtype() ||
      !std::isfinite(exponent.item()) || exponent.item() <= 0 || !std::isfinite(epsilon) ||
      epsilon <= 0)
    throw std::invalid_argument("invalid generalized mean tensor,exponent or epsilon");
  const nn::Tensor clamped = nn::maximum(input, nn::scalar_like(double(epsilon), input));
  const auto power = exponent.reshape({});
  return nn::pow(nn::mean(nn::pow(clamped, power), {1, 2}), nn::div(1.0, power));
}

/// @brief Normalize descriptor rows with the source division/square-root arithmetic.
///
/// Zero rows remain zero with finite norm-floor gradients. No fitting,
/// mutation,optimizer or learned state.
///
/// @param input Float32/64[B,D],D positive,empty B allowed.
/// @return Same-type[B,D] input/sqrt(max(sum(input²),1e-12)),retaining gradients.
/// @throws std::invalid_argument For incompatible rank,shape,dtype,nonpositive exponent,epsilon or model dimensions.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto descriptor = n::models::DescriptorUnit(n::tensor({3., 4.}, {1, 2}));
/// @endcode
/// @par References
/// - Radenovic,Tolias and Chum(2018),Fine-tuning CNN Image Retrieval with No Human Annotation, https://arxiv.org/abs/1711.02512;author implementation https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/functional.py. From-scratch VGG-style compact CIFAR category retrieval and p=1+softplus(rho) are explicit teaching adaptations.
/// @see GeneralizedMean(), RetrievalEncoder()
inline nn::Tensor DescriptorUnit(const nn::Tensor &input) {
  if (input.dim() != 2 || input.shape()[1] < 1 ||
      (input.dtype() != nn::DType::Float32 && input.dtype() != nn::DType::Float64))
    throw std::invalid_argument("floating descriptor matrix required");
  return input / nn::sqrt(nn::maximum(nn::sum(nn::square(input), {1}, true),
                                      nn::scalar_like(double(1e-12f), input)));
}

/// @brief Own normalized descriptors,pooled/raw features,the learned power and19 stage maps.
///
/// Tensor handles retain input/store graphs. No optimizer,BN moments or
/// separate trained state is owned by this result.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(227);
/// const auto out = n::models::RetrievalEncoder(p, n::ones({2, 32, 32, 3}), 8, 32);
/// @endcode
/// @par References
/// - Radenovic,Tolias and Chum(2018),Fine-tuning CNN Image Retrieval with No Human Annotation, https://arxiv.org/abs/1711.02512;author implementation https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/functional.py. From-scratch VGG-style compact CIFAR category retrieval and p=1+softplus(rho) are explicit teaching adaptations.
/// @see RetrievalEncoder()
struct RetrievalResult {
  nn::Tensor descriptor; ///< Floating[B,dimensions] norm-floored unit output.
  nn::Tensor pooled;     ///< Floating[B,512/divisor] GeM output.
  nn::Tensor raw;        ///< Floating[B,dimensions] bias-free linear projection.
  nn::Tensor exponent;   ///< Floating scalar1+softplus(gem/rho),initialized near3.
  std::vector<nn::Tensor>
      stages; ///< Thirteen conv maps,four pools,pooled features,descriptor.
};
/// @brief Encode observed RGB with thirteen convolutions,learned GeM and a unit projection.
///
/// Subtract.5;thirteen biased He-normal SAME ReLU convolutions,four VALID
/// 2x2 stride2 pools;shared learned p=1+softplus(rho),initially3;bias-free
/// normal projection with SD sqrt(1/input width);unit normalization. No
/// pretraining,BN,whitening,dropout or optimizer. Warm every leaf,including
/// rho and projection,before constructing the caller optimizer.
///
/// @param p Same-dtype learned store,creates/reuses28 leaves including gem/rho.
/// @param rgb Float32/64[B,H,W,3],B positive,H/W at least16.
/// @param divisor Positive exact divisor of64,default1;compact8.
/// @param dimensions Positive descriptor width,default128;compact32.
/// @return Unit descriptors,intermediate features,learned pooling power and19 stages.
/// @throws std::invalid_argument For incompatible rank,shape,dtype,nonpositive exponent,epsilon or model dimensions.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(227);
/// const auto out = n::models::RetrievalEncoder(p, n::ones({2, 32, 32, 3}), 8, 32);
/// @endcode
/// @par References
/// - Radenovic,Tolias and Chum(2018),Fine-tuning CNN Image Retrieval with No Human Annotation, https://arxiv.org/abs/1711.02512;author implementation https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/functional.py. From-scratch VGG-style compact CIFAR category retrieval and p=1+softplus(rho) are explicit teaching adaptations.
/// @see GeneralizedMean(), RetrievalEncoder()
inline RetrievalResult RetrievalEncoder(nn::ParameterStore &p, const nn::Tensor &rgb,
                                        int divisor = 1, int dimensions = 128) {
  if (divisor < 1 || 64 % divisor || dimensions < 1)
    throw std::invalid_argument(
        "retrieval divisor must divide64;positive descriptor dimensions");
  if (rgb.dim() != 4 || rgb.shape()[0] < 1 || rgb.shape()[1] < 16 || rgb.shape()[2] < 16 ||
      rgb.shape()[3] != 3 || rgb.dtype() != p.dtype())
    throw std::invalid_argument("invalid retrieval RGB input");
  RetrievalResult result;
  nn::Tensor x = rgb - .5;
  nn::ConvLayerOptions config;
  config.init = nn::init::he_normal();
  config.activation = nn::Activation::Relu;
  const int counts[] = {2, 2, 3, 3, 3}, widths[] = {64, 128, 256, 512, 512};
  for (int block = 0; block < 5; ++block) {
    for (int layer = 0; layer < counts[block]; ++layer) {
      x = nn::conv2d(p, "block" + std::to_string(block) + "/conv" + std::to_string(layer), x,
                     widths[block] / divisor, 3, config);
      result.stages.push_back(x);
    }
    if (block < 4) {
      x = nn::max_pool2d(x, 2, 2);
      result.stages.push_back(x);
    }
  }
  const std::int64_t in = x.size(3);
  const nn::Tensor rho =
      p.get("gem/rho", {}, nn::init::constant(double(std::log(std::expm1(2.f)))));
  result.exponent = 1.0 + nn::softplus(rho);
  result.pooled = GeneralizedMean(x, result.exponent);
  result.stages.push_back(result.pooled);
  const nn::Tensor weights =
      p.get("projection/W", {in, dimensions}, nn::init::normal(std::sqrt(1.f / float(in))));
  result.raw = nn::matmul(result.pooled, weights);
  result.descriptor = DescriptorUnit(result.raw);
  result.stages.push_back(result.descriptor);
  return result;
}

} // namespace datamunge::nn::models
