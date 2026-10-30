#pragma once
/// @file siamese_models.hpp
/// @brief Shared four-convolution sigmoid embeddings and learned weighted-L1 Siamese logits,without hidden fitting.
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/layers.hpp"
#include <array>
#include <limits>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Configure the full105px Siamese encoder or a same-stage compact teaching model.
///
/// Full64/128/128/256 channels,kernels10/7/4/4 and4096 embedding units.
/// No learned parameters,optimizer or fitting are owned by configuration.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto cfg = n::models::SiameseConfig::Compact();
/// const auto out = n::models::SiameseEncoder(p, n::ones({2, 48, 48, 1}), cfg);
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. He/Glorot initialization,negative initial score weights,compact kernels and Adam are explicit teaching adaptations.
/// @see SiameseEncoder(), SiameseLogit()
struct SiameseConfig {
  int side =
      105; ///< Positive square grayscale input side,sufficient for all VALID layers/pools.
  int embedding = 4096; ///< Positive sigmoid embedding dimension.
  std::array<int, 4> channels = {64, 128, 128,
                                 256};        ///< Positive four-convolution output widths.
  std::array<int, 4> kernels = {10, 7, 4, 4}; ///< Positive four VALID square kernel sides.

  /// @brief Return the compact48px/four-stage64-dimensional Siamese settings.
  ///
  /// No parameters or RNG;does not mutate a full configuration.
  ///
  /// @return Independent value configuration with8/16/16/32 channels and5/3/3/2 kernels.
  /// @par Example
  /// @code
  /// const auto cfg = dm::nn::models::SiameseConfig::Compact();
  /// @endcode
  /// @par References
  /// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. He/Glorot initialization,negative initial score weights,compact kernels and Adam are explicit teaching adaptations.
  /// @see SiameseConfig
  static SiameseConfig Compact() {
    SiameseConfig c;
    c.side = 48;
    c.embedding = 64;
    c.channels = {8, 16, 16, 32};
    c.kernels = {5, 3, 3, 2};
    return c;
  }
};

/// @brief Own Siamese sigmoid embeddings and four pre-pooling ReLU stage maps.
///
/// Tensor handles retain the caller-store graph;twins reuse the same leaf
/// objects when called with the same store.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto cfg = n::models::SiameseConfig::Compact();
/// const auto out = n::models::SiameseEncoder(p, n::ones({2, 48, 48, 1}), cfg);
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. He/Glorot initialization,negative initial score weights,compact kernels and Adam are explicit teaching adaptations.
/// @see SiameseEncoder(), SiameseLogit()
struct SiameseFeatures {
  nn::Tensor embedding; ///< Floating[N,cfg.embedding] sigmoid outputs in[0,1].
  std::vector<nn::Tensor>
      stages; ///< Four post-ReLU convolution maps before each optional pool.
};
/// @brief Encode a grayscale image batch with four shared VALID convolutions and a sigmoid dense embedding.
///
/// Four He-normal biased convolutions/ReLU,pool2 after the first three,
/// then Glorot biased dense/sigmoid. Full model38947648 encoder scalars;
/// full weighted-L1 score head adds4097. Compact encoder24272 plus65 head
/// scalars=24337. No BN,dropout or optimizer. Reuse one p for both twins;
/// warm every encoder/head leaf before constructing the optimizer.
///
/// @param p Same-dtype floating learned store,creates/reuses encoder/* leaves.
/// @param images Positive[N,c.side,c.side,1] Float32/64 image batch.
/// @param c Valid positive widths/kernels/embedding and sufficient square image side.
/// @return Sigmoid embeddings and four stage maps,retaining every parameter/input gradient.
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,labels or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto cfg = n::models::SiameseConfig::Compact();
/// const auto out = n::models::SiameseEncoder(p, n::ones({2, 48, 48, 1}), cfg);
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. He/Glorot initialization,negative initial score weights,compact kernels and Adam are explicit teaching adaptations.
/// @see SiameseEncoder(), SiameseLogit()
inline SiameseFeatures SiameseEncoder(nn::ParameterStore &p, const nn::Tensor &images,
                                      const SiameseConfig &c) {
  if (c.side < 1 || c.embedding < 1 || images.dim() != 4 || images.shape()[0] < 1 ||
      images.shape()[1] != c.side || images.shape()[2] != c.side || images.shape()[3] != 1 ||
      images.dtype() != p.dtype())
    throw std::invalid_argument("invalid Siamese input/configuration");
  for (int i = 0; i < 4; ++i)
    if (c.channels[i] < 1 || c.kernels[i] < 1)
      throw std::invalid_argument("positive Siamese kernels/widths required");
  SiameseFeatures out;
  nn::Tensor x = images;
  int side = c.side;
  nn::ConvLayerOptions conv;
  conv.padding = nn::Padding::Valid;
  conv.activation = nn::Activation::Relu;
  conv.init = nn::init::he_normal();
  for (int i = 0; i < 4; ++i) {
    side -= c.kernels[i] - 1;
    if (side < 1)
      throw std::invalid_argument("Siamese image too small");
    x = nn::conv2d(p, "encoder/conv" + std::to_string(i), x, c.channels[i], c.kernels[i],
                   conv);
    out.stages.push_back(x);
    if (i < 3) {
      x = nn::max_pool2d(x, 2, 2);
      side /= 2;
    }
  }
  nn::DenseOptions dense;
  dense.activation = nn::Activation::Sigmoid;
  out.embedding = nn::dense(p, "encoder/embedding",
                            nn::reshape(x, {-1, std::int64_t(side) * side * c.channels[3]}),
                            c.embedding, dense);
  return out;
}

/// @brief Score symmetric absolute embedding differences with learned unconstrained coefficients.
///
/// Logit=sum_j W_j*abs(left_j-right_j)+b. Swap symmetry and identical-input
/// score b hold exactly. Coefficients remain unconstrained and may become
/// positive,so this learned similarity is not a guaranteed metric. At equal
/// coordinates abs has the core zero subgradient. No hidden fitting.
///
/// @param p Same-dtype learned store;similarity/W[dimensions,1] initially-.1,bias[1] initially.1.
/// @param left Floating[N,dimensions] left embeddings,N positive.
/// @param right Same-shape/type right embeddings.
/// @param dimensions Positive embedding dimension.
/// @return Floating[N] logits with gradients through both twins and score weights.
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,labels or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto cfg = n::models::SiameseConfig::Compact();
/// const auto out = n::models::SiameseEncoder(p, n::ones({2, 48, 48, 1}), cfg);
/// const auto scores = n::models::SiameseLogit(p, out.embedding, out.embedding, cfg.embedding);
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. He/Glorot initialization,negative initial score weights,compact kernels and Adam are explicit teaching adaptations.
/// @see SiameseEncoder(), SiameseLogit()
inline nn::Tensor SiameseLogit(nn::ParameterStore &p, const nn::Tensor &left,
                               const nn::Tensor &right, int dimensions) {
  if (dimensions < 1 || left.dim() != 2 || left.shape()[0] < 1 ||
      left.shape()[1] != dimensions || right.shape() != left.shape() ||
      left.dtype() != p.dtype() || right.dtype() != p.dtype())
    throw std::invalid_argument("invalid Siamese embedding pair");
  const nn::Tensor weights = p.get("similarity/W", {dimensions, 1}, nn::init::constant(-.1));
  const nn::Tensor bias = p.get("similarity/b", {1}, nn::init::constant(.1));
  return nn::reshape(nn::bias_add(nn::matmul(nn::abs(left - right), weights), bias), {-1});
}

/// @brief Compute elementwise stable sigmoid cross entropy with the source zero-logit derivative.
///
/// Sign-selected stable arithmetic avoids exponential overflow and gives
/// .5-label at zero. Labels receive no gradient. No reduction,optimizer,
/// mutation or hidden fitting.
///
/// @param logits Float32/64 tensor of any shape,empty allowed.
/// @param labels Same-shape/type finite soft targets in[0,1],detached.
/// @return Same-shape softplus(logits)-labels*logits values and derivative sigmoid(logit)-label.
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,labels or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto z = n::tensor({-100., 0., 100.}).requires_grad_();
/// const auto loss = n::models::SiameseBinaryCrossEntropy(z, n::tensor({0., 1., 1.}));
/// const auto g = n::grad(n::sum(loss), {z});
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. He/Glorot initialization,negative initial score weights,compact kernels and Adam are explicit teaching adaptations.
/// @see SiameseLogit()
inline nn::Tensor SiameseBinaryCrossEntropy(const nn::Tensor &logits,
                                            const nn::Tensor &labels) {
  if (logits.shape() != labels.shape() || logits.dtype() != labels.dtype() ||
      (logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64) ||
      !nn::all(nn::logical_and(nn::greater_equal(labels, 0.), nn::less_equal(labels, 1.)))
           .item<bool>())
    throw std::invalid_argument("invalid Siamese BCE inputs");
  const auto target = labels.detach();
  const nn::Tensor positive = nn::greater_equal(logits, 0.0);
  const nn::Tensor negative_magnitude = nn::where(positive, -logits, logits);
  const nn::Tensor linear = nn::where(positive, logits * (1.0 - target), -(logits * target));
  return linear + nn::log1p(nn::exp(negative_magnitude));
}

} // namespace datamunge::nn::models
