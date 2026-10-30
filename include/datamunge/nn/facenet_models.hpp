#pragma once
/// @file facenet_models.hpp
/// @brief NN1-style convolution/maxout/unit embeddings with no hidden optimizer.
#include "datamunge/nn/classic_cnn_components.hpp"
#include "datamunge/nn/layers.hpp"
#include <array>
#include <cmath>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Normalize each embedding row with a squared-norm floor.
///
/// Zero rows stay zero. Below the floor,the Jacobian is identity/sqrt(epsilon);
/// above it,the radial derivative vanishes. No fitting or mutation.
///
/// @param x Float32/64[B,D],D positive,empty B allowed.
/// @param epsilon Finite positive squared-norm floor,default1e-12.
/// @return Same-type[B,D] x/sqrt(max(sum(x*x),epsilon)),retaining gradients.
/// @throws std::invalid_argument For incompatible rank,geometry,dtype,invalid configuration or nonpositive epsilon.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::UnitEmbedding(n::tensor({3., 4.}, {1, 2}));
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. SAME padding,LRN constants,compact widths,He/Glorot initializers and Adam are explicit source teaching choices.
/// @see FaceNetNN1(), FaceNetConfig
inline nn::Tensor UnitEmbedding(const nn::Tensor &x, float epsilon = 1e-12f) {
  if (x.dim() != 2 || x.shape()[1] < 1 ||
      (x.dtype() != nn::DType::Float32 && x.dtype() != nn::DType::Float64) ||
      !std::isfinite(epsilon) || epsilon <= 0)
    throw std::invalid_argument("invalid unit embedding input or epsilon");
  return x * nn::rsqrt(nn::maximum(nn::sum(nn::square(x), {1}, true),
                                   nn::scalar_like(double(epsilon), x)));
}

/// @brief Take maxima over adjacent pairs of dense feature channels.
///
/// No parameters,optimizer or mutation;uses the core max tie convention.
///
/// @param x Floating[B,2*outputs] matrix.
/// @param outputs Positive output dimension.
/// @return Same-type[B,outputs] with gradients split equally at exact ties.
/// @throws std::invalid_argument For incompatible rank,geometry,dtype,invalid configuration or nonpositive epsilon.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::MaxoutPairs(n::tensor({1., 3., 4., 2.}, {1, 4}), 2);
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. SAME padding,LRN constants,compact widths,He/Glorot initializers and Adam are explicit source teaching choices.
/// @see FaceNetNN1(), FaceNetConfig
inline nn::Tensor MaxoutPairs(const nn::Tensor &x, std::int64_t outputs) {
  if (outputs < 1 || x.dim() != 2 || x.shape()[1] != outputs * 2 ||
      (x.dtype() != nn::DType::Float32 && x.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid maxout pair geometry");
  return nn::max(nn::reshape(x, {-1, outputs, 2}), {2});
}

/// @brief Configure eleven NN1 convolutions,two maxout layers and128D output.
///
/// Full defaults220pxRGB,undivided channels,two4096-unit maxouts.
/// Configuration owns no parameters,RNG,normalization state or fitting.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto c = n::models::FaceNetConfig::Compact();
/// const auto out = n::models::FaceNetNN1(p, n::ones({2, 28, 28, 1}), c);
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. SAME padding,LRN constants,compact widths,He/Glorot initializers and Adam are explicit source teaching choices.
/// @see FaceNetNN1(), FaceNetConfig
struct FaceNetConfig {
  int side = 220;    ///< Positive square input side.
  int input = 3;     ///< Positive input channel count.
  int divisor = 1;   ///< Positive exact divisor of every canonical convolution width.
  int hidden = 4096; ///< Positive units after each maxout.

  /// @brief Return same-stage compact28px grayscale NN1 settings.
  ///
  /// No fitting,RNG or mutation.
  ///
  /// @return Value configuration:side28,input1,divisor8,hidden128.
  /// @par Example
  /// @code
  /// const auto c = dm::nn::models::FaceNetConfig::Compact();
  /// @endcode
  /// @par References
  /// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. SAME padding,LRN constants,compact widths,He/Glorot initializers and Adam are explicit source teaching choices.
  /// @see FaceNetNN1(), FaceNetConfig
  static FaceNetConfig Compact() {
    return {28, 1, 8, 128};
  }
};
/// @brief Own raw128D embeddings,normalized embeddings and13 stage maps.
///
/// Tensor handles retain input/store autograd graphs. No separate learned
/// state is stored in this result.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto c = n::models::FaceNetConfig::Compact();
/// const auto out = n::models::FaceNetNN1(p, n::ones({2, 28, 28, 1}), c);
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. SAME padding,LRN constants,compact widths,He/Glorot initializers and Adam are explicit source teaching choices.
/// @see FaceNetNN1(), FaceNetConfig
struct FaceNetOutputs {
  nn::Tensor raw;                 ///< Floating[B,128] linear output.
  nn::Tensor embedding;           ///< Floating[B,128] squared-norm-floored unit output.
  std::vector<nn::Tensor> stages; ///< Eleven post-ReLU conv maps and two maxout outputs.
};
/// @brief Return the eleven canonical NN1 undivided output widths.
///
/// No fitting or RNG.
///
/// @return Immutable process-lifetime array reference,no mutation.
/// @par Example
/// @code
/// const auto &a = dm::nn::models::FaceNetWidths();
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. SAME padding,LRN constants,compact widths,He/Glorot initializers and Adam are explicit source teaching choices.
/// @see FaceNetNN1(), FaceNetConfig
inline const std::array<int, 11> &FaceNetWidths() {
  /// @brief Immutable canonical NN1 undivided output widths.
  static const std::array<int, 11> values = {64,  64,  192, 192, 384, 384,
                                             256, 256, 256, 256, 256};
  return values;
}
/// @brief Return the eleven canonical NN1 square convolution kernel sides.
///
/// No fitting or RNG.
///
/// @return Immutable process-lifetime array reference,no mutation.
/// @par Example
/// @code
/// const auto &a = dm::nn::models::FaceNetKernels();
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. SAME padding,LRN constants,compact widths,He/Glorot initializers and Adam are explicit source teaching choices.
/// @see FaceNetNN1(), FaceNetConfig
inline const std::array<int, 11> &FaceNetKernels() {
  /// @brief Immutable canonical NN1 square convolution kernel sides.
  static const std::array<int, 11> values = {7, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3};
  return values;
}
/// @brief Run the complete eleven-convolution NN1 unit-embedding encoder.
///
/// He-normal biased SAME ReLU convolutions,stride2 first;four SAME3x3
/// stride2 pools;two LRN(radius2,bias1,alpha1e-4,beta.75);two He-normal
/// biased p=2 maxout dense layers;Glorot biased linear128;unit normalization.
/// Compact108040 parameters;full220RGB140025664. No BN/dropout/optimizer.
/// Warm all learned leaves before constructing the caller optimizer.
///
/// @param p Same-dtype learned parameter store,creates/reuses28 leaves.
/// @param input Float32/64 positive[B,cfg.side,cfg.side,cfg.input].
/// @param cfg Valid positive dimensions and an exact divisor of every width.
/// @return Raw and normalized128D vectors plus13 stage tensors,with gradients.
/// @throws std::invalid_argument For incompatible rank,geometry,dtype,invalid configuration or nonpositive epsilon.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto c = n::models::FaceNetConfig::Compact();
/// const auto out = n::models::FaceNetNN1(p, n::ones({2, 28, 28, 1}), c);
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. SAME padding,LRN constants,compact widths,He/Glorot initializers and Adam are explicit source teaching choices.
/// @see FaceNetNN1(), FaceNetConfig
inline FaceNetOutputs FaceNetNN1(nn::ParameterStore &p, const nn::Tensor &input,
                                 const FaceNetConfig &cfg) {
  /// @brief Stable convolution parameter scopes in NN1 stage order.
  static const std::array<const char *, 11> names = {"conv1", "conv2a", "conv2", "conv3a",
                                                     "conv3", "conv4a", "conv4", "conv5a",
                                                     "conv5", "conv6a", "conv6"};
  if (cfg.divisor < 1 || cfg.side < 1 || cfg.input < 1 || cfg.hidden < 1)
    throw std::invalid_argument("FaceNet dimensions");
  if (input.dim() != 4 || input.shape()[0] < 1 || input.shape()[1] != cfg.side ||
      input.shape()[2] != cfg.side || input.shape()[3] != cfg.input ||
      input.dtype() != p.dtype())
    throw std::invalid_argument("FaceNet input geometry or dtype");
  FaceNetOutputs out;
  nn::Tensor x = input;
  for (int i = 0; i < 11; ++i) {
    if (FaceNetWidths()[i] % cfg.divisor)
      throw std::invalid_argument("FaceNet width divisor");
    nn::ConvLayerOptions c;
    c.activation = nn::Activation::Relu;
    c.init = nn::init::he_normal();
    c.stride = i == 0 ? 2 : 1;
    x = nn::conv2d(p, names[i], x, FaceNetWidths()[i] / cfg.divisor, FaceNetKernels()[i], c);
    out.stages.push_back(x);
    if (i == 2)
      x = LocalResponseNorm(x, 2, 1.0, 1e-4, .75);
    if (i == 0 || i == 2 || i == 4 || i == 10)
      x = nn::max_pool2d(x, 3, 2, nn::Padding::Same);
    if (i == 0)
      x = LocalResponseNorm(x, 2, 1.0, 1e-4, .75);
  }
  nn::DenseOptions dense;
  dense.init = nn::init::he_normal();
  x = MaxoutPairs(nn::dense(p, "fc1", nn::reshape(x, {x.size(0), -1}), cfg.hidden * 2, dense),
                  cfg.hidden);
  out.stages.push_back(x);
  x = MaxoutPairs(nn::dense(p, "fc2", x, cfg.hidden * 2, dense), cfg.hidden);
  out.stages.push_back(x);
  dense.init = nn::init::glorot_uniform();
  out.raw = nn::dense(p, "embedding", x, 128, dense);
  out.embedding = UnitEmbedding(out.raw);
  return out;
}

/// @brief Count NN1 learned scalars from layer geometry without allocating tensors.
///
/// SAME output sides use ceiling division. This metadata-only census does
/// not execute a full-resolution graph,allocate weights,fit or consume RNG.
///
/// @param cfg Valid positive dimensions and exact width divisor.
/// @param tensors Optional output pointer receiving28 leaf tensors,default null.
/// @return Int64 parameter count;writes tensors only after valid geometry.
/// @throws std::invalid_argument For incompatible rank,geometry,dtype,invalid configuration or nonpositive epsilon.
/// @par Example
/// @code
/// int tensors = 0;
/// const auto count = dm::nn::models::FaceNetCensus({}, &tensors);
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. SAME padding,LRN constants,compact widths,He/Glorot initializers and Adam are explicit source teaching choices.
/// @see FaceNetNN1(), FaceNetConfig
inline std::int64_t FaceNetCensus(const FaceNetConfig &cfg, int *tensors = nullptr) {
  if (cfg.side < 1 || cfg.input < 1 || cfg.divisor < 1 || cfg.hidden < 1)
    throw std::invalid_argument("FaceNet census dimensions");
  for (const auto width : FaceNetWidths())
    if (width % cfg.divisor)
      throw std::invalid_argument("FaceNet census width divisor");
  std::int64_t total = 0, channels = cfg.input, side = cfg.side;
  int count = 0;
  for (int i = 0; i < 11; ++i) {
    const std::int64_t next = FaceNetWidths()[i] / cfg.divisor, k = FaceNetKernels()[i];
    total += k * k * channels * next + next;
    count += 2;
    channels = next;
    if (i == 0)
      side = nn::conv_output_size(side, k, 2, 1, nn::Padding::Same);
    if (i == 0 || i == 2 || i == 4 || i == 10)
      side = nn::conv_output_size(side, 3, 2, 1, nn::Padding::Same);
  }
  const std::int64_t flat = side * side * channels, h = cfg.hidden;
  total += flat * 2 * h + 2 * h + h * 2 * h + 2 * h + h * 128 + 128;
  count += 6;
  if (tensors)
    *tensors = count;
  return total;
}

/// @brief Sum squared kernel leaves of rank greater than one.
///
/// Biases and one-dimensional affine/slope leaves are excluded from this
/// penalty,but the caller optimizer can still update them. No fitting. Empty
/// or bias-only stores produce a same-dtype scalar zero without leaf edges.
///
/// @param p Floating learned store,empty or bias-only stores allowed.
/// @return Same-store-dtype differentiable scalar kernel sum of squares.
/// @throws std::invalid_argument For incompatible rank,geometry,dtype,invalid configuration or nonpositive epsilon.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::dense(p, "head", n::ones({1, 3}), 2);
/// const auto penalty = n::models::WeightSquares(p);
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. SAME padding,LRN constants,compact widths,He/Glorot initializers and Adam are explicit source teaching choices.
/// @see FaceNetNN1(), FaceNetConfig
inline nn::Tensor WeightSquares(const nn::ParameterStore &p) {
  nn::Tensor total;
  for (const auto &v : p.parameters())
    if (v.dim() > 1)
      total = total.defined() ? total + nn::sum(nn::square(v)) : nn::sum(nn::square(v));
  return total.defined() ? total : nn::zeros({}, p.dtype());
}

} // namespace datamunge::nn::models
