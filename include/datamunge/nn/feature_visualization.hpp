#pragma once
/// @file feature_visualization.hpp
/// @brief Tensor selected-channel objectives,explicit image priors and uniformnoise starts.
#include "datamunge/nn/ops.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Own selected-feature response and the two explicit image-prior penalties.
///
/// All terms remain differentiable tensors. Per-image terms are[B],mean is
/// scalar. Neither clipping nor optimizer updates occur in this objective helper.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({2, 4, 4, 1});
/// const auto o = m::VisualizeFeature(x, n::concat({x, x * 2.}, 3), n::ones({2}, n::DType::Int32),
///                                    2, .1f, .2f);
/// @endcode
/// @par References
/// - Olah,Mordvintsev and Schubert(2017),Feature Visualization, https://distill.pub/2017/feature-visualization/,doi:10.23915/distill.00007. Repository adaptation maximizes frozen MNISTconv2channel means with explicit centeredL2 and squared-TVpriors,uniformnoise starts and bounded RMS-normalized TensorSGD/backtracking;not a pretrained GoogLeNet feature atlas.
/// @see VisualizeFeature(),DreamAscent()
struct FeatureObjective {
  Tensor feature;   ///< Selected post-ReLUchannel spatial mean[B].
  Tensor l2;        ///< Mean centered pixel square(x-.5)^2[B].
  Tensor tv;        ///< Sum of mean squared vertical and horizontal differences[B].
  Tensor per_image; ///< feature-l2_weight*l2-tv_weight*tv[B].
  Tensor mean;      ///< Scalar mean objective over cases;ordinary derivative has1/Bfactor.
};

/// @brief Evaluate selected-channel spatial activation minus explicit centeredL2 and squared-TVpriors.
///
/// Channel selection uses one-hot tensor metadata;image priors normalize each
/// axis by its actual number of edges. Squared-TV here is not absolute/isotropic
/// TV. Batchmean ordinary derivative includes1/B;RMS step normalization belongs
/// to the explicit calling optimizer. No model fitting or gradient step hidden.
///
/// @param image Floating[B,H,W,1]image,H/W>=2.
/// @param activations Floating[B,Ha,Wa,count]post-ReLUfeature maps of the SAMEimage.
/// @param channels Int32[B]fixed selected feature channel per image.
/// @param count Positive actual feature-channel count.
/// @param l2_weight Finite nonnegative coefficient of mean(image-.5)^2.
/// @param tv_weight Finite nonnegative coefficient of mean(dy^2)+mean(dx^2).
/// @return Selected responses,L2,TV,per-case and batch-mean objective tensors.
/// @throws std::invalid_argument For invalid channel/image/activation geometry,target range or prior coefficients.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({2, 4, 4, 1});
/// const auto o = m::VisualizeFeature(x, n::concat({x, x * 2.}, 3), n::ones({2}, n::DType::Int32),
///                                    2, .1f, .2f);
/// @endcode
/// @par References
/// - Olah,Mordvintsev and Schubert(2017),Feature Visualization, https://distill.pub/2017/feature-visualization/,doi:10.23915/distill.00007. Repository adaptation maximizes frozen MNISTconv2channel means with explicit centeredL2 and squared-TVpriors,uniformnoise starts and bounded RMS-normalized TensorSGD/backtracking;not a pretrained GoogLeNet feature atlas.
/// @see VisualizeFeature(),DreamAscent()
inline FeatureObjective VisualizeFeature(const Tensor &image, const Tensor &activations,
                                         const Tensor &channels, int count, float l2_weight,
                                         float tv_weight) {
  if (image.dim() != 4 || image.size(0) < 1 || image.size(1) < 2 || image.size(2) < 2 ||
      image.size(3) != 1 || activations.dim() != 4 || activations.size(0) != image.size(0) ||
      activations.size(3) != count || channels.shape() != nn::Shape{image.size(0)} ||
      channels.dtype() != nn::DType::Int32 || count < 1 || nn::min(channels).item() < 0 ||
      nn::max(channels).item() >= count || !std::isfinite(l2_weight) ||
      !std::isfinite(tv_weight) || l2_weight < 0 || tv_weight < 0)
    throw std::invalid_argument(
        "aligned selectedchannels and nonnegative finite image priors required");
  FeatureObjective out;
  const Tensor spatial = nn::mean(activations, {1, 2});
  out.feature =
      nn::sum(nn::mul(spatial, nn::one_hot(channels, count, 1.0, 0.0, spatial.dtype())), {1});
  out.l2 = nn::mean(nn::square(nn::sub(image, double(.5f))), {1, 2, 3});
  const auto h = image.size(1), w = image.size(2);
  const Tensor dy =
      nn::sub(nn::slice_axis(image, 1, 1, h), nn::slice_axis(image, 1, 0, h - 1));
  const Tensor dx =
      nn::sub(nn::slice_axis(image, 2, 1, w), nn::slice_axis(image, 2, 0, w - 1));
  out.tv = nn::add(nn::mean(nn::square(dy), {1, 2, 3}), nn::mean(nn::square(dx), {1, 2, 3}));
  out.per_image = nn::sub(out.feature, nn::add(nn::mul(out.l2, double(l2_weight)),
                                               nn::mul(out.tv, double(tv_weight))));
  out.mean = nn::mean(out.per_image, {0});
  return out;
}

/// @brief Create explicit uniform28pximage starts with source-compatible raw mt19937draws.
///
/// Raw integer draws are ingress metadata;Doubletensor scaling and Float32
/// rounding perform all numeric preprocessing. This is SYNTHETICinitialnoise,
/// not sampled MNIST images or a learned data generator.
///
/// @param n Positive image count.
/// @param seed Source mt19937seed.
/// @return Float32[n,28,28,1]values Float32((raw+.5)/2^32).
/// @throws std::invalid_argument For nonpositive image count.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto starts = m::FeatureNoise(2, 1433);
/// @endcode
/// @par References
/// - Olah,Mordvintsev and Schubert(2017),Feature Visualization, https://distill.pub/2017/feature-visualization/,doi:10.23915/distill.00007. Repository adaptation maximizes frozen MNISTconv2channel means with explicit centeredL2 and squared-TVpriors,uniformnoise starts and bounded RMS-normalized TensorSGD/backtracking;not a pretrained GoogLeNet feature atlas.
/// @see VisualizeFeature(),DreamAscent()
inline Tensor FeatureNoise(int n, std::uint32_t seed) {
  if (n < 1)
    throw std::invalid_argument("positive number of noise images required");
  std::mt19937 rng(seed);
  std::vector<std::int64_t> raw(std::size_t(n) * 784);
  for (auto &value : raw)
    value = rng();
  return ((nn::from_vector(raw, {n, 28, 28, 1}).to(nn::DType::Float64) + .5) / 4294967296.)
      .to(nn::DType::Float32);
}

} // namespace datamunge::nn::models
