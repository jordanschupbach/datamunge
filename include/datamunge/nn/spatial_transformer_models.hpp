#pragma once
/// @file spatial_transformer_models.hpp
/// @brief Tensor affine sampling, learned localization, distorted MNIST and explicit CNN forwards.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <numbers>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Sample floating NHWC images on a differentiable aligned-corners affine grid.
///
/// Coordinates normalized to [-1,1] have outer PIXEL CENTRES at corners.
/// Theta maps each OUTPUT grid point (xt,yt,1) to INPUT (xs,ys).
/// A positive x translation samples farther right, moving visible content
/// left. Uses affine_grid/grid_sample with align_corners=true and zero
/// padding. Bilinear derivatives are piecewise; at exact integer positions
/// core sampling takes symmetric one-sided slopes, unlike a floor-only
/// reference. Identity values agree within coordinate roundoff. No fitting.
///
/// @param x Nonempty Float32/64 [N,H,W,C] images.
/// @param theta Matching dtype/batch [N,6] or [N,2,3] output-to-input transforms.
/// @param out_height Positive output row count.
/// @param out_width Positive output column count.
/// @return [N,out_height,out_width,C] bilinear zero-extended samples, differentiable in x/theta.
/// @throws std::invalid_argument For invalid images, transform shape/batch/dtype, output sizes or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto theta = n::tensor({1., 0., 0., 0., 1., 0.}, {1, 6});
/// const auto y = n::models::AffineSample(n::ones({1, 9, 9, 1}), theta, 9, 9);
/// @endcode
/// @par References
/// - Jaderberg et al. (2015), https://arxiv.org/abs/1506.02025
/// @see AffineSample(), SpatialTransformer(), StnMnistClassifier()
inline nn::Tensor AffineSample(const nn::Tensor &x, const nn::Tensor &theta,
                               std::int64_t out_height, std::int64_t out_width) {
  if (x.dim() != 4 || x.numel() == 0 || out_height < 1 || out_width < 1 ||
      (theta.dim() != 2 && theta.dim() != 3) || theta.size(0) != x.size(0) ||
      theta.dtype() != x.dtype())
    throw std::invalid_argument("invalid affine sample images/transform/size");
  nn::GridSampleOptions options;
  options.align_corners = true;
  return nn::grid_sample(x, nn::affine_grid(theta, {out_height, out_width}, true), options);
}

/// @brief Predicted affine transforms and resampled images from a localization network.
///
/// Tensor handles own storage references and retain full localization/sampling
/// graphs. Learned leaves remain owned by the parameter store; no fitting.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::SpatialTransformer(p, "st", n::ones({2, 28, 28, 1}));
/// @endcode
/// @par References
/// - Jaderberg et al. (2015), https://arxiv.org/abs/1506.02025
/// @see SpatialTransformer(), AffineSample()
struct SpatialTransformerOutputs {
  nn::Tensor theta;  ///< [N,6] row-major 2x3 output-to-input affine parameters.
  nn::Tensor output; ///< Resampled [N,H,W,C] images, zero outside input coordinates.
};

/// @brief Predict an identity-initialized affine map and differentiably warp input images.
///
/// Localization uses biased He-uniform 5x5/20/ReLU, SAME 2x2/2 max
/// pool, another 5x5/20/ReLU/pool, Glorot dense 50/ReLU, then dense 6.
/// Theta head weights start at zero and bias (1,0,0,0,1,0), so new
/// transforms are identity within sampling roundoff. The localization
/// filters initially have zero derivatives through the zero head weights;
/// head updates can activate them later. No bounds constrain learned theta,
/// so off-image sampling can become blank and lose useful gradients.
/// No BN/dropout/loss/optimizer/fitting hidden. Store access needs caller
/// synchronization; leaves must outlive retained backward graphs.
///
/// @param p Mutable floating learned store, lazy localization/filter/dense leaves.
/// @param name Localization prefix, normally stn.
/// @param x Nonempty floating NHWC images matching p dtype.
/// @return Predicted [N,6] theta and same-shape bilinear sampled images.
/// @throws std::invalid_argument For invalid images, transform shape/batch/dtype, output sizes or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::SpatialTransformer(p, "st", n::ones({2, 28, 28, 1}));
/// @endcode
/// @par References
/// - Jaderberg et al. (2015), https://arxiv.org/abs/1506.02025
/// @see AffineSample(), SpatialTransformer(), StnMnistClassifier()
inline SpatialTransformerOutputs
SpatialTransformer(nn::ParameterStore &p, const std::string &name, const nn::Tensor &x) {
  if (x.dim() != 4 || x.numel() == 0)
    throw std::invalid_argument("spatial transformer needs nonempty NHWC images");
  const std::int64_t height = x.shape()[1], width = x.shape()[2];
  nn::ConvLayerOptions cfg;
  cfg.init = nn::init::he_uniform();
  cfg.activation = nn::Activation::Relu;
  auto h =
      nn::max_pool2d(nn::conv2d(p, name + "/conv1", x, 20, 5, cfg), 2, 2, nn::Padding::Same);
  h = nn::max_pool2d(nn::conv2d(p, name + "/conv2", h, 20, 5, cfg), 2, 2, nn::Padding::Same);
  const std::int64_t flat = ((height + 3) / 4) * ((width + 3) / 4) * 20;
  auto f = nn::relu(nn::dense(p, name + "/fc1", nn::reshape(h, {-1, flat}), 50));
  p.get(name + "/theta/W", {50, 6}, nn::init::zeros());
  p.get(name + "/theta/b", {6}, [](const nn::Shape &shape, nn::DType dtype, nn::Generator &) {
    return nn::tensor({1, 0, 0, 0, 1, 0}, shape, dtype);
  });
  SpatialTransformerOutputs r;
  r.theta = nn::dense(p, name + "/theta", f, 6);
  r.output = AffineSample(x, r.theta, height, width);
  return r;
}
/// @brief Tensor-generated distorted MNIST images and their known output-to-input maps.
///
/// Detached data-generation tensors, no training graph. Caller keeps labels
/// unchanged and separates development/test generator seeds.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::Generator gen(101);
/// const auto data = n::models::DistortMnist(n::zeros({2, 28, 28, 1}), gen);
/// @endcode
/// @par References
/// - Jaderberg et al. (2015), https://arxiv.org/abs/1506.02025
/// @see DistortMnist()

struct DistortedMnistOutputs {
  nn::Tensor images; ///< [N,42,42,1] Float32/64 normalized uint8-like warped pixels.
  nn::Tensor theta;  ///< [N,6] known normalized output-to-input transforms.
};

/// @brief Generate rotated/scaled/translated 42x42 MNIST using tensor affine sampling.
///
/// Draw angles uniformly within +/-pi/4, scales .7..1.2 and independent
/// centre shifts in available room on a 42-pixel canvas. Extent is
/// 14*scale*(abs(cos)+abs(sin)); room=max(0,20.5-extent), matching
/// the source geometric ranges. Very large rotated squares can still crop.
/// Compute normalized inverse-map coefficients with tensors and AffineSample,
/// then floor(255*pixel+.5), clamp0..255 and divide255. All numerical
/// preprocessing stays in tensor operations under NoGradGuard; no host
/// pixel loops. The tensor RNG draw sequence differs from legacy interleaved
/// std::mt19937_64, so distributions/ranges match but images are not bitwise
/// identical to that source. Caller synchronizes shared RNG access. No fitting.
///
/// @param images Nonempty floating [N,28,28,1] normalized pixels in [0,1].
/// @param generator Borrowed mutable tensor RNG; advances for angle, scale and two shifts.
/// @return Detached warped/quantized images and their generated affine matrices.
/// @throws std::invalid_argument For invalid pixel shape/dtype/range or sampling arguments.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::Generator gen(101);
/// const auto data = n::models::DistortMnist(n::zeros({2, 28, 28, 1}), gen);
/// @endcode
/// @par References
/// - Jaderberg et al. (2015), https://arxiv.org/abs/1506.02025
/// @see AffineSample(), SpatialTransformer()

inline DistortedMnistOutputs DistortMnist(const nn::Tensor &images, nn::Generator &generator) {
  if (images.dim() != 4 || images.numel() == 0 || images.size(1) != 28 ||
      images.size(2) != 28 || images.size(3) != 1 ||
      (images.dtype() != nn::DType::Float32 && images.dtype() != nn::DType::Float64) ||
      !nn::all(nn::logical_and(nn::greater_equal(images, 0.), nn::less_equal(images, 1.)))
           .item<bool>())
    throw std::invalid_argument("DistortMnist needs normalized NHWC28x28 grayscale images");
  nn::NoGradGuard guard;
  const auto type = images.dtype();
  const auto batch = images.size(0);
  const auto angle =
      nn::uniform({batch}, -std::numbers::pi / 4, std::numbers::pi / 4, generator, type);
  const auto scale = nn::uniform({batch}, .7, 1.2, generator, type);
  const auto shift_y = nn::uniform({batch}, -1, 1, generator, type);
  const auto shift_x = nn::uniform({batch}, -1, 1, generator, type);
  const auto cosine = nn::cos(angle), sine = nn::sin(angle);
  const auto extent = 14 * scale * (nn::abs(cosine) + nn::abs(sine));
  const auto room = nn::maximum(20.5 - extent, nn::zeros({}, type));
  const auto cy = 20.5 + room * shift_y, cx = 20.5 + room * shift_x;
  const auto factor = 20.5 / (13.5 * scale);
  const auto tx = (cosine * (20.5 - cx) - sine * (20.5 - cy)) / (13.5 * scale),
             ty = (sine * (20.5 - cx) + cosine * (20.5 - cy)) / (13.5 * scale);
  DistortedMnistOutputs out;
  out.theta =
      nn::stack({cosine * factor, -sine * factor, tx, sine * factor, cosine * factor, ty}, 1);
  const auto sampled = AffineSample(images, out.theta, 42, 42);
  out.images = nn::minimum(nn::maximum(nn::floor(255 * sampled + .5), nn::zeros({}, type)),
                           nn::full({}, 255, type)) /
               255.;
  return out;
}

/// @brief Transformed image, optional predicted theta and MNIST classifier logits.
///
/// Tensor handles own storage references and retain sampling/classifier graphs.
/// No learned leaves are copied out of the parameter store.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::StnMnistClassifier(p, n::ones({1, 42, 42, 1}), true);
/// @endcode
/// @par References
/// - Jaderberg et al. (2015), https://arxiv.org/abs/1506.02025
/// @see StnMnistClassifier()

struct StnClassifierOutputs {
  nn::Tensor transformed; ///< [N,42,42,1] warped image, or input handle for baseline.
  nn::Tensor theta;       ///< [N,6] predicted map, undefined without transformer.
  nn::Tensor logits;      ///< [N,10] raw classifier scores.
};

/// @brief Build the source 42-pixel MNIST CNN with an optional learned spatial transformer.
///
/// Optional localization/affine sampling precedes biased He-uniform
/// 5x5/32/ReLU/SAME2x2 pool, 5x5/64/ReLU/pool, Glorot dense 128/ReLU
/// and dense 10. Two ceiling pools give11x11x64. Baseline learns1044746
/// scalars; the42-pixel localization adds 131896 for 1176642 total. No
/// BN, dropout, preprocessing, loss, optimizer or fitting is hidden.
/// Both training and inference run the same differentiable forward.
/// Initialization consumes the store RNG in architecture order, so baseline
/// and transformer classifier weights need not match despite equal seeds.
/// Parameter leaves must outlive retained graphs; synchronize shared access.
///
/// @param p Mutable floating learned store, lazy classifier and optional localization leaves.
/// @param image Nonempty floating [N,42,42,1] image tensor matching p dtype.
/// @param transformer True by default includes SpatialTransformer at prefix stn; false baseline.
/// @return Transformed images, optional theta and ten-class logits.
/// @throws std::invalid_argument For incompatible images/dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::StnMnistClassifier(p, n::ones({1, 42, 42, 1}), true);
/// @endcode
/// @par References
/// - Jaderberg et al. (2015), https://arxiv.org/abs/1506.02025
/// @see SpatialTransformer(), DistortMnist()

inline StnClassifierOutputs StnMnistClassifier(nn::ParameterStore &p, const nn::Tensor &image,
                                               bool transformer = true) {
  if (image.dim() != 4 || image.numel() == 0 || image.size(1) != 42 || image.size(2) != 42 ||
      image.size(3) != 1)
    throw std::invalid_argument("STN classifier needs NHWC42x42 grayscale input");
  StnClassifierOutputs out;
  out.transformed = image;
  if (transformer) {
    const auto warp = SpatialTransformer(p, "stn", image);
    out.theta = warp.theta;
    out.transformed = warp.output;
  }
  nn::ConvLayerOptions cfg;
  cfg.init = nn::init::he_uniform();
  cfg.padding = nn::Padding::Same;
  cfg.activation = nn::Activation::Relu;
  auto h = nn::max_pool2d(nn::conv2d(p, "conv1", out.transformed, 32, 5, cfg), 2, 2,
                          nn::Padding::Same);
  h = nn::max_pool2d(nn::conv2d(p, "conv2", h, 64, 5, cfg), 2, 2, nn::Padding::Same);
  const auto f = nn::relu(nn::dense(p, "fc1", h.reshape({-1, 11 * 11 * 64}), 128));
  out.logits = nn::dense(p, "output", f, 10);
  return out;
}

} // namespace datamunge::nn::models
