// Spatial transformer networks (port of tflow's common/nn/stn.h): a
// localization network predicts an affine transform, a grid generator maps
// output pixels to input coordinates, and a bilinear sampler reads the input
// there, all differentiably.
// https://arxiv.org/abs/1506.02025
//
// Coordinates are normalized to [-1, 1] with corners at the outer pixel
// centres (the paper's convention, "align corners"): output pixel (i, j)
// maps to (x_s, y_s) = theta * (x_t, y_t, 1) and samples the input
// bilinearly, reading zero outside it. tflow builds the sampler from gathers;
// here it is the library's nn::affine_grid + nn::grid_sample (align_corners,
// zero padding), which compute the same values and gradients except at exact
// integer pixel coordinates: there the library takes the mean of the two
// one-sided slopes, where tflow's floor-based gathers take whichever side
// float rounding picks. Every point of the identity transform the
// transformer starts from is such a point, and the symmetric slope makes
// early training noticeably more stable (see the example). The
// localization head is a dense layer whose weights start at zero and bias at
// the identity transform (1,0,0,0,1,0), so a new transformer passes its input
// through unchanged.
#pragma once

#include <datamunge/nn/image_ops.hpp>
#include <datamunge/nn/nn.hpp>
#include <string>

namespace stn {

namespace nn = datamunge::nn;

/// Samples x [N, H, W, C] on the affine grid of theta [N, 6] at out_height x out_width.
inline nn::Tensor AffineSample(const nn::Tensor& x, const nn::Tensor& theta, std::int64_t out_height,
                               std::int64_t out_width) {
  nn::GridSampleOptions options;
  options.align_corners = true;
  return nn::grid_sample(x, nn::affine_grid(theta, {out_height, out_width}, true), options);
}

struct SpatialTransformerOutputs {
  nn::Tensor theta, output;
};

/// Transformer on x [N, H, W, C] with a small convolutional localization network
/// (5x5 conv 20, pool, 5x5 conv 20, pool, dense 50, dense 6).
inline SpatialTransformerOutputs SpatialTransformer(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x) {
  const std::int64_t   height = x.shape()[1], width = x.shape()[2];
  nn::ConvLayerOptions cfg;
  cfg.init       = nn::init::he_uniform();
  cfg.activation = nn::Activation::Relu;
  auto h = nn::max_pool2d(nn::conv2d(p, name + "/conv1", x, 20, 5, cfg), 2, 2, nn::Padding::Same);
  h      = nn::max_pool2d(nn::conv2d(p, name + "/conv2", h, 20, 5, cfg), 2, 2, nn::Padding::Same);
  const std::int64_t flat = ((height + 3) / 4) * ((width + 3) / 4) * 20;
  auto f = nn::relu(nn::dense(p, name + "/fc1", nn::reshape(h, {-1, flat}), 50));
  p.get(name + "/theta/W", {50, 6}, nn::init::zeros());
  p.get(name + "/theta/b", {6}, [](const nn::Shape& shape, nn::DType dtype, nn::Generator&) {
    return nn::tensor({1, 0, 0, 0, 1, 0}, shape, dtype);
  });
  SpatialTransformerOutputs r;
  r.theta  = nn::dense(p, name + "/theta", f, 6);
  r.output = AffineSample(x, r.theta, height, width);
  return r;
}

}  // namespace stn
