// Optical-flow pixel coordinates, differentiable backward warp and local
// correlation: a port of tflow's common/nn/flow_ops.h (flownets, pwcnet).
// TF -> datamunge: tflow's warp gathers the four bilinear taps explicitly
// (DeformableSamples with a 1x1 kernel); here nn::sample_pixels (zeros
// outside) does the same sampling. The only difference is the position
// derivative exactly at integer coordinates, where nn uses the mean of the two
// one-sided slopes and tflow the right-hand slope.
#pragma once

#include <array>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flow {

namespace nn = datamunge::nn;

struct FlowResult {
  nn::Tensor                flow;
  std::array<nn::Tensor, 5> pyramid;  // levels 2..6, input pixels / 20
  std::vector<nn::Tensor>   stages;
};

struct FlowWarpResult {
  nn::Tensor value, valid;
};

/// Pixel-centre grid + flow as sample_pixels coordinates (x, y): [N, H, W, 2].
inline nn::Tensor FlowCoordinates(const nn::Tensor& flow) {
  const std::int64_t H = flow.size(1), W = flow.size(2);
  std::vector<float> grid(std::size_t(H * W * 2));
  for (std::int64_t y = 0; y < H; ++y)
    for (std::int64_t x = 0; x < W; ++x) grid[std::size_t((y * W + x) * 2)] = float(x), grid[std::size_t((y * W + x) * 2 + 1)] = float(y);
  return flow + nn::from_vector(grid, {1, H, W, 2}).to(flow.dtype());
}

/// f(x,y)=(dx,dy) points from image1 INTO image2; sample image2(x+dx,y+dy)
/// bilinearly with zeros outside. `valid` is 1 where the whole bilinear window
/// is inside (coverage >= .9999); complete_window zeroes the rest.
inline FlowWarpResult FlowWarp(const nn::Tensor& second, const nn::Tensor& flow, bool complete_window = true) {
  const nn::Tensor coords   = FlowCoordinates(flow);
  const nn::Tensor value    = nn::sample_pixels(second, coords);
  nn::Tensor       coverage;
  {
    nn::NoGradGuard guard;
    coverage = nn::sample_pixels(nn::ones({second.size(0), second.size(1), second.size(2), 1}, second.dtype()), coords.detach());
  }
  const nn::Tensor valid = nn::cast(nn::greater_equal(coverage, .9999), second.dtype());
  return {complete_window ? value * valid : value, valid};
}

/// Cost volume: dy-major/dx-minor ordering, channel-mean dot product, zero outside image2.
inline nn::Tensor FlowCorrelation(const nn::Tensor& first, const nn::Tensor& second, int radius) {
  if (radius < 0) throw std::invalid_argument("nonnegative flow correlation radius");
  const nn::Tensor        padded = nn::pad(second, {{0, 0}, {radius, radius}, {radius, radius}, {0, 0}});
  std::vector<nn::Tensor> costs;
  for (int dy = -radius; dy <= radius; ++dy)
    for (int dx = -radius; dx <= radius; ++dx) {
      const nn::Tensor shifted = nn::slice(padded, {0, radius + dy, radius + dx, 0}, second.shape());
      costs.push_back(nn::mean(first * shifted, {3}));
    }
  return nn::stack(costs, 3);
}

/// Half-pixel-centre bilinear resize of a field whose values keep their units.
inline nn::Tensor FlowResizeValues(const nn::Tensor& field, std::int64_t height, std::int64_t width) {
  return nn::resize_bilinear(field, {height, width}, false, true);
}

/// Use ONLY when vector units belong to the source grid. Internal flow/20
/// fields keep INPUT-image units and therefore use FlowResizeValues.
inline nn::Tensor FlowResizeVectors(const nn::Tensor& field, int from_h, int from_w, int to_h, int to_w) {
  return FlowResizeValues(field, to_h, to_w) * nn::tensor({double(float(to_w) / from_w), double(float(to_h) / from_h)}).to(field.dtype());
}

/// Input-pixels/20 field at pyramid `level` -> that level's own pixels.
inline nn::Tensor FlowPyramidPixels(const nn::Tensor& input_units_over20, int level) {
  return input_units_over20 * double(20.f / float(1 << level));
}

/// Stride-2 SAME transposed convolution to [N, 2H, 2W, out]; weights [4, 4, out, in].
inline nn::Tensor FlowTranspose(const nn::Tensor& x, const nn::Tensor& weights, std::int64_t out) {
  return nn::conv2d_transpose(x, weights, {x.size(0), x.size(1) * 2, x.size(2) * 2, out}, nn::conv_options(2, nn::Padding::Same));
}

/// Explicit odd-kernel symmetric padding preserves Torch stride phase; LeakyReLU .1.
inline nn::Tensor FlowConv(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out, int kernel = 3,
                           int stride = 1, bool activate = true, const nn::Initializer& init = nn::init::glorot_uniform()) {
  const std::int64_t pad     = (kernel - 1) / 2;
  const nn::Tensor   weights = p.get(name + "/W", {kernel, kernel, x.size(3), out}, init);
  nn::Conv2DOptions  o;
  o.strides           = {stride, stride};
  o.padding           = nn::Padding::Explicit;
  o.explicit_paddings = {pad, pad, pad, pad};
  const nn::Tensor y  = nn::bias_add(nn::conv2d(x, weights, o), p.get(name + "/b", {out}, nn::init::zeros()));
  return activate ? nn::leaky_relu(y, .1) : y;
}

/// Learned 4x4 stride-2 upsampling (optional bias, LeakyReLU .1).
inline nn::Tensor FlowUp(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out, bool activate = true,
                         bool bias = true, const nn::Initializer& init = nn::init::glorot_uniform()) {
  const nn::Tensor weights = p.get(name + "/W", {4, 4, out, x.size(3)}, init);
  nn::Tensor       y       = FlowTranspose(x, weights, out);
  if (bias) y = nn::bias_add(y, p.get(name + "/b", {out}, nn::init::zeros()));
  return activate ? nn::leaky_relu(y, .1) : y;
}

/// Subtracts the joint per-image, per-channel mean of both frames.
inline std::pair<nn::Tensor, nn::Tensor> FlowCenterPair(const nn::Tensor& first, const nn::Tensor& second) {
  const nn::Tensor mean = .5 * (nn::mean(first, {1, 2}, true) + nn::mean(second, {1, 2}, true));
  return {first - mean, second - mean};
}

}  // namespace flow
