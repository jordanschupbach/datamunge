#pragma once
/// @file flow_components.hpp
/// @brief Tensor flow coordinates,backward warping,correlation,unit conversion and learned convolution primitives.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <array>
#include <limits>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own full-resolution flow,pyramid fields and internal activations.
///
/// Tensor handles preserve the graph. All internal levels share the SAME input-pixel/20
/// units;spatial resizing does not rescale their displacement values.
///
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(163);
/// const auto r = m::FlowNetS(p, n::zeros({1, 64, 64, 3}), n::zeros({1, 64, 64, 3}), 64, 64, 8);
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
struct FlowResult {
  nn::Tensor flow; ///< Full-resolution[N,H,W,2] flow in input-image pixel units.
  std::array<nn::Tensor, 5>
      pyramid; ///< Levels2..6,in input-image pixels/20 at every resolution.
  std::vector<nn::Tensor>
      stages; ///< Internal activations,learned upsampling and skip concatenations.
};

/// @brief Own backward-warped image and detached full-window coverage.
///
/// Value gradients reach source pixels and flow;coverage decisions are fixed.
///
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto r = m::FlowWarp(n::ones({1, 3, 4, 2}), n::zeros({1, 3, 4, 2}));
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
struct FlowWarpResult {
  nn::Tensor value; ///< Sampled NHWC second image,optionally masked at incomplete windows.
  nn::Tensor valid; ///< Detached[N,H,W,1] coverage>=.9999 mask.
};

/// @brief Add signed dx/dy to a tensor pixel-center grid.
///
/// Grid arange first uses Float32 like the source,then converts to flow dtype.
/// No host pixel arithmetic,learned parameters or optimizer.
///
/// @param flow Floating[N,H,W,2],positive geometry,dx then dy.
/// @return Same-shape absolute source sampling coordinates(x+dx,y+dy).
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto grid = m::FlowCoordinates(n::zeros({2, 3, 4, 2}));
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline nn::Tensor FlowCoordinates(const nn::Tensor &flow) {
  if (flow.dim() != 4 || flow.size(0) < 1 || flow.size(1) < 1 || flow.size(2) < 1 ||
      flow.size(3) != 2 ||
      (flow.dtype() != nn::DType::Float32 && flow.dtype() != nn::DType::Float64))
    throw std::invalid_argument("positive floating NHW2 flow required");
  const auto h = flow.size(1), w = flow.size(2);
  const auto x =
      nn::arange(0, double(w), 1, nn::DType::Float32).to(flow.dtype()).reshape({1, 1, w, 1}) +
      nn::zeros({1, h, 1, 1}, flow.dtype());
  const auto y =
      nn::arange(0, double(h), 1, nn::DType::Float32).to(flow.dtype()).reshape({1, h, 1, 1}) +
      nn::zeros({1, 1, w, 1}, flow.dtype());
  return flow + nn::concat({x, y}, 3);
}

/// @brief Sample the second frame at first-frame coordinates plus flow.
///
/// Coordinates point FROM image1 INTO image2. Outside taps contribute zero.
/// At integer coordinates core sample_pixels uses the mean of one-sided
/// position slopes,the documented adaptation from the old right-slope sampler.
/// Coverage is detached;image/flow gradients remain active.
///
/// @param second Floating[N,H,W,C] source image,positive geometry.
/// @param flow Matching[N,H,W,2] input-grid dx/dy pixels,same dtype.
/// @param complete_window Zero windows with coverage<.9999,default true.
/// @return Bilinear sampled image and detached one-channel coverage mask.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto warped = m::FlowWarp(n::ones({1, 3, 4, 3}), n::full({1, 3, 4, 2}, .25));
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline FlowWarpResult FlowWarp(const nn::Tensor &second, const nn::Tensor &flow,
                               bool complete_window = true) {
  if (second.dim() != 4 || flow.dim() != 4 || second.size(0) != flow.size(0) ||
      second.size(1) != flow.size(1) || second.size(2) != flow.size(2) || second.size(3) < 1 ||
      second.dtype() != flow.dtype())
    throw std::invalid_argument("aligned floating source image and flow required");
  const nn::Tensor coords = FlowCoordinates(flow);
  const nn::Tensor value = nn::sample_pixels(second, coords);
  nn::Tensor coverage;
  {
    nn::NoGradGuard guard;
    coverage = nn::sample_pixels(
        nn::ones({second.size(0), second.size(1), second.size(2), 1}, second.dtype()),
        coords.detach());
  }
  const nn::Tensor valid = nn::cast(nn::greater_equal(coverage, .9999), second.dtype());
  return {complete_window ? value * valid : value, valid};
}

/// @brief Compute a local normalized feature correlation volume.
///
/// Channels are dy-major/dx-minor from-r..r;outside second-image features
/// are zero. Both feature gradients remain active. Work O(NHW*C*(2r+1)^2).
/// Host loops enumerate displacement metadata only;no learned state or fitting.
///
/// @param first Floating positive NHWC reference features.
/// @param second Same-shape/dtype matching features.
/// @param radius Nonnegative integer displacement radius.
/// @return [N,H,W,(2r+1)^2] mean-channel dot products.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto cost = m::FlowCorrelation(n::ones({2, 3, 4, 2}), n::ones({2, 3, 4, 2}), 1);
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline nn::Tensor FlowCorrelation(const nn::Tensor &first, const nn::Tensor &second,
                                  int radius) {
  if (radius < 0 || radius > (std::numeric_limits<int>::max() - 1) / 2 || first.dim() != 4 ||
      first.size(0) < 1 || first.size(1) < 1 || first.size(2) < 1 || first.size(3) < 1 ||
      first.shape() != second.shape() || first.dtype() != second.dtype())
    throw std::invalid_argument("nonnegative flow correlation radius");
  const nn::Tensor padded =
      nn::pad(second, {{0, 0}, {radius, radius}, {radius, radius}, {0, 0}});
  std::vector<nn::Tensor> costs;
  for (int dy = -radius; dy <= radius; ++dy)
    for (int dx = -radius; dx <= radius; ++dx) {
      const nn::Tensor shifted =
          nn::slice(padded, {0, radius + dy, radius + dx, 0}, second.shape());
      costs.push_back(nn::mean(first * shifted, {3}));
    }
  return nn::stack(costs, 3);
}

/// @brief Resize an NHWC field without changing its numerical value units.
///
/// align_corners=false,half_pixel_centers=true. Use for internal input-pixel/20
/// flow fields. No component scale multiplication or optimizer.
///
/// @param field Floating NHWC source field.
/// @param height Positive destination height.
/// @param width Positive destination width.
/// @return Half-pixel bilinear[N,height,width,C] field,graph retained.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto f = m::FlowResizeValues(n::ones({1, 2, 3, 2}), 4, 6);
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline nn::Tensor FlowResizeValues(const nn::Tensor &field, std::int64_t height,
                                   std::int64_t width) {
  return nn::resize_bilinear(field, {height, width}, false, true);
}

/// @brief Resize flow whose components are measured in the SOURCE grid pixels.
///
/// Ratios first use Float32 to preserve the source convention,then field dtype.
/// Do not use for fixed input-image-pixel units;use FlowResizeValues there.
///
/// @param field Floating[N,from_h,from_w,2] dx/dy field.
/// @param from_h Positive source height matching field.
/// @param from_w Positive source width matching field.
/// @param to_h Positive destination height.
/// @param to_w Positive destination width.
/// @return Resized flow scaled componentwise by(to_w/from_w,to_h/from_h).
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto f = m::FlowResizeVectors(n::ones({1, 2, 3, 2}), 2, 3, 4, 9);
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline nn::Tensor FlowResizeVectors(const nn::Tensor &field, int from_h, int from_w, int to_h,
                                    int to_w) {
  if (from_h < 1 || from_w < 1 || to_h < 1 || to_w < 1 || field.dim() != 4 ||
      field.size(1) != from_h || field.size(2) != from_w || field.size(3) != 2)
    throw std::invalid_argument("positive aligned vector-resize geometry required");
  const auto scale =
      (nn::from_vector(std::vector<std::int32_t>{to_w, to_h}, {2}).to(nn::DType::Float32) /
       nn::from_vector(std::vector<std::int32_t>{from_w, from_h}, {2}).to(nn::DType::Float32))
          .to(field.dtype());
  return FlowResizeValues(field, to_h, to_w) * scale;
}

/// @brief Convert fixed input-pixel/20 flow to one pyramid grid's pixel units.
///
/// The ratio is computed in Float32 like the source,then field dtype.
/// No resize,learned parameters or optimizer.
///
/// @param input_units_over20 Floating internal field with dx/dy components.
/// @param level Integer pyramid exponent0..30.
/// @return Field multiplied by20/(2^level),graph retained.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto f = m::FlowPyramidPixels(n::ones({1, 3, 4, 2}), 2);
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline nn::Tensor FlowPyramidPixels(const nn::Tensor &input_units_over20, int level) {
  if (level < 0 || level > 30)
    throw std::invalid_argument("flow pyramid level must be0..30");
  const auto scale = (nn::full({}, 20., nn::DType::Float32) /
                      nn::full({}, std::int64_t(1) << level, nn::DType::Float32))
                         .to(input_units_over20.dtype());
  return input_units_over20 * scale;
}

/// @brief Apply a stride2 SAME learned transpose kernel.
///
/// No bias,activation or parameter creation. All input/kernel derivatives
/// flow through the canonical transpose convolution.
///
/// @param x Floating positive NHWC activation.
/// @param weights Floating[4,4,out,Cin] transpose weights.
/// @param out Positive output width matching weights.
/// @return [N,2H,2W,out] raw transpose output,graph retained.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = m::FlowTranspose(n::ones({1, 2, 3, 2}), n::ones({4, 4, 3, 2}), 3);
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline nn::Tensor FlowTranspose(const nn::Tensor &x, const nn::Tensor &weights,
                                std::int64_t out) {
  return nn::conv2d_transpose(x, weights, {x.size(0), x.size(1) * 2, x.size(2) * 2, out},
                              nn::conv_options(2, nn::Padding::Same));
}

/// @brief Apply explicitly symmetric-padded odd flow convolution.
///
/// Explicit(kernel-1)/2 padding on BOTH sides preserves source stride phase.
/// Lazy learned creation only;warm all leaves before caller optimizer.
///
/// @param p Mutable learned parameter store.
/// @param name Prefix for W and b leaves.
/// @param x Floating positive NHWC input matching p dtype.
/// @param out Positive output-channel count.
/// @param kernel Positive odd square kernel,default3.
/// @param stride Positive spatial stride,default1.
/// @param activate Apply LeakyReLU.1 when true,default true.
/// @param init Kernel initializer,default Glorot uniform.
/// @return Biased convolution with optional LeakyReLU.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(163);
/// const auto x = m::FlowConv(p, "c", n::ones({1, 5, 7, 3}), 8, 3, 2);
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline nn::Tensor FlowConv(nn::ParameterStore &p, const std::string &name, const nn::Tensor &x,
                           std::int64_t out, int kernel = 3, int stride = 1,
                           bool activate = true,
                           const nn::Initializer &init = nn::init::glorot_uniform()) {
  if (x.dim() != 4 || x.size(0) < 1 || x.size(1) < 1 || x.size(2) < 1 || x.size(3) < 1 ||
      out < 1 || kernel < 1 || kernel % 2 != 1 || stride < 1)
    throw std::invalid_argument("positive odd-kernel flow convolution required");
  const std::int64_t pad = (kernel - 1) / 2;
  const nn::Tensor weights = p.get(name + "/W", {kernel, kernel, x.size(3), out}, init);
  nn::Conv2DOptions o;
  o.strides = {stride, stride};
  o.padding = nn::Padding::Explicit;
  o.explicit_paddings = {pad, pad, pad, pad};
  const nn::Tensor y =
      nn::bias_add(nn::conv2d(x, weights, o), p.get(name + "/b", {out}, nn::init::zeros()));
  return activate ? nn::leaky_relu(y, .1) : y;
}

/// @brief Apply a learned4x4 stride2 upsampling convolution.
///
/// Internal flow upsampling uses activate=false,bias=false. Lazy creation only;
/// no hidden optimizer or output clamping.
///
/// @param p Mutable learned store.
/// @param name Prefix for W and optional b.
/// @param x Floating positive NHWC activations.
/// @param out Positive output-channel count.
/// @param activate Apply LeakyReLU.1,default true.
/// @param bias Include learned bias,default true.
/// @param init Kernel initializer,default Glorot uniform.
/// @return [N,2H,2W,out] output with optional bias/activation.
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(163);
/// const auto x = m::FlowUp(p, "up", n::ones({1, 2, 3, 4}), 8);
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline nn::Tensor FlowUp(nn::ParameterStore &p, const std::string &name, const nn::Tensor &x,
                         std::int64_t out, bool activate = true, bool bias = true,
                         const nn::Initializer &init = nn::init::glorot_uniform()) {
  const nn::Tensor weights = p.get(name + "/W", {4, 4, out, x.size(3)}, init);
  nn::Tensor y = FlowTranspose(x, weights, out);
  if (bias)
    y = nn::bias_add(y, p.get(name + "/b", {out}, nn::init::zeros()));
  return activate ? nn::leaky_relu(y, .1) : y;
}

/// @brief Jointly center two frames per image and color channel.
///
/// Batch rows and channels never mix. The mean is part of the tensor graph,
/// not a fitted dataset statistic;no optimizer or persistent state.
///
/// @param first Floating positive NHWC first frame.
/// @param second Matching second frame,same dtype.
/// @return Both frames minus.5*(spatial_mean(first)+spatial_mean(second)).
/// @throws std::invalid_argument For invalid geometry,dtype,configuration or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto centered = m::FlowCenterPair(n::zeros({2, 3, 4, 3}), n::ones({2, 3, 4, 3}));
/// @endcode
/// @par References
/// - Fischer et al.(2015),FlowNet:Learning Optical Flow with Convolutional Networks, https://arxiv.org/abs/1504.06852;FlowNetS graph https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py. Repository teaching flow units are input pixels/20;backward warping uses core sample_pixels.
/// @see FlowNetS(), FlowWarp(), FlowObjective()
inline std::pair<nn::Tensor, nn::Tensor> FlowCenterPair(const nn::Tensor &first,
                                                        const nn::Tensor &second) {
  if (first.dim() != 4 || first.shape() != second.shape() || first.numel() < 1 ||
      first.dtype() != second.dtype())
    throw std::invalid_argument("aligned positive NHWC flow frame pair required");
  const nn::Tensor mean =
      .5 * (nn::mean(first, {1, 2}, true) + nn::mean(second, {1, 2}, true));
  return {first - mean, second - mean};
}

} // namespace datamunge::nn::models
