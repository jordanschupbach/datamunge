#pragma once
/// @file classic_cnn_components.hpp
/// @brief Differentiable LRN, grouped layers and Inception-v1 branches; no fitting.
#include "datamunge/nn/model_plot.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Normalise each last-axis channel by a clipped neighbourhood of squared channels.
///
/// The band is a constant tensor; both derivative orders flow through tensor matmul.
/// Edges clip the window. O(C²) constant storage and O(numel(x)*C) work.
/// Inputs and store state are not mutated; NaNs in x propagate.
///
/// @param x Floating Float32/64 [...,C], C positive; graph and dtype preserved.
/// @param radius Nonnegative number of neighbouring channels on each side.
/// @param bias Finite strictly positive additive denominator offset.
/// @param alpha Finite nonnegative multiplier of the summed squares; no division by window width.
/// @param beta Finite nonnegative denominator exponent.
/// @return Tensor of x.shape(), x / (bias + alpha * neighbourhood_sum(x*x))^beta.
/// @throws std::invalid_argument For invalid dtype, axis, radius or denominator settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::LocalResponseNorm(n::tensor({1., 2., 3.}), 1, 2., .1, .75);
/// @endcode
/// @par References
/// - Krizhevsky et al. (2012), https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
/// @see AlexNet()
inline nn::Tensor LocalResponseNorm(const nn::Tensor &x, int radius, double bias, double alpha,
                                    double beta) {
  if (x.dim() < 1 || x.shape().back() <= 0 || radius < 0 || !std::isfinite(bias) ||
      bias <= 0 || !std::isfinite(alpha) || alpha < 0 || !std::isfinite(beta) || beta < 0 ||
      (x.dtype() != nn::DType::Float32 && x.dtype() != nn::DType::Float64))
    throw std::invalid_argument("LRN needs a channel axis and nonnegative radius");
  const std::int64_t channels = x.shape().back();
  const auto indices = nn::arange(0, channels).to(x.dtype());
  const auto window =
      nn::less_equal(nn::abs(indices.reshape({channels, 1}) - indices.reshape({1, channels})),
                     double(radius))
          .to(x.dtype());
  const nn::Tensor sums =
      nn::reshape(nn::matmul(nn::reshape(nn::square(x), {-1, channels}), window), x.shape());
  const auto result = x * nn::pow(bias + alpha * sums, -beta);
  if (nn::model_plot_detail::recording())
    return nn::model_plot_detail::layer("LRN", "Local response normalization", x, result);
  return result;
}
/// @brief Apply separate contiguous channel partitions.
///
/// Each group has independent fan-in initialization. The two implementations share
/// parameter names and connectivity; gradients reach inputs and every active group.
/// No fitting or parameter mutation beyond lazy creation. Work O(NHoWo*k²*Cin*Cout/groups).
///
/// @param p Mutable parameter store; creates or reuses named learned leaves.
/// @param name Prefix; groups>1 use /group<g>/W and optional /b.
/// @param input Floating NHWC input, positive input channels, same dtype as p.
/// @param out Positive output channels, divisible by groups.
/// @param kernel Positive square kernel width.
/// @param groups Positive divisor of both input and output channels; one uses name/W.
/// @param options Shared stride, dilation, padding, activation, bias and initializer.
/// @return NHWC activation with out channels and normal convolution spatial geometry.
/// @throws std::invalid_argument For invalid groups, kernel, layer options, shapes or store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::GroupedConv2D(p, "g", n::ones({1, 4, 4, 4}), 6, 3, 2);
/// @endcode
/// @see FusedGroupedConv2D()
inline nn::Tensor GroupedConv2D(nn::ParameterStore &p, const std::string &name,
                                const nn::Tensor &input, std::int64_t out, std::int64_t kernel,
                                std::int64_t groups,
                                const nn::ConvLayerOptions &options = {}) {
  const std::int64_t in = input.dim() == 4 ? input.shape()[3] : 0;
  if (groups <= 0 || in <= 0 || out <= 0 || in % groups || out % groups)
    throw std::invalid_argument(
        "grouped convolution channels must be positive multiples of groups");
  if (groups == 1)
    return nn::conv2d(p, name, input, out, kernel, options);
  const auto partitions = nn::split(input, groups, 3);
  std::vector<nn::Tensor> outputs;
  for (std::int64_t group = 0; group < groups; ++group)
    outputs.push_back(nn::conv2d(p, name + "/group" + std::to_string(group),
                                 partitions[static_cast<std::size_t>(group)], out / groups,
                                 kernel, options));
  return nn::concat(outputs, 3);
}

/// @brief Apply one fused grouped core kernel with per-group parameter names.
///
/// Each group has independent fan-in initialization. The two implementations share
/// parameter names and connectivity; gradients reach inputs and every active group.
/// No fitting or parameter mutation beyond lazy creation. Work O(NHoWo*k²*Cin*Cout/groups).
///
/// @param p Mutable parameter store; creates or reuses named learned leaves.
/// @param name Prefix; groups>1 use /group<g>/W and optional /b.
/// @param input Floating NHWC input, positive input channels, same dtype as p.
/// @param out Positive output channels, divisible by groups.
/// @param kernel Positive square kernel width.
/// @param groups Positive divisor of both input and output channels; one uses name/W.
/// @param options Shared stride, dilation, padding, activation, bias and initializer.
/// @return NHWC activation with out channels and normal convolution spatial geometry.
/// @throws std::invalid_argument For invalid groups, kernel, layer options, shapes or store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::FusedGroupedConv2D(p, "g", n::ones({1, 4, 4, 4}), 6, 3, 2);
/// @endcode
/// @see GroupedConv2D()
inline nn::Tensor FusedGroupedConv2D(nn::ParameterStore &p, const std::string &name,
                                     const nn::Tensor &input, std::int64_t out,
                                     std::int64_t kernel, std::int64_t groups,
                                     const nn::ConvLayerOptions &options = {}) {
  const auto in = input.dim() == 4 ? input.size(3) : 0;
  if (groups <= 0 || in <= 0 || out <= 0 || in % groups || out % groups)
    throw std::invalid_argument(
        "grouped convolution channels must be positive multiples of groups");
  if (groups == 1)
    return nn::conv2d(p, name, input, out, kernel, options);
  std::vector<nn::Tensor> weights, biases;
  for (std::int64_t group = 0; group < groups; ++group) {
    const auto prefix = name + "/group" + std::to_string(group);
    weights.push_back(
        p.get(prefix + "/W", {kernel, kernel, in / groups, out / groups}, options.init));
    if (options.bias)
      biases.push_back(p.get(prefix + "/b", {out / groups}, nn::init::zeros()));
  }
  nn::Conv2DOptions op;
  op.groups = groups;
  op.strides = {options.stride, options.stride};
  op.dilations = {options.dilation, options.dilation};
  op.padding = options.padding;
  auto result = nn::conv2d(input, nn::concat(weights, 3), op);
  if (options.bias)
    result = nn::bias_add(result, nn::concat(biases, 0));
  return nn::activate(result, options.activation, options.alpha);
}
/// @brief Positive output and bottleneck widths for the four Inception-v1 branches.
///
/// Widths are integer architecture metadata; no learned state or fitting.
///
/// @par Example
/// @code
/// dm::nn::models::InceptionWidths w{64, 96, 128, 16, 32, 32};
/// const auto count = w.OutputChannels(); // 256
/// @endcode
/// @par References
/// - Szegedy et al. (2015), https://arxiv.org/abs/1409.4842
/// @see InceptionV1()
struct InceptionWidths {
  int one;          ///< Direct 1x1 branch output width.
  int reduce_three; ///< 1x1 bottleneck before 3x3.
  int three;        ///< 3x3 branch output width.
  int reduce_five;  ///< 1x1 bottleneck before 5x5.
  int five;         ///< 5x5 branch output width.
  int pool;         ///< Post-pooling 1x1 output width.

  /// @brief Sum the four concatenated branch output widths.
  ///
  /// @return one+three+five+pool; call Divided(1) to validate widths.
  /// @par Example
  /// @code
  /// dm::nn::models::InceptionWidths w{2, 1, 3, 1, 4, 5};
  /// const auto channels = w.OutputChannels(); // 14
  /// @endcode
  /// @see Divided()
  int OutputChannels() const {
    return one + three + five + pool;
  }
  /// @brief Return a validated exact width-scaled copy.
  ///
  /// Does not mutate this object.
  ///
  /// @param divisor Positive integer dividing all six positive widths.
  /// @return Independent width metadata with every component divided by divisor.
  /// @throws std::invalid_argument For nonpositive widths/divisor or inexact division.
  /// @par Example
  /// @code
  /// const auto w = dm::nn::models::InceptionWidths{64, 96, 128, 16, 32, 32}.Divided(8);
  /// @endcode
  /// @see OutputChannels()
  InceptionWidths Divided(int divisor) const {
    if (divisor <= 0)
      throw std::invalid_argument("positive Inception width divisor required");
    for (int width : {one, reduce_three, three, reduce_five, five, pool})
      if (width <= 0 || width % divisor)
        throw std::invalid_argument("Inception widths must divide exactly");
    return {one / divisor,         reduce_three / divisor, three / divisor,
            reduce_five / divisor, five / divisor,         pool / divisor};
  }
  /// @brief Count kernel weights or convolution MACs per spatial position.
  ///
  /// Uses input/reduced widths rather than applying expensive kernels at input width.
  ///
  /// @param in Positive number of input channels.
  /// @return 64-bit kernel scalar count, excluding biases and pooling comparisons.
  /// @throws std::invalid_argument For invalid widths or nonpositive input width.
  /// @par Example
  /// @code
  /// const auto count =
  ///     dm::nn::models::InceptionWidths{64, 96, 128, 16, 32, 32}.KernelWeights(192); // 163328
  /// @endcode
  /// @see InceptionV1()
  std::int64_t KernelWeights(int in) const {
    Divided(1);
    if (in <= 0)
      throw std::invalid_argument("positive Inception input width required");
    return std::int64_t(in) * (one + reduce_three + reduce_five + pool) +
           9LL * reduce_three * three + 25LL * reduce_five * five;
  }
};

/// @brief Four branch activations and their channel concatenation.
///
/// Tensors own their storage references and retain differentiation graphs.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto out = n::models::InceptionV1(p, "i", n::ones({1, 4, 4, 2}), {2, 1, 2, 1, 2, 2});
/// @endcode
/// @see InceptionV1()
struct InceptionOutputs {
  std::array<nn::Tensor, 4>
      branches;            ///< Order: 1x1, reduced 3x3, reduced 5x5, pool projection.
  nn::Tensor concatenated; ///< NHWC, sum of the four output widths.
};

/// @brief Apply the four SAME-padded Inception-v1 branches and concatenate channels.
///
/// All convolutions use He initialization, biases and ReLU. Pooling is 3x3 stride
/// one SAME max-pooling before projection. No fitting; graph retained. Work
/// O(NHW*widths.KernelWeights(Cin)), excluding pooling comparisons.
///
/// @param p Mutable floating parameter store, lazily creates or reuses leaves.
/// @param name Layer prefix with /one, /three_reduce, /three, /five_reduce, /five, /pool suffixes.
/// @param input Floating NHWC tensor, with positive channel width, dtype matching p.
/// @param widths Positive validated branch and bottleneck widths.
/// @return Four ReLU branch maps and their NHWC concatenation, preserving spatial shape.
/// @throws std::invalid_argument For invalid widths, input, convolution options or stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::InceptionV1(p, "i", n::ones({1, 4, 4, 2}), {2, 1, 2, 1, 2, 2});
/// @endcode
/// @par References
/// - Szegedy et al. (2015), https://arxiv.org/abs/1409.4842
/// @see InceptionWidths, GoogLeNet()
inline InceptionOutputs InceptionV1(nn::ParameterStore &p, const std::string &name,
                                    const nn::Tensor &input, const InceptionWidths &widths) {
  const auto w = widths.Divided(1);
  if (input.dim() != 4 || input.shape()[3] <= 0)
    throw std::invalid_argument("positive Inception input width required");
  nn::ConvLayerOptions options;
  options.activation = nn::Activation::Relu;
  options.init = nn::init::he_uniform();
  InceptionOutputs out;
  out.branches[0] = nn::conv2d(p, name + "/one", input, w.one, 1, options);
  const auto three = nn::conv2d(p, name + "/three_reduce", input, w.reduce_three, 1, options);
  out.branches[1] = nn::conv2d(p, name + "/three", three, w.three, 3, options);
  const auto five = nn::conv2d(p, name + "/five_reduce", input, w.reduce_five, 1, options);
  out.branches[2] = nn::conv2d(p, name + "/five", five, w.five, 5, options);
  out.branches[3] = nn::conv2d(
      p, name + "/pool", nn::max_pool2d(input, 3, 1, nn::Padding::Same), w.pool, 1, options);
  out.concatenated =
      nn::concat(std::vector<nn::Tensor>(out.branches.begin(), out.branches.end()), 3);
  return out;
}
} // namespace datamunge::nn::models
