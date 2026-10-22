// Shared numerical checks and compact CIFAR-10 builder for the ResNet examples
// (port of tflow's common/ml/resnet_demo.h). Canonical gradient checks use a
// two-image training-mode batch, so deep untrained networks are normalized by
// their current activation statistics. Running-statistic updates are
// disabled during those derivative checks.
#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/resnet.hpp"

namespace resnet_demo {

namespace nn = datamunge::nn;

/// Zero-residual blocks: identity/projection shortcuts, post-add ReLU
/// clipping of negative inputs, and the input derivative through the shortcut.
inline void ResNetBlockChecks(bool bottleneck) {
  for (bool projection : {false, true}) {
    nn::ParameterStore     p(17), state(17);
    resnet::ConvNormConfig cfg;
    cfg.scale   = true;
    cfg.epsilon = 1e-5;  // inference mode: moving mean 0, variance 1
    const std::int64_t in = projection ? 2 : bottleneck ? 16 : 4, width = projection && bottleneck ? 1 : 4;
    const std::int64_t spatial = projection ? 5 : 3, output_channels = bottleneck ? 4 * width : width;
    const int          stride  = projection ? 2 : 1;
    std::vector<float> values(static_cast<std::size_t>(spatial * spatial * in));
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = (i % 2 ? -1.f : 1.f) * (.2f + .01f * i);
    nn::Tensor x = nn::from_vector(values, {1, spatial, spatial, in});
    {
      nn::NoGradGuard guard;  // create the parameters, then zero the residual and control the shortcut
      resnet::ResNetBlock(p, state, "fixture", x, width, cfg, stride, bottleneck);
      const std::string last = bottleneck ? "fixture/conv3/bn/gamma" : "fixture/conv2/bn/gamma";
      p.set(last, nn::zeros({output_channels}));
      if (projection) {
        std::vector<float> kernel(static_cast<std::size_t>(in * output_channels), 0.f);
        for (std::int64_t c = 0; c < output_channels; ++c) kernel[static_cast<std::size_t>((c % in) * output_channels + c)] = 1;
        p.set("fixture/shortcut/W", nn::from_vector(kernel, {1, 1, in, output_channels}));
        p.set("fixture/shortcut/bn/gamma", nn::full({output_channels}, std::sqrt(1 + cfg.epsilon)));
      }
    }
    x.requires_grad_(true);
    const auto block    = resnet::ResNetBlock(p, state, "fixture", x, width, cfg, stride, bottleneck);
    const auto gradient = nn::grad(nn::sum(block.output), {x})[0].to_vector<float>();
    const std::int64_t out_spatial = projection ? 3 : spatial;
    check::Require(block.output.shape() == nn::Shape({1, out_spatial, out_spatial, output_channels}),
                   "ResNet shortcut shape");
    const auto output = block.output.to_vector<float>(), residual = block.residual.to_vector<float>();
    for (std::int64_t r = 0; r < out_spatial; ++r)
      for (std::int64_t c = 0; c < out_spatial; ++c)
        for (std::int64_t channel = 0; channel < output_channels; ++channel) {
          const auto i = static_cast<std::size_t>((r * out_spatial + c) * output_channels + channel);
          const auto source = values[static_cast<std::size_t>(((stride * r) * spatial + stride * c) * in + channel % in)];
          check::Near(residual[i], 0, 0, "zero final BN scale zeros residual branch");
          check::Near(output[i], std::max(0.f, source), 2e-6, "zero-residual block is ReLU(shortcut), including negative inputs");
        }
    for (std::int64_t r = 0; r < spatial; ++r)
      for (std::int64_t c = 0; c < spatial; ++c)
        for (std::int64_t channel = 0; channel < in; ++channel) {
          const auto   i          = static_cast<std::size_t>((r * spatial + c) * in + channel);
          const double derivative = r % stride == 0 && c % stride == 0 && values[i] > 0 ? double(output_channels) / in : 0;
          check::Near(gradient[i], derivative, 2e-6, "shortcut subsampling and post-add ReLU derivative");
        }
  }
  std::cout << "  " << (bottleneck ? "bottleneck" : "basic")
            << " blocks: zero-residual identity/projection, negative-input clipping and shortcut derivatives passed\n";
}

/// Canonical 224-pixel ResNet: parameter/state counts, stage dimensions and
/// finite nonzero training-mode derivatives for every parameter.
inline void FullResNetChecks(int depth, std::int64_t parameter_count, std::int64_t state_count) {
  nn::ParameterStore   p(17), state(17);
  resnet::ResNetConfig cfg;
  cfg.depth        = depth;
  cfg.update_stats = false;
  std::vector<float> pixels(2 * 224 * 224 * 3);
  for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = .1f * std::sin(static_cast<std::int64_t>(i) * .13f);
  const auto net = resnet::ResNet(p, state, nn::from_vector(pixels, {2, 224, 224, 3}), cfg, true);
  check::Require(p.count() == parameter_count && state.count() == state_count, "canonical ResNet parameter/state count");
  const bool bottleneck = depth >= 50;
  check::Require((bottleneck ? 3 : 2) * int(net.blocks.size()) + 2 == depth,
                 "canonical ResNet depth excludes shortcut projections");
  for (int stage = 0; stage < 5; ++stage) {
    const std::int64_t spatial  = stage < 2 ? 56 : 56 / (1 << (stage - 1));
    const std::int64_t channels = stage == 0 ? 64 : (64 << (stage - 1)) * (bottleneck ? 4 : 1);
    const auto&        shape    = (stage == 0 ? net.stem : net.stages[stage - 1]).shape();
    check::Require(shape == nn::Shape({2, spatial, spatial, channels}), "canonical ResNet stage dimensions");
  }
  check::Require(net.logits.shape() == nn::Shape({2, 1000}), "ResNet classifier shape");
  const auto parameters = p.parameters();
  const auto gradients  = nn::grad(nn::mean(nn::square(net.logits)), parameters);
  double     minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double norm = nn::sqrt(nn::sum(nn::square(gradients[i]))).item();
    check::Require(std::isfinite(norm) && norm > 0, "finite nonzero canonical ResNet derivative: " + p.names()[i]);
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  std::cout << "  ResNet-" << depth << ": parameters=" << p.count() << ", state=" << state.count()
            << "; stage sizes 56/28/14/7; training-mode gradient norms [" << minimum << ',' << maximum << "]"
            << std::endl;
}

/// Quarter-width ResNet with a 3x3 stride-one stem for 32x32 CIFAR images.
inline image_training::LogitsBuilder CompactResNet(int depth) {
  return [=](nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image, bool training) {
    resnet::ResNetConfig cfg;
    cfg.depth         = depth;
    cfg.classes       = 10;
    cfg.width_divisor = 4;
    cfg.small_stem    = true;
    return resnet::ResNet(p, state, image, cfg, training).logits;
  };
}

}  // namespace resnet_demo
