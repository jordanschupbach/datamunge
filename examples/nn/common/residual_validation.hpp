// Shared numerical/architecture diagnostics for the CNN examples.
// No fitting workflow lives here: each .cpp / Python / R script owns its
// data split, epoch/batch loop, objective, optimizer, checkpoint and report.
// These diagnostics are the same C++ checks called by all three languages.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <datamunge/nn/memory.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>

#include "common/architecture_checks.hpp"
#include "common/attention_cnn.hpp"
#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"
#include "common/non_local.hpp"
#include "common/octconv.hpp"
#include "common/preact_resnet.hpp"
#include "common/pyramidnet.hpp"
#include "common/repvgg.hpp"
#include "common/res2net.hpp"
#include "common/resnext.hpp"
#include "common/revnet.hpp"

namespace preactivation_resnet_checks {

namespace nn = datamunge::nn;

inline void IdentityChecks() {
  for (bool projection : {false, true}) {
    nn::ParameterStore p(17), state(17);
    const std::int64_t in    = projection ? 2 : 4;
    const int          count = projection ? 1 : 50, stride = projection ? 2 : 1;
    std::vector<float> values(static_cast<std::size_t>(25 * in));
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = (i % 2 ? -1.f : 1.f) * (.2f + .01f * i);
    nn::Tensor x     = nn::from_vector(values, {1, 5, 5, in});
    auto       build = [&](const nn::Tensor& input) {
      nn::Tensor h = input;
      for (int b = 0; b < count; ++b)
        h = preact::PreActBottleneck(p, state, "block" + std::to_string(b), h, 1, stride).output;
      return h;
    };
    {
      nn::NoGradGuard guard;
      build(x);
      for (int b = 0; b < count; ++b) {
        const auto name = "block" + std::to_string(b);
        p.set(name + "/conv3/W", nn::zeros({1, 1, 1, 4}));
        if (projection) {
          std::vector<float> kernel(static_cast<std::size_t>(in * 4), 0.f);
          for (int c = 0; c < 4; ++c)
            kernel[static_cast<std::size_t>((c % in) * 4 + c)] = 1;
          p.set(name + "/shortcut/W", nn::from_vector(kernel, {1, 1, in, 4}));
          p.set(name + "/bn1/gamma", nn::full({in}, std::sqrt(1.f + 1e-5f)));
        }
      }
    }
    x.requires_grad_(true);
    const nn::Tensor h        = build(x);
    const auto       gradient = nn::grad(nn::sum(h), {x})[0].to_vector<float>();
    const int        spatial  = projection ? 3 : 5;
    check::Require(
        h.shape() == nn::Shape({1, spatial, spatial, 4}), "preactivation shortcut dimensions");
    const auto y = h.to_vector<float>();
    for (int r = 0; r < spatial; ++r)
      for (int c = 0; c < spatial; ++c)
        for (int channel = 0; channel < 4; ++channel) {
          const float value =
              values[static_cast<std::size_t>((stride * r * 5 + stride * c) * in + channel % in)];
          check::Near(
              y[static_cast<std::size_t>((r * spatial + c) * 4 + channel)],
              projection ? std::max(0.f, value) : value,
              2e-6,
              "identity retains signed inputs; projection consumes BN/ReLU");
        }
    for (int r = 0; r < 5; ++r)
      for (int c = 0; c < 5; ++c)
        for (int channel = 0; channel < in; ++channel) {
          const auto  i = static_cast<std::size_t>((r * 5 + c) * in + channel);
          const float derivative =
              projection ? (r % 2 == 0 && c % 2 == 0 && values[i] > 0 ? 2.f : 0.f) : 1.f;
          check::Near(
              gradient[i], derivative, 2e-6, "raw identity or preactivated projection derivative");
        }
  }
  std::cout << "  fifty zero-residual blocks retain signed inputs and unit derivatives; "
               "preactivated projection checks passed\n";
}

inline void NonzeroResidualCheck() {
  nn::ParameterStore p(17), state(17);
  std::vector<float> values(36);
  for (int i = 0; i < 36; ++i)
    values[static_cast<std::size_t>(i)] = (i % 5 < 2 ? -1.f : 1.f) * (.3f + .03f * i);
  nn::Tensor x = nn::from_vector(values, {1, 3, 3, 4});
  {
    nn::NoGradGuard guard;
    preact::PreActBottleneck(p, state, "block", x, 1);
    for (int layer = 1; layer <= 3; ++layer) {
      const std::int64_t in = layer == 1 ? 4 : 1, out = layer == 3 ? 4 : 1,
                         kernel = layer == 2 ? 3 : 1;
      std::vector<float> weight(static_cast<std::size_t>(kernel * kernel * in * out), 0.f);
      for (std::int64_t c = 0; c < out; ++c)
        weight[static_cast<std::size_t>(
            (((kernel / 2) * kernel + kernel / 2) * in + 0) * out + c)] = layer == 3 ? .25f : 1.f;
      p.set(
          "block/conv" + std::to_string(layer) + "/W",
          nn::from_vector(weight, {kernel, kernel, in, out}));
      p.set("block/bn" + std::to_string(layer) + "/gamma", nn::full({in}, std::sqrt(1.f + 1e-5f)));
    }
  }
  auto total = [&](const nn::Tensor& input) {
    return nn::sum(preact::PreActBottleneck(p, state, "block", input, 1).output);
  };
  x.requires_grad_(true);
  const auto block      = preact::PreActBottleneck(p, state, "block", x, 1);
  const auto analytical = nn::grad(nn::sum(block.output), {x})[0].to_vector<float>();
  const auto y          = block.output.to_vector<float>();
  for (int position = 0; position < 9; ++position)
    for (int c = 0; c < 4; ++c) {
      const auto  i  = static_cast<std::size_t>(position * 4 + c);
      const float x0 = values[static_cast<std::size_t>(position * 4)];
      check::Near(
          y[i], values[i] + .25f * std::max(0.f, x0), 2e-6, "raw input plus controlled residual");
      check::Near(
          analytical[i],
          1.f + (c == 0 && x0 > 0 ? 1.f : 0.f),
          2e-6,
          "identity plus residual Jacobian");
    }
  const float     epsilon = .001f;
  double          error   = 0;
  nn::NoGradGuard guard;
  for (int index : {0, 4, 5, 8}) {
    auto plus = values, minus = values;
    plus[static_cast<std::size_t>(index)] += epsilon;
    minus[static_cast<std::size_t>(index)] -= epsilon;
    const float hi         = total(nn::from_vector(plus, {1, 3, 3, 4})).item<float>();
    const float lo         = total(nn::from_vector(minus, {1, 3, 3, 4})).item<float>();
    const float derivative = (hi - lo) / (2 * epsilon);
    error =
        std::max(error, double(std::abs(derivative - analytical[static_cast<std::size_t>(index)])));
    check::Near(
        derivative,
        analytical[static_cast<std::size_t>(index)],
        .002,
        "nonzero preactivation finite difference");
  }
  std::cout << "  nonzero residual matches host values and identity-plus-residual derivatives; "
               "finite-difference error="
            << error << '\n';
}

inline void FullChecks(int depth, std::int64_t parameters, std::int64_t state_parameters) {
  nn::ParameterStore         p(17), state(17);
  preact::PreActResNetConfig cfg;
  cfg.depth        = depth;
  cfg.update_stats = false;
  const auto net =
      preact::PreActResNet(p, state, architecture::SinePixels({2, 32, 32, 3}, .1, .13), cfg, true);
  check::Require(
      p.count() == parameters && state.count() == state_parameters,
      "canonical preactivation parameter/state counts");
  check::Require(3 * int(net.blocks.size()) + 2 == depth, "9n+2 bottleneck depth");
  for (int stage = 0; stage < 4; ++stage) {
    const std::int64_t spatial  = stage < 2 ? 32 : 32 / (1 << (stage - 1)),
                       channels = stage == 0 ? 16 : 64 << (stage - 1);
    check::Require(
        (stage == 0 ? net.stem : net.stages[stage - 1]).shape() ==
            nn::Shape({2, spatial, spatial, channels}),
        "canonical preactivation stage dimensions");
  }
  check::Require(net.logits.shape() == nn::Shape({2, 10}), "preactivation classifier dimensions");
  const auto [minimum, maximum] =
      architecture::GradientNormRange(p, nn::mean(nn::square(net.logits)), "preactivation");
  std::cout << "  preactivation ResNet-" << depth << ": parameters=" << parameters
            << ", state=" << state_parameters << "; stage sizes 32/16/8; gradient norms ["
            << minimum << ',' << maximum << "]" << std::endl;
}

} // namespace preactivation_resnet_checks

namespace pyramidnet_checks {

namespace nn = datamunge::nn;

inline std::vector<float> PatternValues(std::int64_t n, float scale, float phase) {
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i)
    v[static_cast<std::size_t>(i)] = scale * std::sin(.37f * static_cast<float>(i) + phase);
  return v;
}
inline nn::Tensor Pattern(const nn::Shape& shape, float scale, float phase) {
  std::int64_t n = 1;
  for (auto d : shape)
    n *= d;
  return nn::from_vector(PatternValues(n, scale, phase), shape);
}

inline void WidthChecks() {
  pyramidnet::PyramidNetConfig cfg; // 110 layers, alpha 48, basic units
  const auto                   widths = cfg.UnitWidths();
  check::Require(
      widths.size() == 54 && widths[0] == 17 && widths[1] == 18 && widths[2] == 19 &&
          widths.back() == 64,
      "PyramidNet-110 (alpha 48) width schedule");
  for (std::size_t i = 1; i < widths.size(); ++i)
    check::Require(
        widths[i] >= widths[i - 1] && widths[i] - widths[i - 1] <= 1,
        "additive unit-by-unit widening");
  pyramidnet::PyramidNetConfig deep;
  deep.depth             = 272;
  deep.alpha             = 200;
  deep.bottleneck        = true;
  const auto deep_widths = deep.UnitWidths();
  check::Require(
      deep_widths.size() == 90 && deep_widths.back() == 216,
      "PyramidNet-272 (alpha 200) width schedule");
  std::cout << "  PyramidNet-110 alpha 48 widths: " << widths[0] << ',' << widths[1] << ','
            << widths[2] << ",...," << widths[17] << " | " << widths[18] << ",...," << widths[35]
            << " | " << widths[36] << ",...," << widths.back()
            << "; PyramidNet-272 alpha 200 ends at " << deep_widths.back()
            << "x4=" << 4 * deep_widths.back() << std::endl;
}

// Zero last BN: output is the parameter-free shortcut; derivatives are exact.
inline void UnitChecks() {
  for (bool bottleneck : {false, true})
    for (int stride : {1, 2}) {
      nn::ParameterStore p(17), state(17);
      const int          planes = bottleneck ? 2 : 5, out_channels = bottleneck ? 8 : 5,
                spatial = 6 / stride;
      const auto xv     = PatternValues(2 * 6 * 6 * 3, 1, .9f);
      nn::Tensor x      = nn::from_vector(xv, {2, 6, 6, 3});
      {
        nn::NoGradGuard guard;
        pyramidnet::PyramidBlock(p, state, "unit", x, planes, bottleneck, stride);
        const std::string last = bottleneck ? "unit/bn4/" : "unit/bn3/";
        for (const auto* part : {"gamma", "beta"})
          p.set(last + part, nn::zeros({out_channels}));
      }
      const auto weights = Pattern({2, spatial, spatial, out_channels}, 1, .4f);
      x.requires_grad_(true);
      const auto unit = pyramidnet::PyramidBlock(p, state, "unit", x, planes, bottleneck, stride);
      const auto gradient = nn::grad(nn::sum(unit.output * weights), {x})[0].to_vector<float>();
      check::Require(
          unit.output.shape() == nn::Shape({2, spatial, spatial, out_channels}),
          "pyramidal unit output dimensions");
      const auto y = unit.output.to_vector<float>(), w = weights.to_vector<float>();
      auto       xt = [&](int b, int r, int c, int ch) {
        return xv[static_cast<std::size_t>(((b * 6 + r) * 6 + c) * 3 + ch)];
      };
      auto at = [&](int b, int r, int c, int ch) {
        return static_cast<std::size_t>(((b * spatial + r) * spatial + c) * out_channels + ch);
      };
      const int k            = stride;
      double    output_error = 0, gradient_error = 0;
      int       negatives = 0;
      for (int b = 0; b < 2; ++b)
        for (int r = 0; r < spatial; ++r)
          for (int c = 0; c < spatial; ++c)
            for (int ch = 0; ch < out_channels; ++ch) {
              double expected = 0;
              if (ch < 3)
                for (int dr = 0; dr < k; ++dr)
                  for (int dc = 0; dc < k; ++dc)
                    expected += xt(b, r * k + dr, c * k + dc, ch) / double(k * k);
              if (expected < 0)
                negatives += y[at(b, r, c, ch)] < 0;
              output_error = std::max(output_error, std::abs(y[at(b, r, c, ch)] - expected));
            }
      for (int b = 0; b < 2; ++b)
        for (int r = 0; r < 6; ++r)
          for (int c = 0; c < 6; ++c)
            for (int ch = 0; ch < 3; ++ch)
              gradient_error = std::max(
                  gradient_error,
                  std::abs(
                      gradient[static_cast<std::size_t>(((b * 6 + r) * 6 + c) * 3 + ch)] -
                      w[at(b, r / k, c / k, ch)] / double(k * k)));
      check::Near(
          output_error, 0, 1e-6, "zero-residual unit returns the pooled, zero-padded input");
      check::Near(gradient_error, 0, 1e-6, "identity or pooling-spread input derivatives");
      check::Require(negatives > 0, "negative shortcut values pass without post-addition ReLU");
      std::cout << "  " << (bottleneck ? "bottleneck" : "basic") << " unit, stride " << stride
                << ": 3->" << out_channels << " channels, output error=" << output_error
                << ", derivative error=" << gradient_error
                << ", negative outputs kept=" << negatives << std::endl;
    }
}

inline void FullChecks() {
  struct Case {
    int          depth;
    double       alpha;
    bool         bottleneck, imagenet;
    std::int64_t params, state;
    double       paper;
  };
  const std::vector<Case> cases = {
      {110, 48, false, false, 1772706, 13168, 1.7},
      {110, 84, false, false, 3904446, 19108, 3.8},
      {110, 270, false, false, 28485477, 49798, 28.3},
      {164, 48, true, false, 1715442, 43744, 1.7},
      {164, 270, true, false, 27216021, 165844, 27.0},
      {200, 240, true, false, 26752702, 181984, 26.6},
      {236, 220, true, false, 26969046, 198824, 26.8},
      {272, 200, true, false, 26210842, 210864, 26.0},
      {200, 300, true, true, 62368168, 285736, 62.1},
      {200, 450, true, true, 116775213, 386236, 116.4}};
  for (const auto& c : cases) {
    nn::ParameterStore           p(17), state(17);
    pyramidnet::PyramidNetConfig cfg;
    cfg.depth      = c.depth;
    cfg.alpha      = c.alpha;
    cfg.bottleneck = c.bottleneck;
    cfg.imagenet   = c.imagenet;
    cfg.classes    = c.imagenet ? 1000 : 10;
    {
      nn::NoGradGuard guard;
      const int       size = c.imagenet ? 64 : 32;
      pyramidnet::PyramidNet(p, state, Pattern({1, size, size, 3}, .5f, 0), cfg);
    }
    check::Require(
        p.count() == c.params && state.count() == c.state, "PyramidNet parameter/state counts");
    // Running state holds a mean and variance per BN channel, as many as BN gamma/beta.
    const double without_bn = double(p.count() - state.count()) / 1e6;
    std::cout << "  " << (c.imagenet ? "ImageNet" : "CIFAR-10") << " PyramidNet-" << c.depth
              << " alpha " << c.alpha << (c.bottleneck ? " bottleneck" : " basic")
              << ": parameters=" << p.count() << ", state=" << state.count()
              << ", without BN scale/shift=" << std::fixed << std::setprecision(2) << without_bn
              << "M (paper " << std::setprecision(1) << c.paper << "M)" << std::defaultfloat
              << std::setprecision(6) << std::endl;
  }
  nn::ParameterStore           p(17), state(17);
  pyramidnet::PyramidNetConfig cfg;
  cfg.depth        = 272;
  cfg.alpha        = 200;
  cfg.bottleneck   = true;
  cfg.update_stats = false;
  const auto net   = pyramidnet::PyramidNet(p, state, Pattern({2, 32, 32, 3}, .5f, 0), cfg, true);
  const std::int64_t spatial[3]  = {32, 16, 8},
                     channels[3] = {net.channels[29], net.channels[59], net.channels[89]};
  for (int stage = 0; stage < 3; ++stage)
    check::Require(
        net.stages[stage].shape() ==
            nn::Shape({2, spatial[stage], spatial[stage], channels[stage]}),
        "PyramidNet stage dimensions");
  check::Require(
      channels[2] == 864 && net.logits.shape() == nn::Shape({2, 10}),
      "PyramidNet-272 final width and classifier");
  // Training-mode BN removes any per-channel constant from its input. The
  // shifts of the stem BN and of every unit's last BN add such constants to
  // the residual stream, whose consumers are all BNs; a bottleneck's first BN
  // feeds an unpadded 1x1 convolution, which keeps its shift constant for the
  // next BN. Those derivatives vanish analytically; every other one must not.
  const auto          gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
  std::vector<double> norms;
  double              minimum = std::numeric_limits<double>::infinity(), maximum = 0, redundant = 0;
  int                 redundant_count = 0;
  for (const auto& g : gradients) {
    norms.push_back(nn::sqrt(nn::sum(nn::square(g))).item());
    maximum = std::max(maximum, norms.back());
  }
  const auto& names = p.names();
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const double norm = norms[i];
    const auto&  name = names[i];
    check::Require(std::isfinite(norm), "finite PyramidNet derivative: " + name);
    auto ends = [&](const std::string& suffix) {
      return name.size() > suffix.size() &&
             name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (name == "stem/bn/beta" || ends("/bn1/beta") || ends("/bn4/beta")) {
      redundant = std::max(redundant, norm);
      ++redundant_count;
    } else {
      minimum = std::min(minimum, norm);
    }
  }
  check::Require(
      redundant_count == 181 && redundant < 1e-6 * maximum,
      "vanishing derivatives of BN shifts removed downstream");
  check::Require(
      minimum > 1e-5 * maximum, "nonzero derivatives for every other PyramidNet parameter");
  std::cout << "  PyramidNet-272 alpha 200: stage outputs " << channels[0] << '/' << channels[1]
            << '/' << channels[2] << " channels at 32/16/8; training-mode gradient norms ["
            << minimum << ',' << maximum << "]; " << redundant_count
            << " stem/first/last BN shift derivatives vanish (largest " << redundant << ")"
            << std::endl;
}

} // namespace pyramidnet_checks

namespace res2net_checks {

namespace nn = datamunge::nn;

inline void ScaleChecks(int scale, bool stage, int stride) {
  nn::ParameterStore     p(17), state(17);
  resnet::ConvNormConfig cfg;
  cfg.scale                  = true;
  cfg.epsilon                = 1e-5; // inference mode: moving mean 0, variance 1
  const int          spatial = (11 + stride - 1) / stride, processed = scale == 1 ? 1 : scale - 1;
  std::vector<float> x(static_cast<std::size_t>(11 * 11 * scale));
  for (int r = 0; r < 11; ++r)
    for (int c = 0; c < 11; ++c)
      for (int group = 0; group < scale; ++group)
        x[static_cast<std::size_t>((r * 11 + c) * scale + group)] = group + 1 + .01f * r + .02f * c;
  nn::Tensor input = nn::from_vector(x, {1, 11, 11, scale});
  {
    nn::NoGradGuard guard; // create the parameters, then control the filters
    res2net::Res2ScaleTransform(p, state, "scales", input, 1, scale, cfg, stage, stride);
    for (int group = 0; group < processed; ++group) {
      const auto name = "scales/group" + std::to_string(group);
      p.set(name + "/W", nn::full({3, 3, 1, 1}, 1. / 9));
      p.set(name + "/bn/gamma", nn::full({1}, std::sqrt(1 + cfg.epsilon)));
    }
  }
  input.requires_grad_(true);
  const auto net =
      res2net::Res2ScaleTransform(p, state, "scales", input, 1, scale, cfg, stage, stride);
  check::Require(
      net.mixed.shape() == nn::Shape({1, spatial, spatial, scale}),
      "Res2Net scale concatenation dimensions");
  const auto                       mixed = net.mixed.to_vector<float>();
  std::vector<std::vector<double>> reference(
      static_cast<std::size_t>(scale),
      std::vector<double>(static_cast<std::size_t>(spatial * spatial)));
  double error = 0;
  for (int group = 0; group < scale; ++group) {
    const bool bypass = scale > 1 && group == scale - 1;
    for (int r = 0; r < spatial; ++r)
      for (int c = 0; c < spatial; ++c) {
        double value = 0;
        if (bypass && !stage)
          value = x[static_cast<std::size_t>((r * 11 + c) * scale + group)];
        else
          for (int dr = -1; dr <= 1; ++dr)
            for (int dc = -1; dc <= 1; ++dc) {
              const int ir = r * stride + dr, ic = c * stride + dc;
              if (ir < 0 || ir >= 11 || ic < 0 || ic >= 11)
                continue;
              value += (x[static_cast<std::size_t>((ir * 11 + ic) * scale + group)] +
                        (!stage && group > 0 ? reference[static_cast<std::size_t>(group - 1)]
                                                        [static_cast<std::size_t>(ir * 11 + ic)]
                                             : 0)) /
                       9;
            }
        reference[static_cast<std::size_t>(group)][static_cast<std::size_t>(r * spatial + c)] =
            value;
        error = std::max(
            error,
            std::abs(mixed[static_cast<std::size_t>((r * spatial + c) * scale + group)] - value));
      }
    nn::GradOptions keep;
    keep.retain_graph = true;
    // Derivative of the group's centre output with respect to the whole input.
    const nn::Tensor center = nn::slice(
        net.groups[static_cast<std::size_t>(group)],
        {0, spatial / 2, spatial / 2, 0},
        {1, 1, 1, 1});
    const auto derivative = nn::grad(nn::sum(center), {input}, keep)[0].to_vector<float>();
    for (int channel = 0; channel < scale; ++channel) {
      int diameter = 0;
      if (stage || scale == 1)
        diameter = channel == group ? 3 : 0;
      else if (bypass)
        diameter = channel == group ? 1 : 0;
      else if (channel <= group)
        diameter = 2 * (group - channel + 1) + 1;
      int    support = 0, low_r = 11, high_r = -1, low_c = 11, high_c = -1;
      double mass = 0;
      for (int r = 0; r < 11; ++r)
        for (int c = 0; c < 11; ++c) {
          const double d = derivative[static_cast<std::size_t>((r * 11 + c) * scale + channel)];
          mass += d;
          if (std::abs(d) > 1e-9) {
            ++support;
            low_r  = std::min(low_r, r);
            high_r = std::max(high_r, r);
            low_c  = std::min(low_c, c);
            high_c = std::max(high_c, c);
          }
        }
      check::Require(support == diameter * diameter, "hierarchical receptive-field support count");
      if (diameter)
        check::Require(
            high_r - low_r + 1 == diameter && high_c - low_c + 1 == diameter,
            "hierarchical receptive-field diameter");
      check::Near(
          mass,
          diameter ? 1 : 0,
          2e-6,
          "connected split derivative mass and disconnected split zeros");
    }
  }
  check::Near(
      error, 0, 1e-5, "host multi-scale convolution/pooling outputs including padded edges");
  std::cout << "  scale=" << scale << ", " << (stage ? "stage" : "normal") << ", stride=" << stride
            << ": host output error=" << error << "; all split derivative supports passed\n";
}

inline void FullChecks() {
  nn::ParameterStore     p(17), state(17);
  res2net::Res2NetConfig cfg;
  cfg.update_stats = false;
  std::vector<float> pixels(2 * 224 * 224 * 3);
  for (std::size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = .1f * std::sin(static_cast<std::int64_t>(i) * .13f);
  const auto net = res2net::Res2Net(p, state, nn::from_vector(pixels, {2, 224, 224, 3}), cfg, true);
  check::Require(
      p.count() == 25699120 && state.count() == 59492 && net.blocks.size() == 16,
      "canonical Res2Net parameter/state/block counts");
  for (int stage = 0; stage < 5; ++stage) {
    const std::int64_t spatial  = stage < 2 ? 56 : 56 / (1 << (stage - 1)),
                       channels = stage == 0 ? 64 : 256 << (stage - 1);
    const auto& shape =
        (stage == 0 ? net.stem : net.stages[static_cast<std::size_t>(stage - 1)]).shape();
    check::Require(
        shape == nn::Shape({2, spatial, spatial, channels}), "canonical Res2Net stage dimensions");
  }
  check::Require(net.logits.shape() == nn::Shape({2, 1000}), "Res2Net classifier dimensions");
  const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
  double     minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double norm = nn::sqrt(nn::sum(nn::square(gradients[i]))).item();
    check::Require(
        std::isfinite(norm) && norm > 0, "finite nonzero Res2Net derivative: " + p.names()[i]);
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  std::cout << "  Res2Net-50 26w/4s: parameters=" << p.count() << ", state=" << state.count()
            << "; gradient norms [" << minimum << ',' << maximum << "]" << std::endl;
}

} // namespace res2net_checks

namespace resnext_checks {

namespace nn = datamunge::nn;

inline void AggregationChecks() {
  for (int stride : {1, 2}) {
    nn::ParameterStore     p(17), state(17);
    resnet::ConvNormConfig cfg;
    cfg.scale   = true;
    cfg.epsilon = 1e-5;
    std::vector<float> values(2 * 5 * 5 * 8);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = .2f + .3f * std::sin(.17f * static_cast<std::int64_t>(i));
    nn::Tensor input = nn::from_vector(values, {2, 5, 5, 8});
    {
      nn::NoGradGuard guard; // create the parameters, then set nontrivial normalization state
      resnext::ResNeXtBlock(p, state, "block", input, 2, 2, 64, cfg, stride);
      for (const auto& name : state.names()) {
        const auto         n = state.at(name).numel();
        std::vector<float> v(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
          v[static_cast<std::size_t>(i)] =
              name.find("moving_var") != std::string::npos ? .7f + .03f * i : .03f * (i - 2);
        state.set(name, nn::from_vector(v, {n}));
      }
      for (const auto& name : p.names())
        if (name.find("/bn/") != std::string::npos) {
          const auto         n = p.at(name).numel();
          std::vector<float> v(static_cast<std::size_t>(n));
          for (std::int64_t i = 0; i < n; ++i)
            v[static_cast<std::size_t>(i)] =
                name.find("gamma") != std::string::npos ? .8f + .03f * i : .1f + .01f * i;
          p.set(name, nn::from_vector(v, {n}));
        }
    }
    input.requires_grad_(true);
    const auto grouped = resnext::ResNeXtBlock(p, state, "block", input, 2, 2, 64, cfg, stride);
    auto       norm =
        [&](const nn::Tensor& x, const std::string& layer, std::int64_t start, std::int64_t count) {
          const auto name  = "block/" + layer + "/bn/";
          auto       slice = [&](const nn::ParameterStore& store, const std::string& part) {
            return nn::slice_axis(store.at(name + part), 0, start, start + count);
          };
          return (x - slice(state, "moving_mean")) *
                     nn::rsqrt(slice(state, "moving_var") + cfg.epsilon) * slice(p, "gamma") +
                 slice(p, "beta");
        };
    std::vector<nn::Tensor> branches;
    for (int group = 0; group < 2; ++group) {
      const nn::Tensor first  = nn::slice_axis(p.at("block/conv1/W"), 3, 2 * group, 2 * group + 2);
      nn::Tensor       h      = nn::relu(norm(
          nn::conv2d(input, first, nn::conv_options(1, nn::Padding::Valid)),
          "conv1",
          2 * group,
          2));
      const nn::Tensor middle = nn::slice_axis(p.at("block/conv2/W"), 3, 2 * group, 2 * group + 2);
      h                       = nn::conv2d(
          nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}),
          middle,
          nn::conv_options(stride, nn::Padding::Valid));
      h                     = nn::relu(norm(h, "conv2", 2 * group, 2));
      const nn::Tensor last = nn::slice_axis(p.at("block/conv3/W"), 2, 2 * group, 2 * group + 2);
      branches.push_back(nn::conv2d(h, last, nn::conv_options(1, nn::Padding::Valid)));
    }
    nn::Tensor shortcut = input;
    if (stride == 2)
      shortcut = norm(
          nn::conv2d(input, p.at("block/shortcut/W"), nn::conv_options(2, nn::Padding::Valid)),
          "shortcut",
          0,
          8);
    const nn::Tensor aggregated =
        nn::relu(shortcut + norm(branches[0] + branches[1], "conv3", 0, 8));
    auto variables = p.parameters();
    variables.insert(variables.begin(), input);
    const auto ga = nn::grad(nn::mean(nn::square(grouped.output)), variables);
    const auto gb = nn::grad(nn::mean(nn::square(aggregated)), variables);
    std::vector<std::pair<nn::Tensor, nn::Tensor>> pairs = {{grouped.output, aggregated}};
    for (std::size_t i = 0; i < ga.size(); ++i)
      pairs.emplace_back(ga[i], gb[i]);
    double error = 0;
    for (const auto& [x, y] : pairs) {
      check::Require(x.shape() == y.shape(), "equivalent bottleneck tensor shapes");
      const auto a = x.to_vector<float>(), b = y.to_vector<float>();
      for (std::size_t i = 0; i < a.size(); ++i)
        error = std::max(
            error,
            std::abs(double(a[i]) - b[i]) /
                std::max({1., std::abs(double(a[i])), std::abs(double(b[i]))}));
    }
    check::Near(
        error, 0, 3e-5, "split-transform-sum and grouped outputs/input/all parameter derivatives");
    std::cout << "  grouped versus aggregated, stride " << stride
              << ": all outputs and derivatives agree; error=" << error << '\n';
  }
  const auto kernel_budget = [](int cardinality, int width) {
    return 2 * 256 * cardinality * width + 9 * cardinality * width * width;
  };
  check::Require(
      kernel_budget(1, 64) == 69632 && kernel_budget(32, 4) == 70144,
      "matched bottleneck kernel budgets");
  std::cout << "  residual-kernel counts at 256 input/output channels: 1x64d="
            << kernel_budget(1, 64) << ", 32x4d=" << kernel_budget(32, 4)
            << " (excludes BN and projection)\n";
}

inline void FullChecks() {
  nn::ParameterStore     p(17), state(17);
  resnext::ResNeXtConfig cfg;
  cfg.update_stats = false;
  std::vector<float> pixels(2 * 224 * 224 * 3);
  for (std::size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = .1f * std::sin(static_cast<std::int64_t>(i) * .13f);
  const auto net = resnext::ResNeXt(p, state, nn::from_vector(pixels, {2, 224, 224, 3}), cfg, true);
  check::Require(
      p.count() == 25028904 && state.count() == 68224,
      "canonical ResNeXt-50 parameter/state counts");
  check::Require(
      net.blocks.size() == 16 && net.stages.size() == 4, "canonical ResNeXt-50 block counts");
  for (int stage = 0; stage < 5; ++stage) {
    const std::int64_t spatial  = stage < 2 ? 56 : 56 / (1 << (stage - 1)),
                       channels = stage == 0 ? 64 : 256 << (stage - 1);
    const auto& shape =
        (stage == 0 ? net.stem : net.stages[static_cast<std::size_t>(stage - 1)]).shape();
    check::Require(
        shape == nn::Shape({2, spatial, spatial, channels}), "canonical ResNeXt stage dimensions");
  }
  check::Require(net.logits.shape() == nn::Shape({2, 1000}), "ResNeXt classifier dimensions");
  const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
  double     minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double norm = nn::sqrt(nn::sum(nn::square(gradients[i]))).item();
    check::Require(
        std::isfinite(norm) && norm > 0, "finite nonzero ResNeXt derivative: " + p.names()[i]);
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  std::cout << "  ResNeXt-50 32x4d: parameters=" << p.count() << ", state=" << state.count()
            << "; gradient norms [" << minimum << ',' << maximum << "]" << std::endl;
}

} // namespace resnext_checks

namespace revnet_checks {

namespace nn = datamunge::nn;
using nn::Tensor;

inline Tensor Pattern(const nn::Shape& shape, float scale, float phase) {
  std::vector<float> v(static_cast<std::size_t>(nn::shape_numel(shape)));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = scale * std::sin(.37f * static_cast<float>(i) + phase);
  return nn::from_vector(v, shape);
}
inline double RelativeDifference(const Tensor& a, const Tensor& b) {
  check::Require(a.shape() == b.shape(), "matching RevNet comparison shapes");
  const auto x = a.to_vector<float>(), y = b.to_vector<float>();
  double     difference = 0, norm = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double d = double(x[i]) - y[i];
    difference += d * d;
    norm += double(y[i]) * y[i];
  }
  return std::sqrt(difference / std::max(norm, 1e-30));
}

inline void InverseChecks() {
  for (bool bottleneck : {false, true})
    for (bool training : {true, false}) {
      // tflow seeds this fixture with 17. With nn's generator, seed 17 leaves
      // the one-channel bottleneck's inference-mode ReLU almost dead (the unit
      // changes its input by only .06), so the fixture uses seed 1 (>.2 for
      // seeds 1-7 except 17's stream).
      nn::ParameterStore       p(1), state(1);
      const revnet::RevContext c{p, state, training, .9, bottleneck, false};
      const Tensor             value  = Pattern({4, 5, 5, 8}, 1, .2f);
      auto                     halves = nn::split(value, 2, 3);
      {
        nn::NoGradGuard guard; // create the parameters, then give the statistics nontrivial values
        revnet::RevBlockForward(c, "unit", {halves[0], halves[1]}, 8, 8, 1, false);
        for (const auto& name : state.names()) {
          Tensor values = Pattern(state.at(name).shape(), .3f, static_cast<float>(name.size()));
          if (name.find("moving_var") != std::string::npos)
            values = values + .8;
          state.set(name, values);
        }
      }
      nn::NoGradGuard guard;
      const auto   y = revnet::RevBlockForward(c, "unit", {halves[0], halves[1]}, 8, 8, 1, false);
      const auto   x = revnet::RevBlockInverse(c, "unit", y, 8);
      const Tensor restored = nn::concat({x.first, x.second}, 3);
      const double error    = nn::max(nn::abs(restored - value)).item();
      const double change   = nn::max(nn::abs(y.first - halves[0])).item();
      check::Require(change > .1, "the unit changes its input");
      check::Near(error, 0, 1e-5, "RevNet unit inverse restores its input");
      std::cout << "  " << (bottleneck ? "bottleneck" : "basic") << " unit, "
                << (training ? "training" : "inference") << " BN: inverse error=" << error
                << ", change=" << change << std::endl;
    }
}

inline std::pair<Tensor, Tensor> Batch(int batch) {
  std::vector<std::int64_t> labels(static_cast<std::size_t>(batch));
  for (int i = 0; i < batch; ++i)
    labels[static_cast<std::size_t>(i)] = (7 * i + 3) % 10;
  return {Pattern({batch, 32, 32, 3}, 1, .5f), nn::from_vector(labels)};
}
inline Tensor Loss(const Tensor& logits, const Tensor& labels, int classes) {
  return -nn::mean(
      nn::sum(
          nn::one_hot(labels, classes, 1.0, 0.0, logits.dtype()) * nn::log_softmax(logits, 1),
          {1}));
}

inline void GradientChecks() {
  for (int depth : {38, 164}) {
    const auto         cfg = revnet::RevNetConfig::Cifar(depth);
    nn::ParameterStore p(17), state(17);
    const auto         batch = Batch(8);
    const Tensor       x = batch.first, labels = batch.second;
    auto               derivatives = [&](revnet::Mode mode, revnet::RevNetOutputs* keep) {
      auto net = revnet::RevNet(p, state, x, cfg, true, mode, false);
      auto g   = nn::grad(Loss(net.logits, labels, cfg.classes), p.parameters());
      if (keep)
        *keep = std::move(net);
      return g;
    };
    revnet::RevNetOutputs net;
    const auto            standard   = derivatives(revnet::Mode::Ordinary, &net);
    const auto            reversible = derivatives(revnet::Mode::Reversible, nullptr);
    const auto            stored     = derivatives(revnet::Mode::StoredUnits, nullptr);
    // The reconstructions the reversible pass sees: walk back from the last
    // output, inverting every invertible unit and taking stored inputs as is.
    double drift   = 0;
    int    rebuilt = 0;
    {
      nn::NoGradGuard          guard;
      const revnet::RevContext c{p, state, true, cfg.momentum, cfg.bottleneck, false};
      revnet::RevPair          y{
          net.units.back().output.first.detach(), net.units.back().output.second.detach()};
      for (std::size_t k = net.units.size(); k-- > 0;) {
        const auto& u = net.units[k];
        if (u.stored) {
          y = {u.input.first.detach(), u.input.second.detach()};
          continue;
        }
        y     = revnet::RevBlockInverse(c, u.name, y, u.out);
        drift = std::max(
            drift,
            RelativeDifference(
                nn::concat({y.first, y.second}, 3),
                nn::concat({u.input.first, u.input.second}, 3)));
        ++rebuilt;
      }
    }
    const std::size_t n        = standard.size();
    double            gradient = 0, unit = 0;
    std::string       worst;
    for (std::size_t i = 0; i < n; ++i) {
      const double d = RelativeDifference(reversible[i], standard[i]);
      if (d > gradient) {
        gradient = d;
        worst    = p.names()[i];
      }
      unit = std::max(unit, RelativeDifference(stored[i], standard[i]));
    }
    check::Near(
        unit, 0, 1e-6, "unit-by-unit derivatives from stored inputs equal ordinary backprop");
    check::Near(drift, 0, 1e-4, "reconstructed unit inputs match the forward pass");
    check::Near(
        gradient,
        0,
        .05,
        "reversible derivatives stay close to ordinary backprop (worst: " + worst + ")");
    std::cout << "  RevNet-" << depth << ": " << n
              << " derivatives; unit-by-unit from stored inputs vs ordinary backprop: " << unit
              << "; " << rebuilt << " reconstructed inputs drift up to " << drift
              << ", which moves the reversible derivatives by up to " << gradient << " (" << worst
              << ")" << std::endl;
  }
}

// Peak live tensor bytes of one training step (forward and gradients) above
// the parameters, after a warm-up step that creates them.
inline double PeakMiB(const revnet::RevNetConfig& cfg, revnet::Mode mode, int batch) {
  nn::ParameterStore p(17), state(17);
  const auto         data = Batch(batch);
  const Tensor       x = data.first, labels = data.second;
  auto               step = [&] {
    auto net = revnet::RevNet(p, state, x, cfg, true, mode, false);
    return nn::grad(Loss(net.logits, labels, cfg.classes), p.parameters());
  };
  step();
  nn::PeakMemoryScope scope;
  step();
  return scope.peak_above_baseline() / 1048576.0;
}
inline void MemoryChecks() {
  std::vector<double> ratios;
  for (int n : {3, 9, 18}) {
    auto cfg       = revnet::RevNetConfig::Cifar(38);
    cfg.units      = {n, n, n};
    const double a = PeakMiB(cfg, revnet::Mode::Ordinary, 64),
                 b = PeakMiB(cfg, revnet::Mode::Reversible, 64),
                 s = PeakMiB(cfg, revnet::Mode::StoredUnits, 64);
    check::Require(a > 0 && b > 0, "measured peak step memory");
    ratios.push_back(a / b);
    std::cout << "  " << n << " units per stage (" << 12 * n + 2
              << " layers), batch 64: peak step memory ordinary=" << std::fixed
              << std::setprecision(1) << a << " MiB, reversible=" << b
              << " MiB (every unit checkpointed=" << s << " MiB)" << std::defaultfloat
              << std::setprecision(6) << std::endl;
  }
  check::Require(
      ratios[0] < ratios[1] && ratios[1] < ratios[2] && ratios[2] > 3,
      "reversible saving grows with depth");
}

inline void CountChecks() {
  struct Case {
    const char*          name;
    revnet::RevNetConfig cfg;
    int                  size;
    std::int64_t         params, state;
  };
  const std::vector<Case> cases = {
      {"RevNet-38", revnet::RevNetConfig::Cifar(38), 32, 464890, 2672},
      {"RevNet-110", revnet::RevNetConfig::Cifar(110), 32, 1729194, 8256},
      {"RevNet-164", revnet::RevNetConfig::Cifar(164), 32, 1748778, 24768},
      {"ImageNet RevNet-56", revnet::RevNetConfig::ImageNet(56), 64, 27553128, 51200},
      {"ImageNet RevNet-104", revnet::RevNetConfig::ImageNet(104), 64, 45428072, 100352}};
  for (const auto& c : cases) {
    nn::ParameterStore p(17), state(17);
    nn::NoGradGuard    guard;
    const auto         net = revnet::RevNet(p, state, nn::zeros({1, c.size, c.size, 3}), c.cfg);
    check::Require(
        p.count() == c.params && state.count() == c.state,
        std::string(c.name) + " parameter/state counts");
    int stored = 0;
    for (const auto& u : net.units)
      stored += u.stored;
    std::cout << "  " << c.name << ": parameters=" << p.count() << ", state=" << state.count()
              << "; " << net.units.size() << " units, " << stored << " with stored inputs"
              << std::endl;
  }
}

} // namespace revnet_checks

namespace cbam_checks {

namespace nn = datamunge::nn;
using mobile_checks::EndsWith;
using mobile_checks::Pattern;

inline void ModuleChecks() {
  const int          channels = 16, reduction = 4, hidden = channels / reduction, size = 9;
  nn::ParameterStore p(17), state(17);
  const nn::Tensor   x = Pattern({1, size, size, channels}, 1, .2f);
  attention_cnn::Cbam(p, state, "cbam", x, reduction, false, .9); // create the parameters
  {
    nn::NoGradGuard guard;
    for (const auto& name : p.names())
      if (EndsWith(name, "/b") || name.find("/bn/") != std::string::npos)
        p.set(name, Pattern(p.at(name).shape(), .3f, static_cast<float>(name.size())));
    state.set("cbam/spatial/bn/moving_mean", Pattern({1}, .2f, 1));
    state.set("cbam/spatial/bn/moving_var", nn::full({1}, 1.3));
  }
  const auto output =
      attention_cnn::Cbam(p, state, "cbam", x, reduction, false, .9).output.to_vector<float>();
  auto get = [&](nn::ParameterStore& s, const std::string& name) {
    return s.at(name).to_vector<float>();
  };
  const auto w1 = get(p, "cbam/mlp1/W"), b1 = get(p, "cbam/mlp1/b"), w2 = get(p, "cbam/mlp2/W"),
             b2 = get(p, "cbam/mlp2/b"), sw = get(p, "cbam/spatial/W"),
             gamma = get(p, "cbam/spatial/bn/gamma"), beta = get(p, "cbam/spatial/bn/beta"),
             mm    = get(state, "cbam/spatial/bn/moving_mean"),
             mv    = get(state, "cbam/spatial/bn/moving_var");
  const auto value = x.to_vector<float>();
  auto at = [&](int r, int c, int ch) { return double(value[(r * size + c) * channels + ch]); };
  // Channel gate.
  std::vector<double> mean(channels, 0), peak(channels, -INFINITY), logit(channels, 0);
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c)
      for (int ch = 0; ch < channels; ++ch) {
        mean[ch] += at(r, c, ch) / (size * size);
        peak[ch] = std::max(peak[ch], at(r, c, ch));
      }
  for (const auto& pooled : {mean, peak}) {
    std::vector<double> h(hidden);
    for (int j = 0; j < hidden; ++j) {
      h[j] = b1[j];
      for (int ch = 0; ch < channels; ++ch)
        h[j] += pooled[ch] * w1[ch * hidden + j];
      h[j] = std::max(h[j], 0.);
    }
    for (int ch = 0; ch < channels; ++ch) {
      logit[ch] += b2[ch];
      for (int j = 0; j < hidden; ++j)
        logit[ch] += h[j] * w2[j * channels + ch];
    }
  }
  std::vector<double> gate(channels);
  for (int ch = 0; ch < channels; ++ch)
    gate[ch] = 1 / (1 + std::exp(-logit[ch]));
  // Spatial gate on the refined features.
  std::vector<double> maxmap(size * size, -INFINITY), meanmap(size * size, 0);
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c)
      for (int ch = 0; ch < channels; ++ch) {
        const double v       = at(r, c, ch) * gate[ch];
        maxmap[r * size + c] = std::max(maxmap[r * size + c], v);
        meanmap[r * size + c] += v / channels;
      }
  const double scale = gamma[0] / std::sqrt(mv[0] + 1e-5), shift = beta[0] - mm[0] * scale;
  double       error = 0;
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c) {
      double z = 0;
      for (int dr = -3; dr <= 3; ++dr)
        for (int dc = -3; dc <= 3; ++dc) {
          const int rr = r + dr, cc = c + dc;
          if (rr < 0 || rr >= size || cc < 0 || cc >= size)
            continue;
          z += maxmap[rr * size + cc] * sw[((dr + 3) * 7 + dc + 3) * 2] +
               meanmap[rr * size + cc] * sw[((dr + 3) * 7 + dc + 3) * 2 + 1];
        }
      const double spatial = 1 / (1 + std::exp(-(z * scale + shift)));
      for (int ch = 0; ch < channels; ++ch)
        error = std::max(
            error,
            std::abs(output[(r * size + c) * channels + ch] - at(r, c, ch) * gate[ch] * spatial));
    }
  check::Near(error, 0, 2e-6, "CBAM channel and spatial attention");
  std::cout << "  CBAM module (16 channels, reduction 4, 9x9): output equals host-computed channel "
               "gate x spatial gate x "
               "input (error "
            << error << ")" << std::endl;
}

inline void FullChecks() {
  nn::ParameterStore                   p(17), state(17);
  const nn::Tensor                     input = Pattern({2, 224, 224, 3}, 1, 0);
  attention_cnn::AttentionResNetConfig cfg;
  cfg.attention    = attention_cnn::ChannelAttention::kCbam;
  cfg.update_stats = false;
  {
    nn::NoGradGuard guard;
    attention_cnn::AttentionResNet(
        p, state, nn::zeros({1, 64, 64, 3}), cfg); // create the parameters
  }
  check::Require(p.count() == 28089624, "CBAM-ResNet-50 parameter count");
  mobile_checks::ShiftBetas(p, false);
  const auto  net     = attention_cnn::AttentionResNet(p, state, input, cfg, true);
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite CBAM-ResNet derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  check::Require(minimum > 1e-5 * maximum, "clearly nonzero CBAM-ResNet derivatives");
  std::cout << "  CBAM-ResNet-50: " << p.count()
            << " parameters; training-mode gradient norms with shifted BN betas [" << minimum << ','
            << maximum << "] (smallest " << smallest << ")" << std::endl;
}

} // namespace cbam_checks

namespace non_local_checks {

namespace nn = datamunge::nn;
using mobile_checks::EndsWith;
using mobile_checks::Pattern;
using non_local::Pairwise;

using Matrix = std::vector<std::vector<double>>;
// Rows times a [from, to] kernel plus bias.
inline Matrix Project(
    const Matrix&             rows,
    const std::vector<float>& w,
    const std::vector<float>& b,
    int                       from,
    int                       to) {
  Matrix out(rows.size(), std::vector<double>(to));
  for (std::size_t r = 0; r < rows.size(); ++r)
    for (int j = 0; j < to; ++j) {
      double v = b[j];
      for (int i = 0; i < from; ++i)
        v += rows[r][i] * w[i * to + j];
      out[r][j] = v;
    }
  return out;
}

inline void BlockChecks() {
  const int   size = 6, channels = 8, inner = 4;
  const char* labels[4] = {"embedded Gaussian", "Gaussian", "dot product", "concatenation"};
  std::string summary;
  for (int mode = 0; mode < 4; ++mode)
    for (bool subsample : {false, true}) {
      if (subsample && mode != 0)
        continue;
      const auto         f = Pairwise(mode);
      nn::ParameterStore p(17), state(17);
      nn::Tensor         x        = Pattern({1, size, size, channels}, 1, .2f);
      const auto         value    = x.to_vector<float>();
      double             identity = 0;
      {
        const auto fresh = non_local::NonLocalBlock(p, state, "nl", x, f, subsample, false)
                               .output.to_vector<float>();
        for (std::size_t i = 0; i < value.size(); ++i)
          identity = std::max(identity, double(std::abs(fresh[i] - value[i])));
      }
      check::Near(identity, 0, 0, "a new non-local block is the identity");
      {
        nn::NoGradGuard guard;
        for (const auto& name : p.names()) {
          nn::Tensor values = Pattern(
              p.at(name).shape(),
              name.find("/W") != std::string::npos && name.find("concat") == std::string::npos
                  ? .4f
                  : .3f,
              float(name.size()));
          if (name == "nl/bn/gamma")
            values = values + 1.0;
          p.set(name, values);
        }
        for (const auto& name : state.names()) {
          nn::Tensor values = Pattern(state.at(name).shape(), .2f, float(name.size()));
          if (name.find("moving_var") != std::string::npos)
            values = values + 1.0;
          state.set(name, values);
        }
      }
      x.requires_grad_(true);
      const auto block = non_local::NonLocalBlock(p, state, "nl", x, f, subsample, false);
      const auto support =
          nn::grad(nn::sum(nn::slice(block.output, {0, 0, 0, 0}, {1, 1, 1, channels})), {x})[0]
              .to_vector<float>();
      const auto                                output = block.output.to_vector<float>();
      std::map<std::string, std::vector<float>> v;
      for (const auto& name : p.names())
        v[name] = p.at(name).to_vector<float>();
      const auto mean = state.at("nl/bn/moving_mean").to_vector<float>(),
                 var  = state.at("nl/bn/moving_var").to_vector<float>();
      // Host reference.
      Matrix rows(size * size, std::vector<double>(channels)), pooled;
      for (int i = 0; i < size * size; ++i)
        for (int c = 0; c < channels; ++c)
          rows[i][c] = value[i * channels + c];
      if (subsample) {
        for (int r = 0; r < size; r += 2)
          for (int c = 0; c < size; c += 2) {
            std::vector<double> m(channels, -std::numeric_limits<double>::infinity());
            for (int dr = 0; dr < 2; ++dr)
              for (int dc = 0; dc < 2; ++dc)
                for (int ch = 0; ch < channels; ++ch)
                  m[ch] = std::max(m[ch], rows[(r + dr) * size + c + dc][ch]);
            pooled.push_back(m);
          }
      } else
        pooled = rows;
      const auto g = Project(pooled, v["nl/g/W"], v["nl/g/b"], channels, inner);
      Matrix     a(rows.size(), std::vector<double>(pooled.size()));
      if (f == Pairwise::kGaussian) {
        for (std::size_t i = 0; i < rows.size(); ++i)
          for (std::size_t j = 0; j < pooled.size(); ++j)
            for (int c = 0; c < channels; ++c)
              a[i][j] += rows[i][c] * pooled[j][c];
      } else {
        const auto theta = Project(rows, v["nl/theta/W"], v["nl/theta/b"], channels, inner),
                   phi   = Project(pooled, v["nl/phi/W"], v["nl/phi/b"], channels, inner);
        for (std::size_t i = 0; i < rows.size(); ++i)
          for (std::size_t j = 0; j < pooled.size(); ++j) {
            if (f == Pairwise::kConcatenation) {
              double z = 0;
              for (int k = 0; k < inner; ++k)
                z += theta[i][k] * v["nl/concat/W1"][k] + phi[j][k] * v["nl/concat/W2"][k];
              a[i][j] = std::max(z, 0.);
            } else {
              for (int k = 0; k < inner; ++k)
                a[i][j] += theta[i][k] * phi[j][k];
              if (f == Pairwise::kEmbeddedGaussian)
                a[i][j] /= std::sqrt(double(inner));
            }
          }
      }
      for (auto& row : a) {
        if (f == Pairwise::kEmbeddedGaussian || f == Pairwise::kGaussian) {
          const double top   = *std::max_element(row.begin(), row.end());
          double       total = 0;
          for (auto& e : row) {
            e = std::exp(e - top);
            total += e;
          }
          for (auto& e : row)
            e /= total;
        } else
          for (auto& e : row)
            e /= row.size();
      }
      Matrix y(rows.size(), std::vector<double>(inner, 0));
      for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < pooled.size(); ++j)
          for (int k = 0; k < inner; ++k)
            y[i][k] += a[i][j] * g[j][k];
      const auto w     = Project(y, v["nl/W/W"], v["nl/W/b"], inner, channels);
      double     error = 0;
      for (std::size_t i = 0; i < rows.size(); ++i)
        for (int c = 0; c < channels; ++c) {
          const double bn = (w[i][c] - mean[c]) / std::sqrt(var[c] + 1e-5) * v["nl/bn/gamma"][c] +
                            v["nl/bn/beta"][c];
          error = std::max(error, std::abs(output[i * channels + c] - (bn + rows[i][c])));
        }
      int reached = 0;
      for (int i = 0; i < size * size; ++i) {
        double m = 0;
        for (int c = 0; c < channels; ++c)
          m += std::abs(support[i * channels + c]);
        reached += m > 0;
      }
      check::Near(error, 0, 2e-5, std::string("non-local block: ") + labels[mode]);
      // Max-pooling passes derivatives only to window maxima; a concatenation
      // pair whose ReLU is off contributes none.
      if (!subsample && f != Pairwise::kConcatenation)
        check::Require(
            reached == size * size, "one output position depends on every input position");
      std::ostringstream e;
      e << error;
      summary += std::string("\n    ") + labels[mode] +
                 (subsample ? ", subsampled (36 x 9 affinities)" : " (36 x 36 affinities)") +
                 ": error " + e.str() + ", output 0 depends on " + std::to_string(reached) +
                 " of 36 positions";
    }
  std::cout << "  non-local blocks on 6x6x8 (identity when new; a 3x3 convolution would reach 9 "
               "positions):"
            << summary << std::endl;
}

/// tflow's cfg.non_local: embedded-Gaussian blocks after alternate blocks of res3 and res4.
inline attention_cnn::AfterBlockHook
NonLocalHook(double momentum, bool update_stats, bool subsample = true) {
  return [=](nn::ParameterStore& p,
             nn::ParameterStore& state,
             const std::string&  name,
             const nn::Tensor&   h,
             int                 stage,
             int                 block,
             bool                training) {
    if ((stage == 1 || stage == 2) && block % 2 == 0)
      return non_local::NonLocalBlock(
                 p,
                 state,
                 name + "/non_local",
                 h,
                 Pairwise::kEmbeddedGaussian,
                 subsample,
                 training,
                 momentum,
                 update_stats)
          .output;
    return h;
  };
}

inline void FullChecks() {
  nn::ParameterStore                   p(17), state(17);
  attention_cnn::AttentionResNetConfig cfg;
  cfg.update_stats = false;
  cfg.after_block  = NonLocalHook(cfg.momentum, false);
  {
    nn::NoGradGuard guard;
    attention_cnn::AttentionResNet(
        p, state, nn::zeros({1, 64, 64, 3}), cfg); // create the parameters
  }
  check::Require(p.count() == 32915496, "ResNet-50 with five non-local blocks");
  {
    // Activate the blocks (BN scale one) so their inner layers receive derivatives.
    nn::NoGradGuard guard;
    for (const auto& name : p.names()) {
      if (name.find("non_local/bn/gamma") != std::string::npos)
        p.set(name, nn::ones(p.at(name).shape()));
      else if (EndsWith(name, "/beta"))
        p.set(name, Pattern(p.at(name).shape(), .5f, float(name.size())));
    }
  }
  const auto net =
      attention_cnn::AttentionResNet(p, state, Pattern({2, 224, 224, 3}, 1, 0), cfg, true);
  const auto norms = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  // Embedded Gaussian: phi's bias shifts every logit of a softmax row by
  // theta_i . b, g's bias passes through rows summing to one, and W_z's bias is
  // per-channel; the training-mode BN removes all three exactly.
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0, removed = 0;
  int         removed_count = 0;
  std::string smallest;
  for (double norm : norms)
    maximum = std::max(maximum, norm);
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const auto& name = p.names()[i];
    check::Require(std::isfinite(norms[i]), "finite non-local ResNet derivative");
    if (EndsWith(name, "non_local/phi/b") || EndsWith(name, "non_local/g/b") ||
        EndsWith(name, "non_local/W/b")) {
      removed = std::max(removed, norms[i]);
      ++removed_count;
    } else if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = name;
    }
  }
  check::Require(
      removed_count == 15 && removed < 1e-7 * maximum, "vanishing non-local bias derivatives");
  check::Require(minimum > 1e-6 * maximum, "clearly nonzero non-local ResNet derivatives");
  std::cout << "  ResNet-50 + 5 non-local blocks: " << p.count()
            << " parameters; training-mode gradient norms with active blocks [" << minimum << ','
            << maximum << "] (smallest " << smallest
            << "); the 15 phi, g and W_z bias derivatives vanish (largest " << removed << ")"
            << std::endl;
}

// Install the concrete non-local architecture hook without host-language callbacks.
inline void UseNonLocal(
    attention_cnn::AttentionResNetConfig& cfg,
    bool                                  update_stats = true,
    bool                                  subsample    = true) {
  cfg.after_block = NonLocalHook(cfg.momentum, update_stats, subsample);
}

} // namespace non_local_checks

namespace octconv_checks {

namespace nn = datamunge::nn;
using mobile_checks::Pattern;

using Image = std::vector<std::vector<std::vector<double>>>; // [row][column][channel]
inline Image Convolve(const Image& x, const std::vector<float>& w, int from, int to) {
  const int size = int(x.size());
  Image     y(size, std::vector<std::vector<double>>(size, std::vector<double>(to, 0)));
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c)
      for (int dr = -1; dr <= 1; ++dr)
        for (int dc = -1; dc <= 1; ++dc) {
          const int rr = r + dr, cc = c + dc;
          if (rr < 0 || rr >= size || cc < 0 || cc >= size)
            continue;
          for (int i = 0; i < from; ++i)
            for (int o = 0; o < to; ++o)
              y[r][c][o] += x[rr][cc][i] * w[(((dr + 1) * 3 + dc + 1) * from + i) * to + o];
        }
  return y;
}
inline Image Pool(const Image& x) {
  const int size = int(x.size()) / 2, channels = int(x[0][0].size());
  Image     y(size, std::vector<std::vector<double>>(size, std::vector<double>(channels, 0)));
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c)
      for (int ch = 0; ch < channels; ++ch)
        for (int dr = 0; dr < 2; ++dr)
          for (int dc = 0; dc < 2; ++dc)
            y[r][c][ch] += x[2 * r + dr][2 * c + dc][ch] / 4;
  return y;
}

inline void LayerChecks() {
  const int    size = 8, channels = 16;
  const double alpha = .25;
  const int    low = octconv::LowChannels(channels, alpha), high = channels - low;
  {
    // alpha 0: an ordinary convolution with the same kernel.
    nn::ParameterStore  p(17);
    const nn::Tensor    x = Pattern({1, size, size, channels}, 1, .2f);
    octconv::OctaveMaps in;
    in.high             = x;
    in.high_channels    = channels;
    const auto        y = octconv::OctaveConv(p, "oct", in, channels, 0, 3, 1, size);
    nn::Conv2DOptions o = nn::conv_options(1, nn::Padding::Explicit);
    o.explicit_paddings = {1, 1, 1, 1};
    const auto a        = y.high.to_vector<float>(),
               b        = nn::conv2d(x, p.at("oct/hh/W"), o).to_vector<float>();
    double error        = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
      error = std::max(error, double(std::abs(a[i] - b[i])));
    check::Require(!y.low.defined() && y.low_channels == 0, "alpha 0 has no low-frequency output");
    check::Near(error, 0, 0, "alpha 0 octave convolution is an ordinary convolution");
  }
  nn::ParameterStore p(17);
  const nn::Tensor   vh = Pattern({1, size, size, high}, 1, .2f),
                   vl   = Pattern({1, size / 2, size / 2, low}, 1, .9f);
  octconv::OctaveMaps in{vh, vl, high, low};
  octconv::OctaveCost cost;
  const auto          y  = octconv::OctaveConv(p, "oct", in, channels, alpha, 3, 1, size, &cost);
  const auto          yh = y.high.to_vector<float>(), yl = y.low.to_vector<float>(),
             h = vh.to_vector<float>(), l = vl.to_vector<float>();
  Image H(size, std::vector<std::vector<double>>(size, std::vector<double>(high))),
      L(size / 2, std::vector<std::vector<double>>(size / 2, std::vector<double>(low)));
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c)
      for (int ch = 0; ch < high; ++ch)
        H[r][c][ch] = h[(r * size + c) * high + ch];
  for (int r = 0; r < size / 2; ++r)
    for (int c = 0; c < size / 2; ++c)
      for (int ch = 0; ch < low; ++ch)
        L[r][c][ch] = l[(r * size / 2 + c) * low + ch];
  const auto hh = Convolve(H, p.at("oct/hh/W").to_vector<float>(), high, high),
             lh = Convolve(L, p.at("oct/lh/W").to_vector<float>(), low, high),
             hl = Convolve(Pool(H), p.at("oct/hl/W").to_vector<float>(), high, low),
             ll = Convolve(L, p.at("oct/ll/W").to_vector<float>(), low, low);
  double error  = 0;
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c)
      for (int o = 0; o < high; ++o)
        error = std::max(
            error, std::abs(yh[(r * size + c) * high + o] - (hh[r][c][o] + lh[r / 2][c / 2][o])));
  for (int r = 0; r < size / 2; ++r)
    for (int c = 0; c < size / 2; ++c)
      for (int o = 0; o < low; ++o)
        error = std::max(
            error, std::abs(yl[(r * size / 2 + c) * low + o] - (hl[r][c][o] + ll[r][c][o])));
  const std::int64_t plain = std::int64_t(size) * size * 9 * channels * channels;
  const double       ratio = double(cost.macs) / plain, formula = 1 - .75 * alpha * (2 - alpha);
  check::Near(error, 0, 1e-5, "octave convolution paths");
  check::Require(
      p.count() == 9 * channels * channels, "octave kernels partition an ordinary kernel");
  check::Near(ratio, formula, 1e-12, "octave multiply-add ratio");
  std::cout << "  octave convolution 16->16, alpha 0.25, 8x8 (12 high / 4 low channels): paths "
               "match the host (error "
            << error << "), " << p.count() << " weights as an ordinary 3x3, multiply-adds x"
            << ratio << " = 1 - 3/4 alpha (2 - alpha); alpha 0 equals Conv2D" << std::endl;
}

inline void CostChecks() {
  for (double alpha : {0., .125, .25, .5, .75}) {
    nn::ParameterStore       p(17), state(17);
    octconv::OctResNetConfig cfg;
    cfg.alpha = alpha;
    std::int64_t macs;
    {
      nn::NoGradGuard guard;
      macs = octconv::OctResNet(p, state, nn::zeros({1, 64, 64, 3}), cfg)
                 .macs; // MACs at cfg.image_size = 224
    }
    check::Require(
        p.count() == 25557032 && (alpha > 0 || macs == 4089184256),
        "Oct-ResNet-50 parameters (and alpha 0 multiply-adds)");
    std::cout << "  Oct-ResNet-50, alpha " << alpha << ": " << p.count() << " parameters, "
              << std::fixed << std::setprecision(3) << macs / 1e9 << "G mult-adds ("
              << std::setprecision(1) << 100. * macs / 4089184256 << "% of ResNet-50)"
              << std::defaultfloat << std::setprecision(6) << std::endl;
  }
}

inline void GradientChecks() {
  nn::ParameterStore       p(17), state(17);
  octconv::OctResNetConfig cfg;
  cfg.classes         = 10;
  cfg.width_divisor   = 4;
  cfg.small_stem      = true;
  cfg.image_size      = 32;
  cfg.update_stats    = false;
  const auto  net     = octconv::OctResNet(p, state, Pattern({2, 32, 32, 3}, 1, 0), cfg, true);
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite Oct-ResNet derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  check::Require(minimum > 1e-6 * maximum, "clearly nonzero Oct-ResNet derivatives");
  std::cout << "  compact Oct-ResNet-50 (alpha 0.25): " << p.count() << " parameters, "
            << net.macs / 1e6 << "M mult-adds at 32 px; derivatives in [" << minimum << ','
            << maximum << "] (smallest " << smallest << ")" << std::endl;
}

} // namespace octconv_checks

namespace repvgg_checks {

namespace nn = datamunge::nn;
using mobile_checks::Pattern;

// Nontrivial BN parameters and statistics, as after training.
inline void Perturb(nn::ParameterStore& p, nn::ParameterStore& state) {
  nn::NoGradGuard guard;
  for (const auto& name : p.names())
    if (name.find("/bn/") != std::string::npos) {
      nn::Tensor v = Pattern(p.at(name).shape(), .3f, float(name.size()));
      if (name.find("gamma") != std::string::npos)
        v = v + 1.0;
      p.set(name, v);
    }
  for (const auto& name : state.names()) {
    nn::Tensor v = Pattern(state.at(name).shape(), .2f, float(name.size()));
    if (name.find("moving_var") != std::string::npos)
      v = v + 1.0;
    state.set(name, v);
  }
}

inline double MaxDifference(const nn::Tensor& a, const nn::Tensor& b) {
  const auto x = a.to_vector<float>(), y = b.to_vector<float>();
  double     error = 0, scale = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    error = std::max(error, double(std::abs(x[i] - y[i])));
    scale = std::max(scale, double(std::abs(y[i])));
  }
  return error / std::max(1., scale);
}

inline void BlockChecks() {
  const std::vector<repvgg::RepVGGBlockSpec> specs = {
      {"identity", 8, 8, 1, 1, true},
      {"strided", 8, 12, 2, 1, false},
      {"grouped", 8, 8, 1, 2, true}};
  nn::NoGradGuard guard;
  for (const auto& spec : specs) {
    nn::ParameterStore p(17), state(17);
    const nn::Tensor   value = Pattern({2, 7, 7, 8}, 1, .3f);
    repvgg::RepVGGBlock(p, state, spec, value, false, false); // create the parameters
    Perturb(p, state);
    const nn::Tensor   trained = repvgg::RepVGGBlock(p, state, spec, value, false, false);
    nn::ParameterStore dp(17), ds(17);
    repvgg::RepVGGBlock(dp, ds, spec, value, true, false);
    repvgg::DeployRepVGG(dp, {spec}, p, state);
    const nn::Tensor fused = repvgg::RepVGGBlock(dp, ds, spec, value, true, false);
    const double     error = MaxDifference(fused, trained);
    check::Near(error, 0, 1e-5, "fused block equals training block");
    std::cout << "  " << spec.name << " block (" << spec.in << "->" << spec.out << ", stride "
              << spec.stride << ", " << spec.groups << (spec.groups > 1 ? " groups" : " group")
              << (spec.identity ? ", identity branch" : "") << "): fused 3x3 error " << error
              << std::endl;
  }
}

inline void NetworkChecks() {
  nn::NoGradGuard    guard;
  nn::ParameterStore p(17), state(17);
  const nn::Tensor   x = Pattern({1, 64, 64, 3}, 1, .1f);
  repvgg::RepVGG(p, state, x, {});
  check::Require(p.count() == 9108968 && state.count() == 23648, "RepVGG-A0 training parameters");
  Perturb(p, state);
  const auto           net = repvgg::RepVGG(p, state, x, {});
  nn::ParameterStore   dp(17), ds(17);
  repvgg::RepVGGConfig cfg;
  cfg.deploy = true;
  repvgg::RepVGG(dp, ds, x, cfg);
  check::Require(dp.count() == 8309384 && ds.count() == 0, "RepVGG-A0 deployed parameters");
  repvgg::DeployRepVGG(dp, net.blocks, p, state);
  const auto   plain = repvgg::RepVGG(dp, ds, x, cfg);
  const double error = MaxDifference(plain.logits, net.logits);
  check::Near(error, 0, 1e-4, "deployed RepVGG-A0 reproduces the training network");
  std::cout << "  RepVGG-A0: " << net.blocks.size() << " blocks fused; deployed plain 3x3 stack ("
            << dp.count() << " parameters, training form " << p.count() << " + " << state.count()
            << " BN statistics) reproduces the logits (relative error " << error << ")"
            << std::endl;
}

inline void CostChecks() {
  struct Case {
    const char*  name;
    std::int64_t train, deploy, macs;
  };
  const std::vector<Case> cases = {
      {"A0", 9108968, 8309384, 1361451008},
      {"A1", 14092264, 12789864, 2363967488},
      {"A2", 28210600, 25499944, 5116951552},
      {"B0", 15817960, 14339048, 3057600512},
      {"B1", 57415016, 51829480, 11815485440},
      {"B1g2", 45782376, 41360104, 8809742336},
      {"B1g4", 39966056, 36125416, 7306870784},
      {"B2", 89022376, 80315112, 18376609792},
      {"B2g4", 61758376, 55777512, 11331899392},
      {"B3", 123085288, 110960872, 26208882688}};
  nn::NoGradGuard guard;
  for (const auto& c : cases) {
    nn::ParameterStore   p(17), state(17);
    repvgg::RepVGGConfig cfg;
    cfg.model = c.name;
    repvgg::RepVGG(p, state, nn::zeros({1, 32, 32, 3}), cfg);
    nn::ParameterStore dp(17), ds(17);
    cfg.deploy         = true;
    const auto   plain = repvgg::RepVGG(dp, ds, nn::zeros({1, 224, 224, 3}), cfg);
    std::int64_t macs  = std::int64_t(plain.blocks.back().out) * 1000;
    for (std::size_t i = 0; i < plain.blocks.size(); ++i) {
      const auto&        b = plain.blocks[i];
      const std::int64_t n = plain.outputs[i].shape()[1];
      macs += n * n * 9 * b.in * b.out / b.groups;
    }
    check::Require(
        p.count() == c.train && dp.count() == c.deploy && macs == c.macs,
        "RepVGG parameters and multiply-adds");
    std::cout << "  RepVGG-" << c.name << ": " << std::fixed << std::setprecision(2)
              << p.count() / 1e6 << "M training / " << dp.count() / 1e6 << "M deployed parameters, "
              << macs / 1e9 << "G mult-adds" << std::defaultfloat << std::setprecision(6)
              << std::endl;
  }
}

inline void GradientChecks() {
  nn::ParameterStore   p(17), state(17);
  repvgg::RepVGGConfig cfg;
  cfg.update_stats    = false;
  const auto  net     = repvgg::RepVGG(p, state, Pattern({2, 224, 224, 3}, 1, 0), cfg, true);
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite RepVGG derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  check::Require(minimum > 1e-6 * maximum, "clearly nonzero RepVGG derivatives");
  std::cout << "  RepVGG-A0 training form: " << p.names().size() << " derivatives in [" << minimum
            << ',' << maximum << "] (smallest " << smallest << ")" << std::endl;
}

} // namespace repvgg_checks

namespace sknet_checks {

namespace nn = datamunge::nn;
using mobile_checks::Pattern;

inline void UnitChecks() {
  for (int stride : {1, 2}) {
    const int channels = 8, size = 9, out_size = (size + stride - 1) / stride,
              centre = out_size / 2;
    nn::ParameterStore p(17), state(17);
    nn::Tensor         x = Pattern({1, size, size, channels}, 1, .2f);
    x.requires_grad_(true);
    const auto sk =
        attention_cnn::SelectiveKernel(p, state, "sk", x, channels, 2, stride, 2, 4, false, .9);
    std::vector<std::vector<float>> supports;
    for (const auto& branch : sk.branches) {
      // All channels: ReLU may silence one.
      const nn::Tensor picked = nn::slice(branch, {0, centre, centre, 0}, {1, 1, 1, channels});
      supports.push_back(
          nn::grad(nn::sum(picked), {x}, nn::GradOptions{true, false, false})[0]
              .to_vector<float>());
    }
    const auto out = sk.output.to_vector<float>(), b0 = sk.branches[0].to_vector<float>(),
               b1 = sk.branches[1].to_vector<float>(), w0 = sk.weights[0].to_vector<float>(),
               w1 = sk.weights[1].to_vector<float>();
    double error = 0, sum = 0, low = 1;
    for (std::size_t i = 0; i < out.size(); ++i) {
      const std::size_t c = i % channels;
      const float       a = w0[c], b = w1[c];
      error = std::max(error, double(std::abs(out[i] - (a * b0[i] + b * b1[i]))));
      sum   = std::max(sum, std::abs(a + b - 1.));
      low   = std::min({low, double(a), double(b)});
    }
    check::Require(
        sk.output.shape()[1] == out_size && low > 0, "SK output size and positive branch weights");
    check::Near(sum, 0, 1e-6, "SK branch weights sum to one");
    check::Near(error, 0, 1e-6, "SK output is the attention-weighted branch sum");
    std::string text;
    for (int branch = 0; branch < 2; ++branch) {
      int taps = 0, low_r = size, high_r = -1;
      for (int r = 0; r < size; ++r)
        for (int c = 0; c < size; ++c) {
          double mass = 0;
          for (int ch = 0; ch < channels; ++ch)
            mass += std::abs(supports[branch][(r * size + c) * channels + ch]);
          if (mass > 0) {
            ++taps;
            low_r  = std::min(low_r, r);
            high_r = std::max(high_r, r);
          }
        }
      const int span = high_r - low_r + 1;
      check::Require(
          taps == 9 && span == (branch ? 5 : 3) && (low_r + high_r) / 2 == centre * stride,
          "SK branch receptive fields");
      text += (branch ? ", dilated " : "plain ") + std::to_string(taps) + " taps over " +
              std::to_string(span) + "x" + std::to_string(span);
    }
    std::cout << "  SK unit, stride " << stride
              << ": weights positive and summing to one, output = weighted branch sum (error "
              << error << "); centre-output supports: " << text << std::endl;
  }
}

inline void FullChecks() {
  nn::ParameterStore                   p(17), state(17);
  attention_cnn::AttentionResNetConfig cfg;
  cfg.cardinality  = 32;
  cfg.base_width   = 4;
  cfg.update_stats = false;
  cfg.conv2 =
      attention_cnn::SelectiveKernelHook(cfg.cardinality, cfg.reduction, cfg.momentum, false);
  {
    nn::NoGradGuard guard;
    attention_cnn::AttentionResNet(
        p, state, nn::zeros({1, 64, 64, 3}), cfg); // create the parameters
  }
  check::Require(p.count() == 27479784, "SKNet-50 parameter count");
  mobile_checks::ShiftBetas(p, false);
  const auto net =
      attention_cnn::AttentionResNet(p, state, Pattern({2, 224, 224, 3}, 1, 0), cfg, true);
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite SKNet derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  // The stem's derivative sums over 112x112 positions while a late SK unit's
  // fc sees one pooled vector, so the norms span about seven decades here.
  check::Require(minimum > 1e-7 * maximum, "clearly nonzero SKNet derivatives");
  std::cout << "  SKNet-50: " << p.count()
            << " parameters; training-mode gradient norms with shifted BN betas [" << minimum << ','
            << maximum << "] (smallest " << smallest << ")" << std::endl;
}

} // namespace sknet_checks
