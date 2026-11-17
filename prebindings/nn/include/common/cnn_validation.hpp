// Shared numerical/architecture diagnostics for the CNN examples.
// No fitting workflow lives here: each .cpp / Python / R script owns its
// data split, epoch/batch loop, objective, optimizer, checkpoint and report.
// These diagnostics are the same C++ checks called by all three languages.
#pragma once

#include <cctype>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>

#include "common/alexnet.hpp"
#include "common/architecture_checks.hpp"
#include "common/checks.hpp"
#include "common/convnext.hpp"
#include "common/densenet.hpp"
#include "common/ghostnet.hpp"
#include "common/image_training.hpp"
#include "common/mbconv.hpp"
#include "common/mobile_checks.hpp"
#include "common/mobilenet.hpp"
#include "common/xception.hpp"

namespace alexnet_checks {

namespace nn = datamunge::nn;

inline void BuildingBlockChecks() {
  nn::ParameterStore p(2);
  nn::Tensor         x = nn::ones({1, 5, 5, 4});
  x.requires_grad_(true);
  const auto grouped   = grouped::GroupedConv2D(p, "conv", x, 6, 3, 2);
  const auto first     = nn::sum(nn::slice(grouped, {0, 0, 0, 0}, {1, 5, 5, 3}));
  const auto dx        = nn::grad(first, {x})[0].to_vector<float>();
  double     connected = 0;
  for (int i = 0; i < 100; ++i) {
    if (i % 4 < 2)
      connected += std::abs(dx[static_cast<std::size_t>(i)]);
    else
      check::Near(
          dx[static_cast<std::size_t>(i)],
          0,
          0,
          "grouped convolution cannot read the other partition");
  }
  check::Require(connected > 0, "connected grouped inputs affect output");
  const auto normalized =
      alexnet::AlexNetLrn(nn::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6, 7}, {1, 1, 1, 7}))
          .to_vector<float>();
  for (int c = 0; c < 7; ++c) {
    double squares = 0;
    for (int j = std::max(0, c - 2); j <= std::min(6, c + 2); ++j)
      squares += (j + 1) * (j + 1);
    check::Near(
        normalized[static_cast<std::size_t>(c)],
        (c + 1) / std::pow(2 + .0001 * squares, .75),
        5e-7,
        "LRN formula and clipped channel window");
  }
  std::cout << "  grouped connectivity and paper LRN formula checks passed\n";
}

inline void FullArchitectureChecks() {
  nn::ParameterStore p(17);
  const auto         net = alexnet::AlexNet(p, architecture::SinePixels({1, 227, 227, 3}, .5, .13));
  check::Require(p.count() == 60965224, "full AlexNet parameter count");
  const std::vector<nn::Tensor> stages = {
      net.convolution[0],
      net.convolution[1],
      net.convolution[2],
      net.convolution[3],
      net.convolution[4],
      net.pooling[0],
      net.pooling[1],
      net.pooling[2],
      net.fc6,
      net.fc7,
      net.logits};
  const std::vector<nn::Shape> shapes = {
      {1, 55, 55, 96},
      {1, 27, 27, 256},
      {1, 13, 13, 384},
      {1, 13, 13, 384},
      {1, 13, 13, 256},
      {1, 27, 27, 96},
      {1, 13, 13, 256},
      {1, 6, 6, 256},
      {1, 4096},
      {1, 4096},
      {1, 1000}};
  for (std::size_t i = 0; i < shapes.size(); ++i)
    architecture::RequireShape(stages[i], shapes[i], "full AlexNet stage");
  const auto [minimum, maximum] = architecture::GradientNormRange(
      p, nn::mean(nn::square(net.logits)), "full AlexNet learned tensor");
  std::cout << "  full AlexNet: parameters=" << p.count()
            << "; conv spatial sizes 55/27/13/13/13; final pool 6x6\n"
            << "  dense widths 4096/4096/1000; parameter gradient norms [" << minimum << ','
            << maximum << "]\n";
}

inline alexnet::AlexNetConfig CompactConfig() {
  alexnet::AlexNetConfig cfg;
  cfg.image_size    = 67;
  cfg.classes       = 10;
  cfg.width_divisor = 4;
  cfg.dense_width   = 256;
  return cfg;
}

inline image_training::LogitsBuilder CompactArchitecture() {
  return [](nn::ParameterStore& p, nn::ParameterStore&, const nn::Tensor& x, bool training) {
    const auto image = nn::resize_bilinear(x, {67, 67}, false, true);
    return alexnet::AlexNet(p, image, CompactConfig(), training).logits;
  };
}

inline void DropoutChecks() {
  // Opposite training/inference paths make the update direction diagnostic.
  // At w=1, training CE plus L2 has a positive derivative; inference CE plus
  // L2 has a negative derivative. This catches accidentally replacing the
  // training objective with inference loss when adding regularization.
  {
    nn::ParameterStore probe(17), state(17);
    probe.set("w", nn::ones({1, 1}));
    const image_training::LogitsBuilder model =
        [](nn::ParameterStore& p, nn::ParameterStore&, const nn::Tensor& x, bool training) {
          const auto score = nn::matmul(x, p.at("w"));
          return training ? nn::concat({-score, score}, 1) : nn::concat({score, -score}, 1);
        };
    image_training::ImageTrainingConfig cfg;
    cfg.classes = 2;
    cfg.l2      = .1;
    nn::OptimizerConfig config;
    config.kind = nn::OptimizerKind::Adam;
    nn::Optimizer optimizer(probe, .01, config);
    const auto    x = nn::ones({1, 1}), y = nn::from_vector(std::vector<std::int64_t>{0}, {1});
    optimizer.minimize(image_training::TrainingObjective(probe, state, model, x, y, cfg));
    const double updated = probe.at("w").item();
    check::Require(updated < 1, "L2 preserves training-path objective and update direction");
    std::cout << "  L2 training-path regression: w=1 -> " << updated
              << " (inference loss would increase w)\n";
  }
  nn::ParameterStore p(17), state(17);
  const auto         model = CompactArchitecture();
  const auto         input = architecture::SinePixels({4, 32, 32, 3}, 1, .11);
  nn::NoGradGuard    guard;
  const auto         first     = model(p, state, input, false).to_vector<float>(),
             second            = model(p, state, input, false).to_vector<float>();
  const auto training          = model(p, state, input, true).to_vector<float>(),
             repeat            = model(p, state, input, true).to_vector<float>();
  double stochastic_difference = 0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    check::Near(first[i], second[i], 0, "AlexNet deterministic inference");
    stochastic_difference += std::abs(training[i] - repeat[i]);
  }
  check::Require(stochastic_difference > 1e-5, "training uses fresh dropout masks");
  std::cout << "  compact AlexNet parameters=" << p.count()
            << "; dropout training/inference checks passed\n";
}

} // namespace alexnet_checks

namespace xception_checks {

namespace nn = datamunge::nn;

inline void SeparableActivationChecks() {
  nn::ParameterStore              p(0), state(0);
  normalized_conv::ConvNormConfig cfg;
  const std::vector<float>        values = {-2, 1, 3, -4, 2, -1, 4, -3, 1};
  nn::Tensor                      input  = nn::from_vector(values, {1, 3, 3, 1});
  {
    nn::NoGradGuard guard;
    xception::XceptionSeparable(p, state, "layer", input, 1, cfg);
  }
  check::Require(
      p.count() == 12 && state.count() == 2, "separable+BN parameters and inference state");
  std::vector<float> kernel(9, 0.f);
  kernel[4] = 1;
  p.set("layer/depthwise/W", nn::from_vector(kernel, {3, 3, 1, 1}));
  p.set("layer/pointwise/W", nn::from_vector(std::vector<float>{-2}, {1, 1, 1, 1}));
  input.requires_grad_(true);
  const auto output    = xception::XceptionSeparable(p, state, "layer", input, 1, cfg);
  const auto gradient  = nn::grad(nn::sum(output), {input})[0].to_vector<float>();
  const auto depthwise = nn::depthwise_conv2d(input, p.at("layer/depthwise/W"));
  const auto intervening_relu =
      (nn::conv2d(nn::relu(depthwise), p.at("layer/pointwise/W")) * (1. / std::sqrt(1.001)))
          .to_vector<float>();
  const auto out        = output.to_vector<float>();
  double     difference = 0;
  for (int i = 0; i < 9; ++i) {
    const auto k = static_cast<std::size_t>(i);
    check::Near(
        out[k],
        -2 * values[k] / std::sqrt(1.001),
        2e-6,
        "no activation between depthwise and pointwise");
    check::Near(gradient[k], -2 / std::sqrt(1.001), 2e-6, "linear separable input derivative");
    check::Near(
        intervening_relu[k],
        -2 * std::max(0.f, values[k]) / std::sqrt(1.001),
        2e-6,
        "intervening ReLU reference");
    difference += std::abs(out[k] - intervening_relu[k]);
  }
  check::Require(difference > 10, "intervening ReLU changes the operation");
  std::cout << "  linear depthwise/pointwise output and derivative passed; inserting ReLU changes "
               "summed absolute outputs by "
            << difference << std::endl;
}

inline void FullArchitectureChecks() {
  nn::ParameterStore p(17), state(17);
  const auto         net =
      xception::Xception(p, state, architecture::SinePixels({1, 299, 299, 3}, .1, .13));
  check::Require(
      p.count() == 22855952 && state.count() == 54528, "canonical Xception learned/state counts");
  const auto                       initial_state = state.flatten();
  std::vector<std::pair<int, int>> expected      = {
      {149, 32}, {147, 64}, {74, 128}, {37, 256}, {19, 728}};
  expected.insert(expected.end(), 8, {19, 728});
  expected.emplace_back(10, 1024);
  expected.emplace_back(10, 1536);
  expected.emplace_back(10, 2048);
  check::Require(net.stages.size() == expected.size(), "all Xception stages");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto [spatial, channels] = expected[i];
    architecture::RequireShape(
        net.stages[i].second,
        {1, spatial, spatial, channels},
        "Xception reference shape: " + net.stages[i].first);
  }
  architecture::RequireShape(net.logits, {1, 1000}, "Xception classifier dimensions");
  const auto [minimum, maximum] =
      architecture::GradientNormRange(p, nn::mean(nn::square(net.logits)), "Xception parameter");
  check::Require(state.flatten() == initial_state, "frozen inference state");
  std::cout << "  full Xception: learned=" << p.count() << ", running state=" << state.count()
            << "; 16 stage shapes; gradient norms [" << minimum << ',' << maximum << "]"
            << std::endl;
}

} // namespace xception_checks

namespace densenet_checks {

namespace nn = datamunge::nn;

inline void FeatureReuseChecks() {
  for (bool bottleneck : {false, true}) {
    nn::ParameterStore p(17), state(17);
    std::vector<float> values(2 * 5 * 5 * 3);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = std::sin(.17f * static_cast<float>(i));
    nn::Tensor                                x = nn::from_vector(values, {2, 5, 5, 3});
    std::vector<densenet::DenseGrowthOutputs> outs;
    auto                                      build = [&](const nn::Tensor& input) {
      outs.clear();
      nn::Tensor h = input;
      for (int layer = 0; layer < 4; ++layer) {
        outs.push_back(
            densenet::DenseGrowthLayer(
                p, state, "layer" + std::to_string(layer), h, 2, bottleneck));
        h = outs.back().output;
      }
      return h;
    };
    {
      nn::NoGradGuard guard;
      build(x);
      for (int layer = 0; layer < 4; ++layer) {
        const int channels = 3 + 2 * layer;
        check::Require(
            outs[layer].output.shape() == nn::Shape({2, 5, 5, channels + 2}),
            "linear DenseNet channel growth");
        const auto out      = outs[layer].output.to_vector<float>(),
                   added    = outs[layer].added.to_vector<float>();
        const auto previous = layer == 0 ? values : outs[layer - 1].output.to_vector<float>();
        for (int position = 0; position < 50; ++position) {
          for (int c = 0; c < channels; ++c)
            check::Near(
                out[position * (channels + 2) + c],
                previous[position * channels + c],
                0,
                "all preceding feature maps retained verbatim");
          for (int c = 0; c < 2; ++c)
            check::Near(
                out[position * (channels + 2) + channels + c],
                added[position * 2 + c],
                0,
                "new features appended after existing channels");
        }
      }
      for (int layer = 0; layer < 4; ++layer) // zero appended features
        p.set(
            "layer" + std::to_string(layer) + "/conv2/W",
            nn::zeros({3, 3, bottleneck ? 8 : 3 + 2 * layer, 2}));
    }
    x.requires_grad_(true);
    const nn::Tensor h        = build(x);
    const auto       gradient = nn::grad(nn::sum(h), {x})[0].to_vector<float>();
    const auto       y        = h.to_vector<float>();
    for (int position = 0; position < 50; ++position)
      for (int c = 0; c < 11; ++c)
        check::Near(
            y[position * 11 + c],
            c < 3 ? values[position * 3 + c] : 0,
            0,
            "signed original features plus zero appended channels");
    for (float g : gradient)
      check::Near(g, 1, 2e-6, "unit direct input derivative through dense concatenations");
  }
  std::cout << "  basic/bottleneck feature ordering, linear growth and direct identity derivatives "
               "passed\n";
}

inline void TransitionCheck() {
  nn::ParameterStore p(17), state(17);
  std::vector<float> values(64);
  for (int i = 0; i < 64; ++i)
    values[i] = .1f + .02f * i;
  nn::Tensor x = nn::from_vector(values, {1, 4, 4, 4});
  {
    nn::NoGradGuard guard;
    densenet::DenseTransition(p, state, "transition", x, 2);
    std::vector<float> weights(8, 0.f);
    weights[0 * 2 + 0] = weights[1 * 2 + 1] = 1;
    p.set("transition/conv/W", nn::from_vector(weights, {1, 1, 4, 2}));
    p.set("transition/bn/gamma", nn::full({4}, std::sqrt(1.f + 1e-5f)));
  }
  x.requires_grad_(true);
  const nn::Tensor h        = densenet::DenseTransition(p, state, "transition", x, 2);
  const auto       gradient = nn::grad(nn::sum(h), {x})[0].to_vector<float>();
  check::Require(
      h.shape() == nn::Shape({1, 2, 2, 2}), "transition spatial and channel compression");
  const auto y = h.to_vector<float>();
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 2; ++c)
      for (int channel = 0; channel < 2; ++channel) {
        double expected = 0;
        for (int dr = 0; dr < 2; ++dr)
          for (int dc = 0; dc < 2; ++dc)
            expected += .25 * values[((2 * r + dr) * 4 + 2 * c + dc) * 4 + channel];
        check::Near(
            y[(r * 2 + c) * 2 + channel],
            expected,
            2e-6,
            "host transition projection and average pooling");
      }
  for (int i = 0; i < 64; ++i)
    check::Near(gradient[i], i % 4 < 2 ? .25 : 0, 2e-6, "host transition derivative");
  std::cout << "  compressed transition matches host values and pooling derivatives\n";
}

inline void FullChecks(bool cifar) {
  nn::ParameterStore       p(17), state(17);
  densenet::DenseNetConfig cfg;
  cfg.cifar_layout  = cifar;
  cfg.depth         = cifar ? 100 : 121;
  cfg.growth_rate   = cifar ? 12 : 32;
  cfg.classes       = cifar ? 10 : 1000;
  cfg.update_stats  = false;
  const int  pixels = cifar ? 32 : 224, initial_spatial = cifar ? 32 : 56;
  const auto channels = cifar ? std::vector<std::int64_t>{216, 300, 342}
                              : std::vector<std::int64_t>{256, 512, 1024, 1024};
  const auto net      = densenet::DenseNet(
      p, state, architecture::SinePixels({2, pixels, pixels, 3}, .1, .13), cfg, true);
  check::Require(
      p.count() == (cifar ? 769162 : 7978856) && state.count() == (cifar ? 23988 : 83648),
      "canonical DenseNet parameter/state counts");
  check::Require(
      net.layers.size() == (cifar ? 48u : 58u) && net.stages.size() == channels.size(),
      "canonical DenseNet layer counts");
  for (std::size_t i = 0; i < channels.size(); ++i) {
    const std::int64_t spatial = initial_spatial / (1 << i);
    check::Require(
        net.stages[i].shape() == nn::Shape({2, spatial, spatial, channels[i]}),
        "canonical dense block dimensions");
    if (i + 1 < channels.size())
      check::Require(
          net.transitions[i].shape() == nn::Shape({2, spatial / 2, spatial / 2, channels[i] / 2}),
          "canonical transition dimensions");
  }
  check::Require(
      net.logits.shape() == nn::Shape({2, cfg.classes}), "DenseNet classifier dimensions");
  const auto [minimum, maximum] =
      architecture::GradientNormRange(p, nn::mean(nn::square(net.logits)), "DenseNet");
  std::cout << "  DenseNet-BC-" << cfg.depth << " k=" << cfg.growth_rate
            << ": parameters=" << p.count() << ", state=" << state.count() << "; gradient norms ["
            << minimum << ',' << maximum << "]" << std::endl;
}

} // namespace densenet_checks

namespace ghostnet_checks {

namespace nn = datamunge::nn;
using mobile_checks::Pattern;

inline void ModuleChecks() {
  const int                           in = 160, out = 960, size = 7;
  nn::ParameterStore                  p(17), state(17);
  std::vector<mobilenet::MobileLayer> ghost_layers, plain_layers;
  mobilenet::MobileContext            ghost_context{p, state, ghost_layers, false},
      plain_context{p, state, plain_layers, false};
  nn::NoGradGuard  guard;
  const nn::Tensor x     = Pattern({1, size, size, in}, 1, .2f);
  const auto       ghost = ghostnet::GhostModule(ghost_context, "ghost", x, out, true);
  ghostnet::GhostConvBN(plain_context, "plain", x, out, 1, 1, false, true);
  // The ghost half recomputed from the intrinsic half with the module's own weights.
  const std::string cheap    = "ghost/cheap";
  const nn::Tensor  filtered = nn::depthwise_conv2d(
      ghostnet::PadSpatial(ghost.intrinsic, 1),
      p.at(cheap + "/W"),
      nn::conv_options(1, nn::Padding::Valid));
  const nn::Tensor normalized = (filtered - state.at(cheap + "/bn/moving_mean")) *
                                    (1.0 / nn::sqrt(state.at(cheap + "/bn/moving_var") + 1e-5)) *
                                    p.at(cheap + "/bn/gamma") +
                                p.at(cheap + "/bn/beta");
  const auto result   = ghost.output.to_vector<float>(),
             expected = nn::relu(normalized).to_vector<float>();
  double error        = 0;
  for (std::int64_t pixel = 0; pixel < size * size; ++pixel)
    for (int ch = 0; ch < out / 2; ++ch)
      error = std::max(
          error,
          double(std::abs(result[pixel * out + out / 2 + ch] - expected[pixel * out / 2 + ch])));
  std::int64_t ghost_macs = 0, plain_macs = 0, ghost_weights = 0, plain_weights = 0;
  for (const auto& l : ghost_layers) {
    ghost_macs += l.Macs();
    ghost_weights += p.at(l.name + "/W").numel();
  }
  for (const auto& l : plain_layers) {
    plain_macs += l.Macs();
    plain_weights += p.at(l.name + "/W").numel();
  }
  const int    sratio = 2, k = 1, d = 3;
  const double formula = double(sratio) * in * k * k / (in * k * k + (sratio - 1) * d * d);
  check::Near(error, 0, 1e-5, "ghost maps are the depthwise image of the intrinsic maps");
  check::Near(double(plain_macs) / ghost_macs, formula, 1e-12, "ghost module speed-up ratio");
  check::Near(
      double(plain_weights) / ghost_weights, formula, 1e-12, "ghost module compression ratio");
  std::cout << "  ghost module 160->960 at 7x7: speed-up " << double(plain_macs) / ghost_macs
            << " and compression " << double(plain_weights) / ghost_weights
            << " = s c k^2 / (c k^2 + (s-1) d^2) = " << formula
            << "; ghost half equals the depthwise image of the intrinsic half (error " << error
            << ")" << std::endl;
}

inline void CostChecks() {
  struct Case {
    double       width;
    std::int64_t params, state, macs;
  };
  for (const auto& c :
       {Case{.5, 2587292, 10984, 42328152},
        Case{1, 5182508, 21968, 141150512},
        Case{1.3, 7358820, 28536, 227003256}}) {
    nn::ParameterStore       p(17), state(17);
    nn::NoGradGuard          guard;
    ghostnet::GhostNetConfig cfg;
    cfg.width       = c.width;
    const auto net  = ghostnet::GhostNet(p, state, nn::zeros({1, 224, 224, 3}), cfg);
    const auto macs = net.Macs();
    check::Require(
        p.count() == c.params && state.count() == c.state && macs == c.macs,
        "GhostNet parameters and multiply-adds");
    std::cout << "  GhostNet " << c.width << "x: " << p.count() << " parameters, " << std::fixed
              << std::setprecision(1) << macs / 1e6 << "M mult-adds" << std::defaultfloat
              << std::setprecision(6) << std::endl;
  }
}

inline void GradientChecks() {
  nn::ParameterStore       p(17), state(17);
  ghostnet::GhostNetConfig cfg;
  cfg.update_stats   = false;
  const nn::Tensor x = Pattern({2, 224, 224, 3}, 1, 0);
  // BN shifts added to the stream: each block's linear cheap ghost maps and
  // projection shortcut. They vanish if every consumer is 1x1 conv + BN.
  const auto        blocks = ghostnet::GhostNetBlocks();
  std::vector<bool> identity;
  int               width = 16;
  for (const auto& b : blocks) {
    identity.push_back(b.stride == 1 && width == b.out);
    width = b.out;
  }
  std::set<std::string> removable;
  for (std::size_t j = 0; j < blocks.size(); ++j) {
    // Depthwise BN shifts that feed only a 1x1 convolution and BN: every
    // projection shortcut's, and a strided block's when no SE sits between.
    const auto block = "block" + std::to_string(j + 1);
    if (!identity[j])
      removable.insert(block + "/shortcut/depthwise/bn/beta");
    if (blocks[j].stride > 1 && blocks[j].se == 0)
      removable.insert(block + "/depthwise/bn/beta");
    bool only_pointwise = true;
    for (std::size_t k = j + 1; k < blocks.size(); ++k)
      if (!identity[k]) {
        only_pointwise = false; // a depthwise shortcut reads the stream
        break;
      }
    if (!only_pointwise)
      continue;
    removable.insert(block + "/ghost2/cheap/bn/beta");
    if (!identity[j])
      removable.insert(block + "/shortcut/pointwise/bn/beta");
  }
  {
    nn::NoGradGuard guard;
    ghostnet::GhostNet(p, state, x, cfg, true);
  }
  mobile_checks::ShiftBetas(p);
  const auto  net     = ghostnet::GhostNet(p, state, x, cfg, true);
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0, removed = 0;
  int         removed_count = 0;
  std::string smallest;
  for (double norm : norms)
    maximum = std::max(maximum, norm);
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const double norm = norms[i];
    const auto&  name = p.names()[i];
    check::Require(std::isfinite(norm), "finite GhostNet derivative");
    if (removable.count(name)) {
      removed = std::max(removed, norm);
      ++removed_count;
    } else if (norm < minimum) {
      minimum  = norm;
      smallest = name;
    }
  }
  check::Require(
      removed_count == int(removable.size()) && removed < 1e-6 * maximum,
      "vanishing BN shifts that only reach 1x1 convolutions");
  check::Require(
      minimum > 1e-5 * maximum, "clearly nonzero GhostNet derivatives with shifted BN and SE");
  std::cout << "  width 1 with shifted BN betas and SE biases: gradient norms [" << minimum << ','
            << maximum << "] (smallest " << smallest << "); " << removed_count
            << " BN shift derivatives vanish (largest " << removed << ")" << std::endl;
}

} // namespace ghostnet_checks

namespace mobilenet_v2_checks {

namespace nn = datamunge::nn;

using mobile_checks::Pattern;
inline void ManifoldChecks() {
  const int          points = 400, trials = 20;
  std::vector<float> spiral(2 * points);
  for (int i = 0; i < points; ++i) {
    const float angle = 3 * std::acos(-1.) * i / points, radius = .2f + angle;
    spiral[i]          = radius * std::cos(angle);
    spiral[points + i] = radius * std::sin(angle);
  }
  auto                            x = nn::from_vector(spiral, {2, points});
  std::mt19937                    random(17);
  std::normal_distribution<float> normal;
  std::vector<double>             errors;
  nn::NoGradGuard                 guard;
  for (int n : {2, 3, 5, 15, 30}) {
    double total = 0;
    for (int trial = 0; trial < trials; ++trial) {
      std::vector<float> values(n * 2);
      for (auto& v : values)
        v = normal(random);
      auto t      = nn::from_vector(values, {n, 2});
      auto z      = nn::matmul(t, x);
      auto active = nn::expand_dims(
          nn::transpose(
              nn::cast(nn::greater(z, nn::scalar(0, nn::DType::Float32)), nn::DType::Float32),
              {1, 0}),
          2);
      auto rows = nn::mul(nn::expand_dims(t, 0), active);
      auto rhs  = nn::expand_dims(nn::transpose(nn::relu(z), {1, 0}), 2);
      auto recovered =
          nn::transpose(nn::squeeze(nn::matrix_solve_ls(rows, rhs, 1e-6), {2}), {1, 0});
      total += nn::sqrt(nn::sum(nn::square(nn::sub(recovered, x)))).item() /
               nn::sqrt(nn::sum(nn::square(x))).item();
    }
    errors.push_back(total / trials);
  }
  check::Require(
      errors[0] > errors[1] && errors[1] > errors[2] && errors[2] > errors[3] && errors[0] > .3 &&
          errors[3] < 1e-3,
      "ReLU manifold recovery improves in higher dimensions");
  std::cout << "manifold relative errors:";
  for (auto e : errors)
    std::cout << ' ' << e;
  std::cout << std::endl;
}
inline void BlockChecks() {
  nn::ParameterStore                  p(17), state(17);
  std::vector<mobilenet::MobileLayer> layers;
  mobilenet::MobileContext            c{p, state, layers, false};
  auto                                x = Pattern({2, 8, 8, 4}, 1, .9f).requires_grad_();
  mobilenet::InvertedResidual(c, "kept", x, 4, 6, 1);
  for (const auto& part : {"gamma", "beta"})
    p.set(std::string("kept/project/bn/") + part, nn::zeros({4}));
  auto kept    = mobilenet::InvertedResidual(c, "kept", x, 4, 6, 1);
  auto strided = mobilenet::InvertedResidual(c, "strided", x, 6, 6, 2);
  auto weights = Pattern({2, 8, 8, 4}, 1, .2f);
  auto g       = nn::grad(nn::sum(nn::mul(kept.output, weights)), {x});
  check::Near(nn::max(nn::abs(nn::sub(kept.output, x))).item(), 0, 0, "zero projection identity");
  check::Near(nn::max(nn::abs(nn::sub(g[0], weights))).item(), 0, 0, "identity input derivatives");
  check::Require(nn::min(kept.output).item() < 0, "negative values survive bottleneck");
  check::Require(
      kept.expanded.size(3) == 24 && strided.filtered.shape() == nn::Shape({2, 4, 4, 24}) &&
          strided.output.shape() == nn::Shape({2, 4, 4, 6}),
      "inverted residual shapes");
}
inline void CostChecks() {
  struct Case {
    double alpha;
    int    resolution;
    long   params, state, macs;
  };
  for (const auto& c :
       {Case{1.4, 224, 6108776, 47936, 582195824},
        Case{1, 224, 3504872, 34112, 300774272},
        Case{.75, 224, 2636424, 26640, 209069792},
        Case{.5, 224, 1968680, 18544, 97131840},
        Case{.35, 224, 1677128, 14080, 59285808},
        Case{1, 192, 3504872, 34112, 221316608},
        Case{1, 160, 3504872, 34112, 154083200},
        Case{1, 128, 3504872, 34112, 99074048},
        Case{1, 96, 3504872, 34112, 56289152}}) {
    nn::NoGradGuard              guard;
    nn::ParameterStore           p(17), state(17);
    mobilenet::MobileNetV2Config cfg;
    cfg.alpha = c.alpha;
    auto net = mobilenet::MobileNetV2(p, state, nn::zeros({1, c.resolution, c.resolution, 3}), cfg);
    long macs = net.classifier_macs;
    for (const auto& l : net.layers)
      macs += l.Macs();
    check::Require(
        p.count() == c.params && state.count() == c.state && macs == c.macs,
        "MobileNet v2 parameters and multiply-adds");
    std::cout << "width " << c.alpha << " resolution " << c.resolution << " params " << p.count()
              << " MACs " << macs << std::endl;
  }
}
inline void GradientChecks() {
  nn::ParameterStore           p(17), state(17);
  mobilenet::MobileNetV2Config cfg;
  cfg.update_stats = false;
  const auto x     = Pattern({2, 224, 224, 3}, 1, 0);
  {
    nn::NoGradGuard guard;
    mobilenet::MobileNetV2(p, state, x, cfg, true);
    mobile_checks::ShiftBetas(p, false);
  }
  auto net = mobilenet::MobileNetV2(p, state, x, cfg, true);
  check::Require(
      net.features.shape() == nn::Shape({2, 7, 7, 1280}) &&
          net.logits.shape() == nn::Shape({2, 1000}),
      "MobileNet v2 output dimensions");
  auto   norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double maximum = *std::max_element(norms.begin(), norms.end()), minimum = INFINITY, removed = 0;
  int    count = 0;
  auto   names = p.names();
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite MobileNet v2 derivative");
    if (mobile_checks::EndsWith(names[i], "/project/bn/beta")) {
      removed = std::max(removed, norms[i]);
      ++count;
    } else
      minimum = std::min(minimum, norms[i]);
  }
  check::Require(count == 17 && removed < 1e-6 * maximum, "vanishing projection BN shifts");
  check::Require(minimum > 1e-5 * maximum, "nonzero MobileNet v2 derivatives with shifted BN");
  std::cout << "gradient range " << minimum << " to " << maximum << " removed shifts " << removed
            << std::endl;
}

} // namespace mobilenet_v2_checks

namespace mnasnet_checks {

namespace nn = datamunge::nn;
using mobile_checks::EndsWith;
using mobile_checks::Pattern;

inline void BlockStringChecks() {
  for (const char* variant : {"a1", "b1"}) {
    int blocks = 0, se = 0, skips = 0;
    for (const auto& b : mbconv::MnasNetBlocks(variant)) {
      check::Require(
          mbconv::DecodeBlock(mbconv::EncodeBlock(b)).repeats == b.repeats &&
              mbconv::EncodeBlock(mbconv::DecodeBlock(mbconv::EncodeBlock(b))) ==
                  mbconv::EncodeBlock(b),
          "block string round trip");
      blocks += b.repeats;
      se += b.se_ratio > 0 ? b.repeats : 0;
      for (int r = 0; r < b.repeats; ++r)
        skips += b.skip && (r > 0 || (b.input == b.output && b.stride == 1));
    }
    std::cout << "  MnasNet-" << variant << ": " << blocks << " blocks, " << se
              << " with squeeze-and-excitation, " << skips
              << " identity connections; third block string "
              << mbconv::EncodeBlock(mbconv::MnasNetBlocks(variant)[2]) << std::endl;
  }
  check::Require(
      mbconv::EncodeBlock(mbconv::DecodeBlock("r3_k5_s22_e6_i112_o160_se0.25")) ==
              "r3_k5_s22_e6_i112_o160_se0.25" &&
          mbconv::EncodeBlock(mbconv::DecodeBlock("r1_k3_s11_e6_i192_o320_noskip")) ==
              "r1_k3_s11_e6_i192_o320_noskip",
      "reference block strings");
}

inline void CostChecks() {
  struct Case {
    const char*  variant;
    double       multiplier;
    std::int64_t params, state, macs;
  };
  const std::vector<Case> cases = {
      {"a1", 1, 3887038, 33488, 311579072},
      {"b1", 1, 4383312, 37920, 314415872},
      {"b1", .5, 2218512, -1, -1},
      {"b1", .75, 3170208, -1, -1},
      {"b1", 1.3, 6282256, -1, -1}};
  for (const auto& c : cases) {
    nn::ParameterStore    p(17), state(17);
    nn::NoGradGuard       guard;
    mbconv::MnasNetConfig cfg;
    cfg.variant     = c.variant;
    cfg.multiplier  = c.multiplier;
    const auto net  = mbconv::MnasNet(p, state, nn::zeros({1, 224, 224, 3}), cfg);
    const auto macs = net.Macs();
    check::Require(
        p.count() == c.params && (c.state < 0 || state.count() == c.state) &&
            (c.macs < 0 || macs == c.macs),
        "MnasNet parameters and multiply-adds");
    std::cout << "  MnasNet-" << c.variant << " x" << c.multiplier << ": " << p.count()
              << " parameters, " << std::fixed << std::setprecision(1) << macs / 1e6
              << "M mult-adds" << std::defaultfloat << std::setprecision(6) << std::endl;
  }
}

inline void GradientChecks() {
  nn::ParameterStore    p(17), state(17);
  mbconv::MnasNetConfig cfg;
  cfg.update_stats   = false;
  const nn::Tensor x = Pattern({2, 224, 224, 3}, 1, 0);
  {
    nn::NoGradGuard guard;
    mbconv::MnasNet(p, state, x, cfg, true);
  }
  mobile_checks::ShiftBetas(p);
  const auto  net     = mbconv::MnasNet(p, state, x, cfg, true);
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0, removed = 0;
  int         removed_count = 0;
  std::string smallest;
  for (double norm : norms)
    maximum = std::max(maximum, norm);
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const double norm = norms[i];
    const auto&  name = p.names()[i];
    check::Require(std::isfinite(norm), "finite MnasNet derivative");
    if (EndsWith(name, "/project/bn/beta")) {
      removed = std::max(removed, norm);
      ++removed_count;
    } else if (norm < minimum) {
      minimum  = norm;
      smallest = name;
    }
  }
  check::Require(
      removed_count == 16 && removed < 1e-6 * maximum, "vanishing projection BN shift derivatives");
  check::Require(
      minimum > 1e-5 * maximum, "clearly nonzero MnasNet derivatives with shifted BN and SE");
  std::cout << "  MnasNet-a1 with shifted BN betas and SE biases: gradient norms [" << minimum
            << ',' << maximum << "] (smallest " << smallest
            << "); 16 projection BN shift derivatives vanish (largest " << removed << ")"
            << std::endl;
}

} // namespace mnasnet_checks

namespace convnext_checks {

namespace nn = datamunge::nn;
using mobile_checks::Pattern;

inline void BlockChecks() {
  const int          size = 11, dim = 8;
  nn::ParameterStore p(17);
  nn::Tensor         x = Pattern({1, size, size, dim}, 1, .2f);
  double             change;
  {
    nn::NoGradGuard guard;
    change =
        nn::max(nn::abs(convnext::ConvNeXtBlock(p, "block", x, false, false).output - x)).item();
    p.set("block/layer_scale", nn::ones({dim}));
  }
  x.requires_grad_(true);
  const auto branch = convnext::ConvNeXtBlock(p, "block", x, false, false).branch;
  const auto g =
      nn::grad(nn::sum(nn::slice(branch, {0, 5, 5, 0}, {1, 1, 1, dim})), {x})[0].to_vector<float>();
  int reached = 0, low = size, high = -1;
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c) {
      double m = 0;
      for (int ch = 0; ch < dim; ++ch)
        m += std::abs(g[(r * size + c) * dim + ch]);
      if (m > 0) {
        ++reached;
        low  = std::min(low, r);
        high = std::max(high, r);
      }
    }
  check::Require(change < 1e-4, "a new block is nearly the identity");
  check::Require(reached == 49 && high - low + 1 == 7, "7x7 receptive field per block");
  std::cout << "  ConvNeXt block (8 channels): a new block changes its input by at most " << change
            << "; one output position depends on " << reached << " inputs in a " << high - low + 1
            << "x" << high - low + 1 << " window" << std::endl;
}

inline void CostChecks() {
  struct Case {
    const char*  size;
    std::int64_t params, macs;
  };
  for (const auto& c :
       {Case{"tiny", 28589128, 4455531264},
        Case{"small", 50223688, 8683712256},
        Case{"base", 88591464, 15354729472},
        Case{"large", 197767336, 34361433600}}) {
    nn::ParameterStore       p(17);
    nn::NoGradGuard          guard;
    convnext::ConvNeXtConfig cfg;
    cfg.size        = c.size;
    const auto net  = convnext::ConvNeXt(p, nn::zeros({1, 224, 224, 3}), cfg);
    const auto macs = convnext::ConvNeXtMacs(net, convnext::ConvNeXtLayout(c.size), 1000, 4);
    check::Require(
        p.count() == c.params && macs == c.macs, "ConvNeXt parameters and multiply-adds");
    std::cout << "  ConvNeXt-" << char(std::toupper(c.size[0])) << ": " << p.count()
              << " parameters, " << std::fixed << std::setprecision(2) << macs / 1e9
              << "G mult-adds" << std::defaultfloat << std::setprecision(6) << std::endl;
  }
}

inline void GradientChecks() {
  nn::ParameterStore p(17);
  const nn::Tensor   input = Pattern({2, 224, 224, 3}, 1, 0);
  {
    nn::NoGradGuard guard;
    convnext::ConvNeXt(p, nn::zeros({1, 32, 32, 3})); // create the parameters
    for (const auto& name : p.names())
      if (name.find("layer_scale") != std::string::npos)
        p.set(name, nn::ones(p.at(name).shape()));
  }
  const auto  net     = convnext::ConvNeXt(p, input, {}, true);
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite ConvNeXt derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  check::Require(minimum > 1e-6 * maximum, "clearly nonzero ConvNeXt derivatives");
  std::cout << "  ConvNeXt-T, layer scales one: " << norms.size() << " derivatives in [" << minimum
            << ',' << maximum << "] (smallest " << smallest << ")" << std::endl;
}

} // namespace convnext_checks

namespace efficientnet_checks {

namespace nn = datamunge::nn;
using mobile_checks::EndsWith;
using mobile_checks::Pattern;

inline void CostChecks() {
  struct Case {
    std::int64_t params, state, macs;
    int          blocks, head;
  };
  const Case cases[8] = {
      {5288548, 42016, 385814752, 16, 1280},
      {7794184, 62048, 686802832, 23, 1280},
      {9109994, 67568, 993561872, 23, 1408},
      {12233232, 87296, 1827141392, 26, 1536},
      {19341616, 125200, 4393771024, 32, 1792},
      {30389784, 172736, 10266384752, 39, 2048},
      {43040704, 224432, 19067593536, 45, 2304},
      {66347960, 310720, 37745884192, 55, 2560}};
  for (int v = 0; v < 8; ++v) {
    const auto                 scale = mbconv::EfficientNetCoefficients(v);
    const int                  r     = scale.resolution;
    nn::ParameterStore         p(17), state(17);
    nn::NoGradGuard            guard;
    mbconv::EfficientNetConfig cfg;
    cfg.variant       = v;
    const auto net    = mbconv::EfficientNet(p, state, nn::zeros({1, r, r, 3}), cfg);
    int        blocks = 0;
    for (const auto& b : mbconv::EfficientNetBlocks(v))
      blocks += b.repeats;
    const auto& c        = cases[v];
    const auto  features = net.features.shape();
    check::Require(
        p.count() == c.params && state.count() == c.state && net.Macs() == c.macs &&
            blocks == c.blocks && features[3] == c.head,
        "EfficientNet scaling, parameters and multiply-adds");
    std::cout << "  B" << v << ": width " << scale.width << ", depth " << scale.depth << ", " << r
              << " px, dropout " << scale.dropout << "; " << blocks << " blocks, head " << c.head
              << ", features " << features[1] << "x" << features[2] << "; " << p.count()
              << " parameters, " << std::fixed << std::setprecision(2) << net.Macs() / 1e9
              << "B mult-adds" << std::defaultfloat << std::setprecision(6) << std::endl;
  }
}

inline void DropConnectChecks() {
  const nn::Tensor branch = Pattern({512, 2, 2, 3}, 1, .1f);
  nn::Generator    gen(77);
  const auto       out   = mbconv::DropConnect(branch, .8, gen).to_vector<float>();
  const auto       value = branch.to_vector<float>();
  int              kept  = 0;
  double           error = 0;
  for (int example = 0; example < 512; ++example) {
    bool keep = false;
    for (int j = 0; j < 12; ++j)
      keep = keep || out[example * 12 + j] != 0;
    kept += keep;
    for (int j = 0; j < 12; ++j)
      error = std::max(
          error,
          double(std::abs(out[example * 12 + j] - (keep ? value[example * 12 + j] / .8f : 0.f))));
  }
  check::Near(error, 0, 1e-6, "drop connect removes or rescales whole examples");
  check::Require(std::abs(kept / 512. - .8) < .06, "drop connect survival rate");
  std::cout << "  drop connect with survival 0.8: " << kept
            << " of 512 residual branches kept, each scaled by 1.25 exactly (error " << error << ")"
            << std::endl;
}

inline void GradientChecks() {
  nn::ParameterStore         p(17), state(17);
  mbconv::EfficientNetConfig cfg;
  cfg.drop_connect   = 0;
  cfg.update_stats   = false;
  const nn::Tensor x = Pattern({2, 224, 224, 3}, 1, 0);
  {
    nn::NoGradGuard guard;
    mbconv::EfficientNet(p, state, x, cfg, true);
  }
  mobile_checks::ShiftBetas(p);
  const auto  net     = mbconv::EfficientNet(p, state, x, cfg, true);
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0, removed = 0;
  int         removed_count = 0;
  std::string smallest;
  for (double norm : norms)
    maximum = std::max(maximum, norm);
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const double norm = norms[i];
    const auto&  name = p.names()[i];
    check::Require(std::isfinite(norm), "finite EfficientNet derivative");
    if (EndsWith(name, "/project/bn/beta")) {
      removed = std::max(removed, norm);
      ++removed_count;
    } else if (norm < minimum) {
      minimum  = norm;
      smallest = name;
    }
  }
  check::Require(
      removed_count == 16 && removed < 1e-6 * maximum, "vanishing projection BN shift derivatives");
  check::Require(
      minimum > 1e-5 * maximum, "clearly nonzero EfficientNet derivatives with shifted BN and SE");
  std::cout << "  B0 with shifted BN betas and SE biases: gradient norms [" << minimum << ','
            << maximum << "] (smallest " << smallest
            << "); 16 projection BN shift derivatives vanish (largest " << removed << ")"
            << std::endl;
}

} // namespace efficientnet_checks

namespace efficientnet_v2_checks {

namespace nn = datamunge::nn;
using mobile_checks::EndsWith;
using mobile_checks::Pattern;

inline void CostChecks() {
  struct Case {
    const char*  name;
    std::int64_t params, state, macs;
    int          blocks;
  };
  const std::vector<Case> cases = {
      {"s", 21458488, 153872, 8365617152, 40},
      {"m", 54139356, 292032, 24582460384, 57},
      {"l", 118515272, 512576, 56079699008, 79},
      {"b0", 7139704, 60608, 718086176, 21},
      {"b1", 8141052, 71072, 1196819360, 27},
      {"b2", 10096086, 82288, 1697542832, 28},
      {"b3", 14377062, 109312, 3213058816, 32}};
  for (const auto& c : cases) {
    const auto                   spec = mbconv::EfficientNetV2Spec(c.name);
    const int                    r    = spec.resolution;
    nn::ParameterStore           p(17), state(17);
    nn::NoGradGuard              guard;
    mbconv::EfficientNetV2Config cfg;
    cfg.variant       = c.name;
    const auto net    = mbconv::EfficientNetV2(p, state, nn::zeros({1, r, r, 3}), cfg);
    const auto macs   = net.Macs();
    int        blocks = 0, fused = 0;
    for (const auto& b : mbconv::EfficientNetV2Blocks(spec)) {
      blocks += b.repeats;
      fused += b.fused ? b.repeats : 0;
    }
    check::Require(
        p.count() == c.params && state.count() == c.state && macs == c.macs && blocks == c.blocks,
        "EfficientNetV2 parameters, multiply-adds and depth");
    std::cout << "  EfficientNetV2-" << c.name << " at " << r << " px: " << blocks << " blocks ("
              << fused << " fused), " << p.count() << " parameters, " << std::fixed
              << std::setprecision(2) << macs / 1e9 << "B mult-adds" << std::defaultfloat
              << std::setprecision(6) << std::endl;
  }
  check::Require(
      mbconv::DecodeBlock("r4_k3_s2_e4_i24_o48_c1").fused &&
          mbconv::DecodeBlock("r4_k3_s2_e4_i24_o48_c1").stride == 2,
      "V2 block strings");
}

// The first block of S's second stage (24 -> 48, expansion 4, stride 2) at 96 px.
inline void FusedChecks() {
  for (bool fused : {true, false}) {
    nn::ParameterStore                  p(17), state(17);
    std::vector<mobilenet::MobileLayer> layers;
    mobilenet::MobileContext            c{p, state, layers, false};
    nn::NoGradGuard                     guard;
    auto                                b = mbconv::DecodeBlock("r1_k3_s2_e4_i24_o48_c1");
    b.fused                               = fused;
    mobilenet::MobileNetOutputs net;
    mbconv::MBConv(c, "block", nn::zeros({1, 96, 96, 24}), 48, b, 2, true);
    net.layers      = layers;
    const auto macs = net.Macs();
    check::Require(
        macs == (fused ? 58392576 : 33841152) && p.count() == (fused ? 25632 : 8256) &&
            layers.size() == (fused ? 2u : 3u),
        "Fused-MBConv and MBConv stage costs");
    std::cout << "  24->48 stride-2 expansion-4 block at 96 px as "
              << (fused ? "Fused-MBConv" : "MBConv      ") << ": " << std::fixed
              << std::setprecision(1) << macs / 1e6 << "M mult-adds, " << std::defaultfloat
              << std::setprecision(6) << p.count() << " parameters, " << layers.size()
              << " convolutions" << std::endl;
  }
}

inline void GradientChecks() {
  nn::ParameterStore           p(17), state(17);
  mbconv::EfficientNetV2Config cfg;
  cfg.variant        = "b0";
  cfg.drop_connect   = 0;
  cfg.update_stats   = false;
  const nn::Tensor x = Pattern({2, 224, 224, 3}, 1, 0);
  // Projection shifts that only reach 1x1 convolutions (then BN).
  std::vector<mbconv::BlockArgs> flat;
  std::vector<int>               strides;
  for (const auto& b : mbconv::EfficientNetV2Blocks(mbconv::EfficientNetV2Spec("b0")))
    for (int r = 0; r < b.repeats; ++r) {
      flat.push_back(b);
      flat.back().input = r ? b.output : b.input;
      strides.push_back(r ? 1 : b.stride);
    }
  std::set<std::string> removable;
  for (std::size_t j = 0; j < flat.size(); ++j) {
    if (flat[j].fused && flat[j].expansion == 1)
      continue; // projection followed by an activation
    bool only_pointwise = true;
    for (std::size_t k = j + 1; k <= flat.size(); ++k) {
      if (k == flat.size())
        break; // the head is a 1x1 convolution
      if (flat[k].fused || flat[k].expansion == 1) {
        only_pointwise = false; // first convolution is not 1x1
        break;
      }
      const bool identity = flat[k].skip && strides[k] == 1 && flat[k].input == flat[k].output;
      if (!identity)
        break;
    }
    if (only_pointwise)
      removable.insert("block" + std::to_string(j + 1) + "/project/bn/beta");
  }
  {
    nn::NoGradGuard guard;
    mbconv::EfficientNetV2(p, state, x, cfg, true);
  }
  mobile_checks::ShiftBetas(p);
  const auto  net     = mbconv::EfficientNetV2(p, state, x, cfg, true);
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0, removed = 0;
  int         removed_count = 0, kept_shifts = 0;
  std::string smallest;
  for (double norm : norms)
    maximum = std::max(maximum, norm);
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const double norm = norms[i];
    const auto&  name = p.names()[i];
    check::Require(std::isfinite(norm), "finite EfficientNetV2 derivative");
    if (removable.count(name)) {
      removed = std::max(removed, norm);
      ++removed_count;
    } else {
      if (EndsWith(name, "/project/bn/beta"))
        ++kept_shifts;
      if (norm < minimum) {
        minimum  = norm;
        smallest = name;
      }
    }
  }
  check::Require(
      removed_count == int(removable.size()) && removed < 1e-6 * maximum,
      "vanishing projection shifts that reach only 1x1 convolutions");
  check::Require(
      minimum > 1e-5 * maximum,
      "clearly nonzero EfficientNetV2 derivatives with shifted BN and SE");
  std::cout << "  base B0 with shifted BN betas and SE biases: gradient norms [" << minimum << ','
            << maximum << "] (smallest " << smallest << "); " << removed_count
            << " projection shift derivatives vanish (largest " << removed << "), " << kept_shifts
            << " that reach a fused block do not" << std::endl;
}

} // namespace efficientnet_v2_checks
