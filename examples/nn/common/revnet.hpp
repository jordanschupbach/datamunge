// Reversible residual networks (RevNet) and their memory-saving backward pass:
// a port of tflow's common/nn/revnet.h.
// https://arxiv.org/abs/1707.04585
// https://github.com/renmengye/revnet-public/blob/master/resnet/models/revnet_model.py
//
// Follow the authors' release. Each unit splits its channels into halves and
// computes y1 = x1' + F(x2), y2 = x2' + G(y1), where F and G are
// pre-activation residual functions (BN/ReLU/3x3 twice, or the bottleneck
// BN/ReLU/1x1, 3x3, 1x1 to four times the width) and x' is the shortcut.
// Basic units average-pool and zero-pad half the added channels on each side;
// bottleneck units project with a bias-free 1x1 when the stride or width
// changes. Only those first units of each stage are non-invertible and their
// inputs are stored; every other unit is inverted as x2 = y2 - G(y1),
// x1 = y1 - F(x2). The very first F skips its BN/ReLU (the stem has one).
// Stems are 3x3 conv/BN/ReLU (CIFAR) or 7x7/2 conv/BN/ReLU and 3x3/2 max-pool
// (ImageNet); the head is BN/ReLU, global average pooling and a biased
// linear layer. Convolutions use SAME padding and no bias; BN epsilon is
// 1e-3. He-uniform fan-in initialization replaces the release's
// truncated-normal fan-out.
//
// tflow builds the reversible gradient graph by hand (RevNetReversibleGradients:
// reconstruct each unit's input from its output, recompute the unit behind
// StopGradient, backpropagate G then F). Here the invertible units of each
// stage run through nn::reversible_sequence, whose backward does exactly that,
// so ordinary nn::grad / Optimizer::minimize use the reversible pass. The
// mode `StoredUnits` wraps every unit in nn::checkpoint instead: the same
// unit-by-unit derivatives from stored inputs (tflow's reconstruct=false).
// Batch norm skips its moving-average update while a unit is recomputed.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace revnet {

namespace nn = datamunge::nn;

struct RevNetConfig {
  std::vector<int> units = {9, 9, 9}, filters = {32, 32, 64, 128}, strides = {1, 2, 2};
  bool             bottleneck = false, imagenet = false;
  int              classes  = 10;
  double           momentum = .9;
  static RevNetConfig Cifar(int depth, int classes = 10) {
    RevNetConfig c;
    c.classes = classes;
    if (depth == 38) {
      c.units   = {3, 3, 3};
      c.filters = {32, 32, 64, 112};
    } else if (depth == 164) {
      c.bottleneck = true;
    } else if (depth != 110) {
      throw std::invalid_argument("CIFAR RevNet depths are 38/110/164");
    }
    return c;
  }
  static RevNetConfig ImageNet(int depth) {
    RevNetConfig c;
    c.imagenet = c.bottleneck = true;
    c.classes                 = 1000;
    c.filters                 = {128, 128, 256, 512, 832};
    c.strides                 = {1, 2, 2, 2};
    if (depth == 56)
      c.units = {2, 2, 3, 2};
    else if (depth == 104)
      c.units = {2, 2, 11, 2};
    else
      throw std::invalid_argument("ImageNet RevNet depths are 56/104");
    return c;
  }
  // Channels entering stage 1, then leaving each stage.
  std::vector<int> Widths() const {
    if (filters.size() != units.size() + 1 || strides.size() != units.size() || classes < 2 ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid RevNet configuration");
    std::vector<int> w = filters;
    if (bottleneck)
      for (std::size_t i = 1; i < w.size(); ++i) w[i] *= 4;
    for (std::size_t i = 0; i < w.size(); ++i)
      if (w[i] < 2 || w[i] % 2 || (i && (w[i] < w[i - 1] || (!bottleneck && (w[i] - w[i - 1]) % 4))))
        throw std::invalid_argument("invalid RevNet widths");
    for (std::size_t i = 0; i < units.size(); ++i)
      if (units[i] < 1 || (strides[i] != 1 && strides[i] != 2)) throw std::invalid_argument("invalid RevNet stage");
    return w;
  }
};

// Normalization context: training mode uses batch statistics; moving
// averages are updated only when `update` is set (and never while a unit is
// recomputed during the backward pass).
struct RevContext {
  nn::ParameterStore& p;
  nn::ParameterStore& state;
  bool                training;
  double              momentum;
  bool                bottleneck;
  bool                update = true;
};
struct RevPair {
  nn::Tensor first, second;
};

inline nn::Tensor RevConv(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out,
                          std::int64_t kernel, std::int64_t stride = 1) {
  nn::ConvLayerOptions o;
  o.bias    = false;
  o.init    = nn::init::he_uniform();
  o.stride  = stride;
  o.padding = nn::Padding::Same;
  return nn::conv2d(p, name, x, out, kernel, o);
}
inline nn::Tensor RevNormRelu(const RevContext& c, const std::string& name, const nn::Tensor& x) {
  nn::BatchNormOptions o;
  o.momentum     = c.momentum;
  o.eps          = 1e-3;
  o.update_stats = c.update && !nn::in_recompute();
  return nn::relu(nn::batch_norm(c.p, c.state, name, x, c.training, o));
}
// F or G: a pre-activation residual function from `in` to `out` channels.
inline nn::Tensor RevResidual(const RevContext& c, const std::string& name, nn::Tensor x, int out, int stride,
                              bool no_activation) {
  if (!no_activation) x = RevNormRelu(c, name + "/bn1", x);
  if (c.bottleneck) {
    x = RevConv(c.p, name + "/conv1", x, out / 4, 1, stride);
    x = RevConv(c.p, name + "/conv2", RevNormRelu(c, name + "/bn2", x), out / 4, 3);
    return RevConv(c.p, name + "/conv3", RevNormRelu(c, name + "/bn3", x), out, 1);
  }
  x = RevConv(c.p, name + "/conv1", x, out, 3, stride);
  return RevConv(c.p, name + "/conv2", RevNormRelu(c, name + "/bn2", x), out, 3);
}
inline nn::Tensor RevShortcut(const RevContext& c, const std::string& name, nn::Tensor x, int in, int out, int stride) {
  if (c.bottleneck) return stride > 1 || in != out ? RevConv(c.p, name + "/projection", x, out, 1, stride) : x;
  if (stride > 1) x = nn::avg_pool2d(x, stride, stride, nn::Padding::Valid);
  if (in == out) return x;
  const std::int64_t pad = (out - in) / 2;
  return nn::pad(x, {{0, 0}, {0, 0}, {0, 0}, {pad, pad}});
}
inline RevPair RevBlockForward(const RevContext& c, const std::string& name, const RevPair& x, int in, int out,
                               int stride, bool no_activation) {
  const int  hi = in / 2, ho = out / 2;
  nn::Tensor y1 = RevResidual(c, name + "/f", x.second, ho, stride, no_activation) +
                  RevShortcut(c, name + "/x1", x.first, hi, ho, stride);
  nn::Tensor y2 = RevResidual(c, name + "/g", y1, ho, 1, false) + RevShortcut(c, name + "/x2", x.second, hi, ho, stride);
  return {y1, y2};
}
// Inverse of an equal-width, stride-one unit.
inline RevPair RevBlockInverse(const RevContext& c, const std::string& name, const RevPair& y, int channels) {
  const int  h  = channels / 2;
  nn::Tensor x2 = y.second - RevResidual(c, name + "/g", y.first, h, 1, false);
  nn::Tensor x1 = y.first - RevResidual(c, name + "/f", x2, h, 1, false);
  return {x1, x2};
}
// The coupling functions of an equal-width unit, for nn::reversible_sequence.
// `c` is copied: the backward pass calls F and G after this returns.
inline nn::ReversibleBlock RevCoupling(const RevContext& c, const std::string& name, int channels) {
  const int h = channels / 2;
  return {[c, name, h](const nn::Tensor& x2) { return RevResidual(c, name + "/f", x2, h, 1, false); },
          [c, name, h](const nn::Tensor& y1) { return RevResidual(c, name + "/g", y1, h, 1, false); }};
}
inline nn::Tensor RevNetHead(const RevContext& c, const RevPair& x, int classes, nn::Tensor* features = nullptr) {
  nn::Tensor h = RevNormRelu(c, "final/bn", nn::concat({x.first, x.second}, 3));
  if (features) *features = h;
  return nn::dense(c.p, "output", nn::global_avg_pool2d(h), classes);
}

struct RevNetUnit {
  std::string name;
  int         in, out, stride;
  bool        no_activation, stored;  // stored: non-invertible, its input is kept for the backward pass
  RevPair     input, output;          // filled in Ordinary mode
};
/// Ordinary: plain autograd keeps every activation. Reversible: invertible
/// units run through nn::reversible_sequence and the stored units are
/// checkpoints (tflow's RevNetReversibleGradients). StoredUnits: every unit is
/// an nn::checkpoint recomputed from its stored input (reconstruct=false).
enum class Mode { Ordinary, Reversible, StoredUnits };
struct RevNetOutputs {
  nn::Tensor              stem, features, logits;
  std::vector<RevNetUnit> units;
};

/// The unit layout (names, widths, strides) of a configuration.
inline std::vector<RevNetUnit> RevNetUnits(const RevNetConfig& cfg) {
  const auto              widths = cfg.Widths();
  std::vector<RevNetUnit> units;
  for (std::size_t stage = 0; stage < cfg.units.size(); ++stage)
    for (int unit = 0; unit < cfg.units[stage]; ++unit)
      units.push_back({"stage" + std::to_string(stage + 1) + "/unit" + std::to_string(unit + 1),
                       unit ? widths[stage + 1] : widths[stage], widths[stage + 1], unit ? 1 : cfg.strides[stage],
                       stage == 0 && unit == 0, unit == 0, {}, {}});
  return units;
}

inline RevNetOutputs RevNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                            const RevNetConfig& cfg = {}, bool training = false, Mode mode = Mode::Ordinary,
                            bool update = true) {
  const auto       widths = cfg.Widths();
  RevNetOutputs    out;
  const RevContext c{p, state, training, cfg.momentum, cfg.bottleneck, update};
  nn::Tensor h = RevNormRelu(c, "stem/bn", RevConv(p, "stem", image, widths[0], cfg.imagenet ? 7 : 3, cfg.imagenet ? 2 : 1));
  if (cfg.imagenet) h = nn::max_pool2d(h, 3, 2, nn::Padding::Same);
  out.stem    = h;
  auto halves = nn::split(h, 2, 3);
  RevPair x{halves[0], halves[1]};
  out.units = RevNetUnits(cfg);
  for (std::size_t k = 0; k < out.units.size();) {
    auto& u = out.units[k];
    if (mode == Mode::Reversible && !u.stored) {
      // Every following invertible unit of this stage in one reversible chain.
      std::vector<nn::ReversibleBlock> chain;
      for (; k < out.units.size() && !out.units[k].stored; ++k) chain.push_back(RevCoupling(c, out.units[k].name, out.units[k].out));
      auto y = nn::reversible_sequence(chain, x.first, x.second);
      x      = {y[0], y[1]};
      continue;
    }
    // Stored (non-invertible) units of the reversible pass, and every unit of
    // StoredUnits, keep only their input and recompute during backward.
    if (mode != Mode::Ordinary) {
      auto unit = [c, u](const std::vector<nn::Tensor>& in) {
        auto y = RevBlockForward(c, u.name, {in[0], in[1]}, u.in, u.out, u.stride, u.no_activation);
        return std::vector<nn::Tensor>{y.first, y.second};
      };
      auto y = nn::checkpoint(unit, {x.first, x.second});
      x      = {y[0], y[1]};
    } else {
      u.input  = x;
      x        = RevBlockForward(c, u.name, x, u.in, u.out, u.stride, u.no_activation);
      u.output = x;
    }
    ++k;
  }
  out.logits = RevNetHead(c, x, cfg.classes, &out.features);
  return out;
}

}  // namespace revnet
