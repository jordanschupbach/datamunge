// SRCNN, ESPCN, VDSR, EDSR and RCAN with explicit subpixel channel ordering,
// and short five-model x2 SR training with honest common-crop bicubic
// comparisons: a port of tflow's common/nn/super_resolution.h and
// common/ml/super_resolution.h (namespace sr; the data/metric half,
// common/ml/sr_data.h, is common/sr_data.hpp).
// TF -> datamunge: parameters are an nn::ParameterStore (seed 17, datamunge's
// generator); SRConv / SRPixelShuffle are reused from
// common/adversarial_super_resolution.hpp (srgan::, identical to tflow's);
// the checkpoint is best.npz; PNGs via nn::save_png; test_outputs.npz via
// nn::save_npz. nn::depth_to_space is TF's DCR order, so (like tflow) the CRD
// shuffle is an explicit reshape/transpose.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "common/adversarial_super_resolution.hpp"
#include "common/sr_data.hpp"

namespace sr {

enum class SRKind { kSrcnn, kEspcn, kVdsr, kEdsr, kRcan };
inline const char* SRName(SRKind k) {
  switch (k) {
    case SRKind::kSrcnn: return "srcnn";
    case SRKind::kEspcn: return "espcn";
    case SRKind::kVdsr: return "vdsr";
    case SRKind::kEdsr: return "edsr";
    case SRKind::kRcan: return "rcan";
  }
  throw std::invalid_argument("SR kind");
}

struct SRConfig {
  SRKind kind = SRKind::kSrcnn;
  int    scale = 2, width = 64, blocks = 20, groups = 10, reduction = 16;
  static SRConfig Full(SRKind kind) {
    SRConfig c;
    c.kind = kind;
    if (kind == SRKind::kEdsr) c.width = 256, c.blocks = 32;
    return c;
  }
  static SRConfig Compact(SRKind kind) {
    auto c = Full(kind);
    if (kind == SRKind::kVdsr) c.width = 16;
    if (kind == SRKind::kEdsr) c.width = 16, c.blocks = 8;
    if (kind == SRKind::kRcan) c.width = 16, c.groups = 2, c.blocks = 3, c.reduction = 4;
    return c;
  }
  bool PreUpsampled() const { return kind == SRKind::kSrcnn || kind == SRKind::kVdsr; }
  int  Channels() const { return kind == SRKind::kEdsr || kind == SRKind::kRcan ? 3 : 1; }
};

/// PyTorch/paper CRD: input channel = c r r + dy r + dx (srgan's explicit reshape/transpose).
inline Tensor SRPixelShuffle(const Tensor& x, int channels, int scale) {
  if (channels < 1 || scale < 2) throw std::invalid_argument("pixel shuffle channels/scale");
  return srgan::SRPixelShuffle(x, channels, scale);
}

/// Fan-in uniform weights (Torch's default Conv2d bound), zero biases, SAME; optional ReLU.
inline Tensor SRConv(nn::ParameterStore& p, const std::string& name, const Tensor& x, int in, int out, int kernel, bool relu = false) {
  const Tensor y = srgan::SRConv(p, name, x, in, out, kernel);
  return relu ? nn::relu(y) : y;
}

/// tflow Conv2DLayer: Glorot-uniform (or given) kernel, zero bias.
inline Tensor SRLayer(nn::ParameterStore& p, const std::string& name, const Tensor& x, int out, int kernel,
                      nn::Activation act = nn::Activation::None, nn::Padding padding = nn::Padding::Same,
                      const nn::Initializer& init = nn::init::glorot_uniform()) {
  nn::ConvLayerOptions o;
  o.activation = act;
  o.padding    = padding;
  o.init       = init;
  return nn::conv2d(p, name, x, out, kernel, o);
}

inline Tensor SRResidual(nn::ParameterStore& p, const std::string& name, const Tensor& input, int width, double scaling) {
  const Tensor h        = SRConv(p, name + "/conv1", input, width, width, 3, true);
  const Tensor residual = SRConv(p, name + "/conv2", h, width, width, 3);
  return input + scaling * residual;
}

struct SRChannelGate {
  Tensor output, gates;
};
inline SRChannelGate SRChannelAttention(nn::ParameterStore& p, const std::string& name, const Tensor& input, int width, int reduction) {
  if (reduction < 1 || width % reduction) throw std::invalid_argument("channel attention reduction");
  const Tensor mean   = nn::mean(input, {1, 2}, true);
  const Tensor narrow = SRConv(p, name + "/reduce", mean, width, width / reduction, 1, true);
  const Tensor gate   = nn::sigmoid(SRConv(p, name + "/expand", narrow, width / reduction, width, 1));
  return {input * gate, gate};
}
inline Tensor SRRcab(nn::ParameterStore& p, const std::string& name, const Tensor& input, int width, int reduction) {
  const Tensor a = SRConv(p, name + "/conv1", input, width, width, 3, true);
  const Tensor b = SRConv(p, name + "/conv2", a, width, width, 3);
  return input + SRChannelAttention(p, name + "/ca", b, width, reduction).output;
}
inline Tensor SRUpsample(nn::ParameterStore& p, const Tensor& input, int width, int scale) {
  if (scale != 2 && scale != 3 && scale != 4) throw std::invalid_argument("SR scale2,3,4 required");
  Tensor    x      = input;
  const int stages = scale == 4 ? 2 : 1, r = scale == 3 ? 3 : 2;
  for (int i = 0; i < stages; ++i) x = SRPixelShuffle(SRConv(p, "upsample" + std::to_string(i), x, width, width * r * r, 3), width, r);
  return x;
}

struct SROutputs {
  Tensor              image;
  std::vector<Tensor> stages;
};
inline const std::array<float, 3> kSRMean = {.4488f, .4371f, .4040f};

inline SROutputs SuperResolution(nn::ParameterStore& p, const Tensor& input, const SRConfig& cfg) {
  if (cfg.width < 1 || cfg.blocks < 1 || cfg.groups < 1 || cfg.scale < 2) throw std::invalid_argument("SR configuration");
  SROutputs net;
  if (cfg.kind == SRKind::kSrcnn) {
    const Tensor a = SRLayer(p, "extract", input, 64, 9, nn::Activation::Relu, nn::Padding::Valid);
    const Tensor b = SRLayer(p, "mapping", a, 32, 1, nn::Activation::Relu, nn::Padding::Valid);
    net.image      = SRLayer(p, "reconstruct", b, 1, 5, nn::Activation::None, nn::Padding::Valid);
    net.stages     = {a, b};
    return net;
  }
  if (cfg.kind == SRKind::kEspcn) {
    const Tensor a      = SRLayer(p, "extract", input, 64, 5, nn::Activation::Tanh);
    const Tensor b      = SRLayer(p, "mapping", a, 32, 3, nn::Activation::Tanh);
    const Tensor packed = SRLayer(p, "subpixel", b, cfg.scale * cfg.scale, 3);
    net.image           = SRPixelShuffle(packed, 1, cfg.scale);
    net.stages          = {a, b, packed};
    return net;
  }
  if (cfg.kind == SRKind::kVdsr) {
    Tensor x = SRLayer(p, "input", input, cfg.width, 3, nn::Activation::Relu, nn::Padding::Same, nn::init::he_normal());
    net.stages.push_back(x);
    for (int i = 0; i < 18; ++i) {
      x = SRLayer(p, "hidden" + std::to_string(i), x, cfg.width, 3, nn::Activation::Relu, nn::Padding::Same, nn::init::he_normal());
      net.stages.push_back(x);
    }
    // Small final initialization starts the short demonstration near bicubic;
    // this is explicit, not an extra shortcut absent from the architecture.
    const Tensor w        = p.get("residual/W", {3, 3, cfg.width, 1}, nn::init::normal(.001));
    const Tensor residual = nn::bias_add(nn::conv2d(x, w, nn::conv_options(1, nn::Padding::Same)), p.get("residual/b", {1}, nn::init::zeros()));
    net.image             = input + residual;
    return net;
  }
  const Tensor mean = nn::from_vector(std::vector<float>(kSRMean.begin(), kSRMean.end()), {3}).to(input.dtype());
  const Tensor head = SRConv(p, "head", input - mean, 3, cfg.width, 3);
  Tensor       x    = head;
  net.stages.push_back(head);
  if (cfg.kind == SRKind::kEdsr) {
    for (int i = 0; i < cfg.blocks; ++i) {
      x = SRResidual(p, "block" + std::to_string(i), x, cfg.width, .1);
      net.stages.push_back(x);
    }
  } else {
    for (int g = 0; g < cfg.groups; ++g) {
      const Tensor group = x;
      for (int b = 0; b < cfg.blocks; ++b)
        x = SRRcab(p, "group" + std::to_string(g) + "/block" + std::to_string(b), x, cfg.width, cfg.reduction);
      x = group + SRConv(p, "group" + std::to_string(g) + "/tail", x, cfg.width, cfg.width, 3);
      net.stages.push_back(x);
    }
  }
  x = head + SRConv(p, "body_tail", x, cfg.width, cfg.width, 3);
  net.stages.push_back(x);
  x = SRUpsample(p, x, cfg.width, cfg.scale);
  net.stages.push_back(x);
  net.image = SRConv(p, "output", x, cfg.width, 3, 3) + mean;
  return net;
}

/// VDSR adjustable clipping: gradient clipped to +-theta / learning_rate.
inline Tensor VdsrClippedGradient(const Tensor& gradient, double learning_rate, double theta = .01) {
  const double bound = double(float(theta) / float(learning_rate));
  return nn::clip(gradient, -bound, bound);
}

// ---------------------------------------------------------------- data extras
/// Studio-range BT.601 Y of RGB NHWC -> [N, H, W, 1].
inline Tensor SRY(const Tensor& rgb) {
  check::Require(rgb.dim() == 4 && rgb.size(3) == 3, "RGB to studio-range BT.601 Y");
  const auto         v = rgb.contiguous().to_vector<float>();
  std::vector<float> y(v.size() / 3);
  for (std::size_t i = 0; i < y.size(); ++i) y[i] = float(SRLuma(v[3 * i], v[3 * i + 1], v[3 * i + 2]));
  return nn::from_vector(y, {rgb.size(0), rgb.size(1), rgb.size(2), 1});
}
inline Tensor SRCrop(const Tensor& x, int border) {
  check::Require(x.dim() == 4 && border >= 0 && x.size(1) > 2 * border && x.size(2) > 2 * border, "SR center crop dimensions");
  return nn::slice(x, {0, border, border, 0}, {x.size(0), x.size(1) - 2 * border, x.size(2) - 2 * border, x.size(3)}).contiguous();
}

// ---------------------------------------------------------------- training
class SRModel {
 public:
  nn::ParameterStore             parameters{17};
  SRConfig                       config;
  std::unique_ptr<nn::Optimizer> optimizer;
  double                         peak = 0, rate = .001;
  std::int64_t                   clipped = 0;
  explicit SRModel(SRKind kind) : config(SRConfig::Compact(kind)) {
    const int side = config.PreUpsampled() ? 32 : 16;
    {
      nn::NoGradGuard guard;
      SuperResolution(parameters, nn::zeros({1, side, side, config.Channels()}), config);
    }
    nn::OptimizerConfig oc;
    oc.kind   = kind == SRKind::kVdsr ? nn::OptimizerKind::Momentum : nn::OptimizerKind::Adam;
    rate      = kind == SRKind::kVdsr ? .1 : .001;
    optimizer = std::make_unique<nn::Optimizer>(parameters, rate, oc);
  }
  Tensor Loss(const Tensor& x, const Tensor& y) {
    const Tensor diff = SuperResolution(parameters, x, config).image - y;
    return config.kind == SRKind::kEdsr || config.kind == SRKind::kRcan ? nn::mean(nn::abs(diff)) : nn::mean(nn::square(diff));
  }
  double Run(const Tensor& x, const Tensor& y, bool update = false) {
    double value = 0;
    if (update) {
      const Tensor loss = Loss(x, y);
      value             = loss.item();
      auto gradients    = nn::grad(loss, parameters.parameters());
      if (config.kind == SRKind::kVdsr) {
        const double bound = double(.01f / float(rate));
        for (auto& g : gradients) {
          for (float v : g.to_vector<float>()) {
            peak = std::max(peak, double(std::abs(v)));
            clipped += std::abs(v) > float(bound);
          }
          g = VdsrClippedGradient(g, rate);
        }
      }
      optimizer->step(gradients);
    } else {
      nn::NoGradGuard guard;
      value = Loss(x, y).item();
    }
    check::Require(std::isfinite(value) && value >= 0, "finite reconstruction loss");
    return value;
  }
  Tensor Predict(const Tensor& x) {
    nn::NoGradGuard guard;
    const Tensor    out = SuperResolution(parameters, x, config).image;
    for (float v : out.to_vector<float>()) check::Require(std::isfinite(v), "finite raw SR pixels");
    return out;
  }
  void Save(const std::string& path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i) arrays.push_back({parameters.names()[i], parameters.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string& path) {
    nn::NoGradGuard guard;
    const auto      arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size(), "complete SR checkpoint");
    for (const auto& [name, value] : arrays) parameters.set(name, value);
  }
};

struct SRPrepared {
  Tensor x, y;
};
inline SRPrepared PrepareSR(const SRData& data, const SRConfig& cfg) {
  Tensor x = cfg.PreUpsampled() ? data.bicubic : data.lr, y = data.hr;
  if (cfg.Channels() == 1) x = SRY(x), y = SRY(y);
  if (cfg.kind == SRKind::kSrcnn) y = SRCrop(y, 6);
  return {x, y};
}
inline std::vector<std::int64_t> SRRows(std::int64_t start, std::int64_t end) {
  std::vector<std::int64_t> rows;
  for (std::int64_t i = start; i < end; ++i) rows.push_back(i);
  return rows;
}
inline double SRLoss(SRModel& model, const SRPrepared& data) {
  double sum = 0;
  for (std::int64_t row = 0; row < data.x.size(0); row += 8) {
    const auto rows = SRRows(row, std::min<std::int64_t>(row + 8, data.x.size(0)));
    sum += model.Run(SRBatch(data.x, rows), SRBatch(data.y, rows)) * double(rows.size());
  }
  return sum / double(data.x.size(0));
}
inline Tensor SRPredict(SRModel& model, const SRPrepared& data) {
  std::vector<Tensor> parts;
  for (std::int64_t row = 0; row < data.x.size(0); row += 8) {
    const auto   rows = SRRows(row, std::min<std::int64_t>(row + 8, data.x.size(0)));
    const Tensor out  = model.Predict(SRBatch(data.x, rows));
    check::Require(out.size(0) == std::int64_t(rows.size()) && out.numel() == std::int64_t(rows.size()) * (data.y.numel() / data.y.size(0)),
                   "SR image alignment");
    parts.push_back(out);
  }
  return nn::concat(parts, 0).contiguous();
}

/// bicubic | SR | HR | 8x |error| (red) Y, central 20x20, 4x scaled, 8 rows.
inline void SRPreview(const SRData& data, const Tensor& prediction, const std::filesystem::path& path) {
  const int  rows = int(std::min<std::int64_t>(8, prediction.size(0))), scale = 4, side = 20, H = rows * side * scale, W = 4 * side * scale;
  const bool crop = prediction.size(1) == 20;
  const int  channels = int(prediction.size(3)), ps = int(prediction.size(1));
  const auto p = prediction.contiguous().to_vector<float>(), hr = data.hr.contiguous().to_vector<float>(),
             bi = data.bicubic.contiguous().to_vector<float>();
  auto at = [](const std::vector<float>& v, int s, int c, int n, int y, int x, int k) { return double(v[((std::size_t(n) * s + y) * s + x) * c + k]); };
  std::vector<float> image(std::size_t(H) * W * 3, 0.f);
  for (int n = 0; n < rows; ++n)
    for (int y = 0; y < side; ++y)
      for (int x = 0; x < side; ++x) {
        const int    py = crop ? y : y + 6, px = crop ? x : x + 6;
        const double estimated = std::clamp(channels == 1 ? at(p, ps, 1, n, py, px, 0)
                                                          : SRLuma(at(p, ps, 3, n, py, px, 0), at(p, ps, 3, n, py, px, 1), at(p, ps, 3, n, py, px, 2)),
                                            0., 1.);
        const double truth    = SRLuma(at(hr, 32, 3, n, y + 6, x + 6, 0), at(hr, 32, 3, n, y + 6, x + 6, 1), at(hr, 32, 3, n, y + 6, x + 6, 2));
        const double baseline = std::clamp(SRLuma(at(bi, 32, 3, n, y + 6, x + 6, 0), at(bi, 32, 3, n, y + 6, x + 6, 1), at(bi, 32, 3, n, y + 6, x + 6, 2)), 0., 1.);
        const double values[] = {baseline, estimated, truth, std::min(1., 8 * std::abs(estimated - truth))};
        for (int col = 0; col < 4; ++col)
          for (int a = 0; a < scale; ++a)
            for (int b = 0; b < scale; ++b)
              for (int c = 0; c < 3; ++c)
                image[(std::size_t((n * side + y) * scale + a) * W + std::size_t((col * side + x) * scale + b)) * 3 + std::size_t(c)] =
                    float(std::lround(255 * (col == 3 && c != 0 ? 0 : values[col])));
      }
  nn::save_png(nn::from_vector(image, {H, W, 3}), path.string(), 0, 255);
}

struct SRRunSummary {
  double initial = 0, learned = 0, best = 0, seconds = 0;
  int    selected = -1;
  SRMetrics metrics;
};

}  // namespace sr
