// RegNet X/Y, port of tflow's quantized-width design and grouped residual models.
// https://arxiv.org/abs/2003.13678
#pragma once
#include <cfenv>
#include <cmath>
#include <map>

#include "common/grouped.hpp"
#include "common/mobilenet.hpp"
namespace regnet {
namespace nn = datamunge::nn;
struct RegNetParams {
  int    depth, w0;
  double wa, wm;
  int    group;
  double se;
};
inline RegNetParams RegNetModel(const std::string& name) {
  static const std::map<std::string, RegNetParams> models = {
      {"x400mf", {22, 24, 24.48, 2.54, 16, 0}},
      {"x800mf", {16, 56, 35.73, 2.28, 16, 0}},
      {"x1.6gf", {18, 80, 34.01, 2.25, 24, 0}},
      {"x3.2gf", {25, 88, 26.31, 2.25, 48, 0}},
      {"x8gf", {23, 80, 49.56, 2.88, 120, 0}},
      {"x16gf", {22, 216, 55.59, 2.1, 128, 0}},
      {"x32gf", {23, 320, 69.86, 2.0, 168, 0}},
      {"y400mf", {16, 48, 27.89, 2.09, 8, .25}},
      {"y800mf", {14, 56, 38.84, 2.4, 16, .25}},
      {"y1.6gf", {27, 48, 20.71, 2.65, 24, .25}},
      {"y3.2gf", {21, 80, 42.63, 2.66, 24, .25}},
      {"y8gf", {17, 192, 76.82, 2.19, 56, .25}},
      {"y16gf", {18, 200, 106.23, 2.48, 112, .25}},
      {"y32gf", {20, 232, 115.89, 2.53, 232, .25}}};
  if (!models.count(name))
    throw std::invalid_argument("unknown RegNet model " + name);
  return models.at(name);
}
struct RegNetStages {
  std::vector<float> continuous;
  std::vector<int>   quantized, widths, depths, groups;
};
inline RegNetStages RegNetDesign(const RegNetParams& m) {
  if (m.depth < 1 || m.w0 <= 0 || m.w0 % 8 || m.wa < 0 || m.wm <= 1)
    throw std::invalid_argument("invalid RegNet parameters");
  const int mode = std::fegetround();
  std::fesetround(FE_TONEAREST); // ties to even, as torch.round
  RegNetStages d;
  const float  w0 = float(m.w0), wa = float(m.wa), wm = float(m.wm), log_wm = float(std::log(m.wm));
  for (int j = 0; j < m.depth; ++j) {
    const float u = float(j) * wa + w0, step = std::nearbyint(std::log(u / w0) / log_wm);
    d.continuous.push_back(u);
    d.quantized.push_back(int(std::nearbyint(w0 * std::pow(wm, step) / 8.f)) * 8);
  }
  std::fesetround(mode);
  for (int w : d.quantized) {
    if (!d.widths.empty() && d.widths.back() == w)
      ++d.depths.back();
    else {
      d.widths.push_back(w);
      d.depths.push_back(1);
    }
  }
  for (auto& w : d.widths) {
    const int g = std::min(m.group, w);
    d.groups.push_back(g);
    w = mobilenet::MakeDivisible(w, g);
  }
  return d;
}

inline nn::Tensor ConvBN(
    mobilenet::MobileContext& c,
    const std::string&        name,
    nn::Tensor                x,
    int                       out,
    int                       kernel,
    int                       stride,
    int                       groups,
    bool                      relu) {
  const auto           in = x.size(3);
  nn::ConvLayerOptions cfg;
  cfg.bias    = false;
  cfg.init    = nn::init::he_uniform();
  cfg.stride  = stride;
  cfg.padding = nn::Padding::Valid;
  if (kernel > 1)
    x = nn::pad(x, {{0, 0}, {kernel / 2, kernel / 2}, {kernel / 2, kernel / 2}, {0, 0}});
  auto h = grouped::FusedGroupedConv2D(c.p, name, x, out, kernel, groups, cfg);
  h      = mobilenet::ContextBatchNorm(c, name + "/bn", h, 1e-5);
  c.layers.push_back({name, h.shape(), std::int64_t(kernel) * kernel * in * out / groups});
  return relu ? nn::relu(h) : h;
}
inline nn::Tensor Block(
    mobilenet::MobileContext& c,
    const std::string&        name,
    const nn::Tensor&         x,
    int                       out,
    int                       stride,
    int                       group_width,
    double                    se) {
  const int in = x.size(3);
  auto      h  = ConvBN(c, name + "/a", x, out, 1, 1, 1, true);
  h            = ConvBN(c, name + "/b", h, out, 3, stride, out / group_width, true);
  if (se > 0) {
    const int            squeeze = std::lround(se * in);
    nn::ConvLayerOptions cfg;
    cfg.init      = nn::init::he_uniform();
    auto pooled   = nn::mean(h, {1, 2}, true);
    auto squeezed = nn::relu(nn::conv2d(c.p, name + "/se/reduce", pooled, squeeze, 1, cfg));
    h = nn::mul(h, nn::sigmoid(nn::conv2d(c.p, name + "/se/expand", squeezed, out, 1, cfg)));
    c.layers.push_back({name + "/se", {1, 1, 1, 1}, 2 * std::int64_t(out) * squeeze});
  }
  h = ConvBN(c, name + "/c", h, out, 1, 1, 1, false);
  auto shortcut =
      in == out && stride == 1 ? x : ConvBN(c, name + "/projection", x, out, 1, stride, 1, false);
  return nn::relu(nn::add(shortcut, h));
}
struct RegNetConfig {
  std::string model        = "y400mf";
  int         classes      = 1000;
  bool        small_stem   = false;
  double      momentum     = .9;
  bool        update_stats = true;
};
inline mobilenet::MobileNetOutputs RegNet(
    nn::ParameterStore& p,
    nn::ParameterStore& state,
    const nn::Tensor&   image,
    const RegNetConfig& cfg      = {},
    bool                training = false) {
  if (cfg.classes < 2)
    throw std::invalid_argument("invalid RegNet configuration");
  auto                        model  = RegNetModel(cfg.model);
  auto                        design = RegNetDesign(model);
  mobilenet::MobileNetOutputs out;
  mobilenet::MobileContext    c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  auto                        h = ConvBN(c, "stem", image, 32, 3, cfg.small_stem ? 1 : 2, 1, true);
  for (std::size_t stage = 0; stage < design.widths.size(); ++stage)
    for (int block = 0; block < design.depths[stage]; ++block)
      h = Block(
          c,
          "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1),
          h,
          design.widths[stage],
          block ? 1 : 2,
          design.groups[stage],
          model.se);
  out.features        = h;
  out.logits          = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  out.classifier_macs = h.size(3) * cfg.classes;
  return out;
}
} // namespace regnet
