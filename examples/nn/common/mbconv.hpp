// Mobile inverted bottleneck (MBConv) blocks described by the TensorFlow
// reference block strings, and the MnasNet-A1 / B1, EfficientNet and
// EfficientNetV2 networks built from them (port of tflow's common/nn/mbconv.h).
// https://arxiv.org/abs/1807.11626
// https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
//
// A string such as "r3_k5_s22_e6_i40_o80_se0.25" repeats a block three
// times: 5x5 depthwise kernel, stride two on the first repeat, expansion
// six, 40 input and 80 output channels (inputs become outputs after the
// first repeat), squeeze-and-excitation with max(1, int(input * 0.25))
// reduced channels and a logistic gate; "noskip" disables the identity
// connection, which otherwise applies at stride one with equal widths; "c1"
// marks a Fused-MBConv block (EfficientNetV2), and "s2" means "s22". A
// block is 1x1 expansion/BN/act (skipped for expansion one), depthwise/BN/act,
// optional SE, then a linear 1x1 projection with BN. Activations are ReLU
// (MnasNet) or swish (EfficientNet). Training may drop the residual branch
// per example (drop connect, rescaled by the survival probability). The
// depth multiplier rounds channels to multiples of eight, never down more
// than 10%. MnasNet adds a 3x3/2 32-channel conv/BN/ReLU stem and a 1x1
// 1280-channel conv/BN/ReLU head, global pooling, dropout(0.2) and a biased
// classifier. Convolutions use TensorFlow SAME padding; BN epsilon is 1e-3.
// Dropout and drop-connect masks draw from the parameter store's generator
// (tflow: seeded TF random ops).
#pragma once

#include <cmath>
#include <regex>
#include <sstream>

#include "common/mobilenet.hpp"

namespace mbconv {

namespace nn = datamunge::nn;
using mobilenet::ConvBN;
using mobilenet::MakeDivisible;
using mobilenet::MobileContext;
using mobilenet::MobileConv;
using mobilenet::MobileNetOutputs;

struct BlockArgs {
  int    repeats = 1, kernel = 3, stride = 1, expansion = 1, input = 0, output = 0;
  double se_ratio = 0;
  bool   skip = true, fused = false;  // fused: "c1", EfficientNetV2's Fused-MBConv
};
inline BlockArgs DecodeBlock(const std::string& text) {
  BlockArgs         b;
  bool              kernel = false, stride = false, expansion = false;
  std::stringstream parts(text);
  std::string       part;
  while (std::getline(parts, part, '_')) {
    std::smatch m;
    if (part == "noskip")
      b.skip = false;
    else if (std::regex_match(part, m, std::regex("r(\\d+)")))
      b.repeats = std::stoi(m[1]);
    else if (std::regex_match(part, m, std::regex("k(\\d+)"))) {
      b.kernel = std::stoi(m[1]);
      kernel   = true;
    } else if (std::regex_match(part, m, std::regex("s(\\d)(\\d?)"))) {
      if (m[2].length() && m[1] != m[2]) throw std::invalid_argument("only square strides are supported: " + text);
      b.stride = std::stoi(m[1]);
      stride   = true;
    } else if (std::regex_match(part, m, std::regex("c(\\d)")))
      b.fused = m[1] == "1";
    else if (std::regex_match(part, m, std::regex("e(\\d+)"))) {
      b.expansion = std::stoi(m[1]);
      expansion   = true;
    } else if (std::regex_match(part, m, std::regex("i(\\d+)")))
      b.input = std::stoi(m[1]);
    else if (std::regex_match(part, m, std::regex("o(\\d+)")))
      b.output = std::stoi(m[1]);
    else if (std::regex_match(part, m, std::regex("se([0-9.]+)")))
      b.se_ratio = std::stod(m[1]);
    else
      throw std::invalid_argument("unknown block field '" + part + "' in " + text);
  }
  if (!kernel || !stride || !expansion || b.repeats < 1 || b.input < 1 || b.output < 1 || !(b.se_ratio >= 0 && b.se_ratio <= 1))
    throw std::invalid_argument("incomplete block string: " + text);
  return b;
}
inline std::string EncodeBlock(const BlockArgs& b) {
  std::ostringstream text;
  text << 'r' << b.repeats << "_k" << b.kernel << "_s" << b.stride << b.stride << "_e" << b.expansion << "_i" << b.input
       << "_o" << b.output;
  if (b.se_ratio > 0) text << "_se" << b.se_ratio;
  if (b.fused) text << "_c1";
  if (!b.skip) text << "_noskip";
  return text.str();
}
// Depth-multiplier rounding of the TensorFlow reference.
inline int RoundFilters(int filters, double multiplier, int divisor = 8) {
  return multiplier == 1 ? filters : MakeDivisible(filters * multiplier, divisor);
}
inline nn::Tensor Swish(const nn::Tensor& x) { return x * nn::sigmoid(x); }
/// Per-example survival mask for the residual branch, rescaled by 1/p.
inline nn::Tensor DropConnect(const nn::Tensor& x, double survival, nn::Generator& gen) {
  const nn::Tensor keep = nn::floor(nn::uniform({x.shape()[0], 1, 1, 1}, 0.0, 1.0, gen, x.dtype()) + survival);
  return x / survival * keep;
}
struct MBConvOutputs {
  nn::Tensor expanded, filtered, gate, projected, output;
};
inline MBConvOutputs MBConv(MobileContext& c, const std::string& name, const nn::Tensor& x, std::int64_t out,
                            const BlockArgs& b, int stride, bool swish, double drop_rate = 0) {
  auto act = [&](const nn::Tensor& h) { return swish ? Swish(h) : nn::relu(h); };
  MBConvOutputs      r;
  const std::int64_t in = x.shape()[3], hidden = in * b.expansion;
  const auto         none = nn::Activation::None;
  if (b.fused) {
    // Fused-MBConv: one strided kxk convolution replaces the 1x1 expansion and
    // the depthwise filter; without expansion it is the whole block.
    if (b.expansion == 1)
      r.expanded = r.filtered = x;
    else
      r.expanded = r.filtered = act(ConvBN(c, name + "/expand", x, hidden, b.kernel, stride, none));
  } else {
    r.expanded = b.expansion == 1 ? x : act(ConvBN(c, name + "/expand", x, hidden, 1, 1, none));
    r.filtered = act(ConvBN(c, name + "/depthwise", r.expanded, hidden, b.kernel, stride, none, MobileConv::kDepthwise));
  }
  nn::Tensor h = r.filtered;
  if (b.se_ratio > 0) {
    const std::int64_t   reduced = std::max<std::int64_t>(1, std::int64_t(in * b.se_ratio));
    nn::ConvLayerOptions o;
    o.init                   = nn::init::he_uniform();
    const nn::Tensor pooled   = nn::mean(h, {1, 2}, true);
    const nn::Tensor squeezed = act(nn::conv2d(c.p, name + "/se/reduce", pooled, reduced, 1, o));
    r.gate = nn::sigmoid(nn::conv2d(c.p, name + "/se/expand", squeezed, hidden, 1, o));
    h      = h * r.gate;
    c.layers.push_back({name + "/se", {1, 1, 1, 1}, 2 * hidden * reduced});
  }
  if (b.fused && b.expansion == 1)
    r.projected = act(ConvBN(c, name + "/project", h, out, b.kernel, stride, none));
  else
    r.projected = ConvBN(c, name + "/project", h, out, 1, 1, none);
  r.output = r.projected;
  if (b.skip && stride == 1 && in == out) {
    nn::Tensor branch = r.projected;
    if (c.training && drop_rate > 0) branch = DropConnect(branch, 1 - drop_rate, c.p.generator());
    r.output = x + branch;
  }
  return r;
}
inline std::vector<BlockArgs> MnasNetBlocks(const std::string& variant) {
  std::vector<std::string> text;
  if (variant == "a1")
    text = {"r1_k3_s11_e1_i32_o16_noskip",  "r2_k3_s22_e6_i16_o24",          "r3_k5_s22_e3_i24_o40_se0.25",
            "r4_k3_s22_e6_i40_o80",         "r2_k3_s11_e6_i80_o112_se0.25",  "r3_k5_s22_e6_i112_o160_se0.25",
            "r1_k3_s11_e6_i160_o320"};
  else if (variant == "b1")
    text = {"r1_k3_s11_e1_i32_o16_noskip", "r3_k3_s22_e3_i16_o24", "r3_k5_s22_e3_i24_o40",
            "r3_k5_s22_e6_i40_o80",        "r2_k3_s11_e6_i80_o96", "r4_k5_s22_e6_i96_o192",
            "r1_k3_s11_e6_i192_o320_noskip"};
  else
    throw std::invalid_argument("MnasNet variants are a1 and b1");
  std::vector<BlockArgs> blocks;
  for (const auto& t : text) blocks.push_back(DecodeBlock(t));
  return blocks;
}
struct MnasNetConfig {
  std::string variant    = "a1";
  double      multiplier = 1;
  int         classes    = 1000;
  bool        small_stem = false;
  double      keep_probability = .8, momentum = .9;
  bool        update_stats = true;
};
inline MobileNetOutputs MnasNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                const MnasNetConfig& cfg = {}, bool training = false) {
  if (!(cfg.multiplier > 0) || cfg.classes < 2 || !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid MnasNet configuration");
  MobileNetOutputs out;
  MobileContext    c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  int              in = RoundFilters(32, cfg.multiplier);
  nn::Tensor h = ConvBN(c, "stem", image, in, 3, cfg.small_stem ? 1 : 2, nn::Activation::Relu);
  int        index = 0;
  for (const auto& b : MnasNetBlocks(cfg.variant)) {
    const int o = RoundFilters(b.output, cfg.multiplier);
    for (int r = 0; r < b.repeats; ++r, in = o)
      h = MBConv(c, "block" + std::to_string(++index), h, o, b, r ? 1 : b.stride, false).output;
  }
  out.features      = ConvBN(c, "head", h, 1280, 1, 1, nn::Activation::Relu);
  nn::Tensor pooled = nn::global_avg_pool2d(out.features);
  if (training && cfg.keep_probability < 1) pooled = nn::dropout(pooled, 1 - cfg.keep_probability, true, p.generator());
  out.logits          = nn::dense(p, "output", pooled, cfg.classes);
  out.classifier_macs = 1280LL * cfg.classes;
  return out;
}

// EfficientNet-B0..B7 (https://arxiv.org/abs/1905.11946, TensorFlow TPU
// reference): the B0 block strings (all with SE 0.25 and swish), scaled by
// width (RoundFilters on every width, including the 32-channel stem and
// 1280-channel head) and depth (ceil(depth * repeats)); each variant's
// resolution and dropout come from the released table. Block i of n drops
// its residual branch with probability drop_connect * i / n in training.
struct EfficientNetScaling {
  double width, depth;
  int    resolution;
  double dropout;
};
inline EfficientNetScaling EfficientNetCoefficients(int variant) {
  static const EfficientNetScaling table[8] = {{1.0, 1.0, 224, .2}, {1.0, 1.1, 240, .2}, {1.1, 1.2, 260, .3},
                                               {1.2, 1.4, 300, .3}, {1.4, 1.8, 380, .4}, {1.6, 2.2, 456, .4},
                                               {1.8, 2.6, 528, .5}, {2.0, 3.1, 600, .5}};
  if (variant < 0 || variant > 7) throw std::invalid_argument("EfficientNet variants are B0 to B7");
  return table[variant];
}
inline std::vector<BlockArgs> EfficientNetBlocks(int variant) {
  const auto             scale = EfficientNetCoefficients(variant);
  std::vector<BlockArgs> blocks;
  for (const char* text : {"r1_k3_s11_e1_i32_o16_se0.25", "r2_k3_s22_e6_i16_o24_se0.25", "r2_k5_s22_e6_i24_o40_se0.25",
                           "r3_k3_s22_e6_i40_o80_se0.25", "r3_k5_s11_e6_i80_o112_se0.25", "r4_k5_s22_e6_i112_o192_se0.25",
                           "r1_k3_s11_e6_i192_o320_se0.25"}) {
    auto b    = DecodeBlock(text);
    b.input   = RoundFilters(b.input, scale.width);
    b.output  = RoundFilters(b.output, scale.width);
    b.repeats = int(std::ceil(scale.depth * b.repeats));
    blocks.push_back(b);
  }
  return blocks;
}
struct EfficientNetConfig {
  int    variant = 0, classes = 1000;
  bool   small_stem   = false;
  double drop_connect = .2, momentum = .9;
  bool   update_stats = true;
};
inline MobileNetOutputs EfficientNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                     const EfficientNetConfig& cfg = {}, bool training = false) {
  const auto scale  = EfficientNetCoefficients(cfg.variant);
  const auto blocks = EfficientNetBlocks(cfg.variant);
  if (cfg.classes < 2 || !(cfg.drop_connect >= 0 && cfg.drop_connect < 1))
    throw std::invalid_argument("invalid EfficientNet configuration");
  MobileNetOutputs out;
  MobileContext    c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  int              total = 0, index = 0;
  for (const auto& b : blocks) total += b.repeats;
  int        in = RoundFilters(32, scale.width);
  nn::Tensor h  = Swish(ConvBN(c, "stem", image, in, 3, cfg.small_stem ? 1 : 2, nn::Activation::None));
  for (const auto& b : blocks)
    for (int r = 0; r < b.repeats; ++r, ++index, in = b.output)
      h = MBConv(c, "block" + std::to_string(index + 1), h, b.output, b, r ? 1 : b.stride, true,
                 cfg.drop_connect * index / total)
              .output;
  const int head    = RoundFilters(1280, scale.width);
  out.features      = Swish(ConvBN(c, "head", h, head, 1, 1, nn::Activation::None));
  nn::Tensor pooled = nn::global_avg_pool2d(out.features);
  if (training) pooled = nn::dropout(pooled, scale.dropout, true, p.generator());
  out.logits          = nn::dense(p, "output", pooled, cfg.classes);
  out.classifier_macs = std::int64_t(head) * cfg.classes;
  return out;
}

// EfficientNetV2 (https://arxiv.org/abs/2104.00298, google/automl
// reference): Fused-MBConv early stages ("c1"), swish, SE 0.25 on MBConv
// stages, a stem as wide as the first block's input, a 1280-channel head and
// the same drop-connect schedule. S/M/L use their own block tables at width
// and depth one; the base table B0-B3 is scaled like EfficientNet.
struct EfficientNetV2Variant {
  std::vector<std::string> blocks;
  double                   width, depth;
  int                      resolution;
  double                   dropout;
};
inline EfficientNetV2Variant EfficientNetV2Spec(const std::string& name) {
  const std::vector<std::string> base = {"r1_k3_s1_e1_i32_o16_c1",     "r2_k3_s2_e4_i16_o32_c1",
                                         "r2_k3_s2_e4_i32_o48_c1",     "r3_k3_s2_e4_i48_o96_se0.25",
                                         "r5_k3_s1_e6_i96_o112_se0.25", "r8_k3_s2_e6_i112_o192_se0.25"};
  if (name == "s")
    return {{"r2_k3_s1_e1_i24_o24_c1", "r4_k3_s2_e4_i24_o48_c1", "r4_k3_s2_e4_i48_o64_c1", "r6_k3_s2_e4_i64_o128_se0.25",
             "r9_k3_s1_e6_i128_o160_se0.25", "r15_k3_s2_e6_i160_o256_se0.25"},
            1, 1, 384, .2};
  if (name == "m")
    return {{"r3_k3_s1_e1_i24_o24_c1", "r5_k3_s2_e4_i24_o48_c1", "r5_k3_s2_e4_i48_o80_c1", "r7_k3_s2_e4_i80_o160_se0.25",
             "r14_k3_s1_e6_i160_o176_se0.25", "r18_k3_s2_e6_i176_o304_se0.25", "r5_k3_s1_e6_i304_o512_se0.25"},
            1, 1, 480, .3};
  if (name == "l")
    return {{"r4_k3_s1_e1_i32_o32_c1", "r7_k3_s2_e4_i32_o64_c1", "r7_k3_s2_e4_i64_o96_c1", "r10_k3_s2_e4_i96_o192_se0.25",
             "r19_k3_s1_e6_i192_o224_se0.25", "r25_k3_s2_e6_i224_o384_se0.25", "r7_k3_s1_e6_i384_o640_se0.25"},
            1, 1, 480, .4};
  if (name == "b0") return {base, 1.0, 1.0, 224, .2};
  if (name == "b1") return {base, 1.0, 1.1, 240, .2};
  if (name == "b2") return {base, 1.1, 1.2, 260, .3};
  if (name == "b3") return {base, 1.2, 1.4, 300, .3};
  throw std::invalid_argument("EfficientNetV2 variants are s, m, l and b0-b3");
}
inline std::vector<BlockArgs> EfficientNetV2Blocks(const EfficientNetV2Variant& v) {
  std::vector<BlockArgs> blocks;
  for (const auto& text : v.blocks) {
    auto b    = DecodeBlock(text);
    b.input   = RoundFilters(b.input, v.width);
    b.output  = RoundFilters(b.output, v.width);
    b.repeats = int(std::ceil(v.depth * b.repeats));
    blocks.push_back(b);
  }
  return blocks;
}
struct EfficientNetV2Config {
  std::string variant = "s";
  int         classes = 1000;
  bool        small_stem   = false;
  double      drop_connect = .2, momentum = .9;
  bool        update_stats = true;
};
inline MobileNetOutputs EfficientNetV2(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                       const EfficientNetV2Config& cfg = {}, bool training = false) {
  const auto spec   = EfficientNetV2Spec(cfg.variant);
  const auto blocks = EfficientNetV2Blocks(spec);
  if (cfg.classes < 2 || !(cfg.drop_connect >= 0 && cfg.drop_connect < 1))
    throw std::invalid_argument("invalid EfficientNetV2 configuration");
  MobileNetOutputs out;
  MobileContext    c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  int              total = 0, index = 0;
  for (const auto& b : blocks) total += b.repeats;
  int        in = blocks.front().input;
  nn::Tensor h  = Swish(ConvBN(c, "stem", image, in, 3, cfg.small_stem ? 1 : 2, nn::Activation::None));
  for (const auto& b : blocks)
    for (int r = 0; r < b.repeats; ++r, ++index, in = b.output)
      h = MBConv(c, "block" + std::to_string(index + 1), h, b.output, b, r ? 1 : b.stride, true,
                 cfg.drop_connect * index / total)
              .output;
  const int head    = RoundFilters(1280, spec.width);
  out.features      = Swish(ConvBN(c, "head", h, head, 1, 1, nn::Activation::None));
  nn::Tensor pooled = nn::global_avg_pool2d(out.features);
  if (training) pooled = nn::dropout(pooled, spec.dropout, true, p.generator());
  out.logits          = nn::dense(p, "output", pooled, cfg.classes);
  out.classifier_macs = std::int64_t(head) * cfg.classes;
  return out;
}

}  // namespace mbconv
