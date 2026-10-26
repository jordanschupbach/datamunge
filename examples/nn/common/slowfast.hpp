// SlowFast R50 alpha8/beta1/8 and four temporal Fast-to-Slow fusions.
// Port of tflow common/nn/slowfast.h; explicit symmetric convolution padding,
// zero-final BN residual gates, and separately saved moving moments.
#pragma once
#include "common/factorized_video.hpp"
#include "common/tf_replay.hpp"
namespace slowfast {
namespace nn = datamunge::nn;
using nn::Tensor;
inline std::vector<int> Indices(int frames, int alpha = 8) {
  if (alpha < 1 || frames < 2 * alpha || frames % alpha)
    throw std::invalid_argument("SlowFast frame count divisible by alpha,at least two slow frames");
  int              count = frames / alpha;
  std::vector<int> ids;
  for (int i = 0; i < count; ++i)
    ids.push_back(int(std::int64_t(i) * (frames - 1) / (count - 1)));
  return ids;
}
inline Tensor Sample(const Tensor& rgb, int frames, int alpha = 8) {
  return nn::gather(rgb, nn::from_vector(Indices(frames, alpha)), 1);
}
inline Tensor Join(const Tensor& slow, const Tensor& lateral, double scale = 1) {
  return nn::concat({slow, lateral * scale}, 4);
}
struct Config {
  int    base = 64, classes = 400, frames = 32;
  double lateral_scale = 1;
};
struct Path {
  Tensor              logits, slow_input, slow_final, fast_final, pooled;
  std::vector<Tensor> slow_stages, fast_stages, slow_blocks, fast_blocks, lateral;
};
inline Tensor Unit(
    nn::ParameterStore& p,
    nn::ParameterStore& state,
    const std::string&  name,
    const Tensor&       x,
    int                 out,
    int                 kt,
    int                 kh,
    int                 kw,
    int                 st,
    int                 sh,
    int                 sw,
    bool                relu,
    bool                training,
    bool                zero_final,
    bool                update_stats) {
  p.get(
      name + "/W",
      {kt, kh, kw, x.size(4), out},
      nn::init::normal(std::sqrt(2. / (kt * kh * kw * out))));
  p.get(name + "/bn/gamma", {out}, nn::init::constant(zero_final ? 0 : 1));
  return factorized_video::FactorUnit(
      p, state, name, x, out, kt, kh, kw, st, sh, sw, relu, training, update_stats);
}
inline Path Forward(
    nn::ParameterStore& p,
    nn::ParameterStore& state,
    const Tensor&       rgb,
    const Config&       cfg          = {},
    bool                training     = false,
    bool                update_stats = false,
    nn::Generator*      dropout      = nullptr) {
  check::Require(
      cfg.base >= 8 && cfg.base % 8 == 0 && cfg.classes >= 2, "SlowFast width preserves beta1/8");
  Path out;
  out.slow_input = Sample(rgb, cfg.frames);
  auto unit      = [&](const std::string& name,
                  const Tensor&      x,
                  int                channels,
                  int                kt,
                  int                kh,
                  int                kw,
                  int                st   = 1,
                  int                sh   = 1,
                  int                sw   = 1,
                  bool               relu = true,
                  bool               zero = false) {
    return Unit(
        p,
        state,
        name,
        x,
        channels,
        kt,
        kh,
        kw,
        st,
        sh,
        sw,
        relu,
        training,
        zero,
        training && update_stats);
  };
  auto slow = out.slow_input * 2. - 1., fast = rgb * 2. - 1.;
  int  slow_in = cfg.base, fast_in = cfg.base / 8;
  slow      = unit("slow/stem", slow, slow_in, 1, 7, 7, 1, 2, 2);
  fast      = unit("fast/stem", fast, fast_in, 5, 7, 7, 1, 2, 2);
  auto pool = [&](const Tensor& x) {
    return nn::max_pool3d(
        nn::pad(x, {{0, 0}, {0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant, -INFINITY),
        {1, 3, 3},
        {1, 2, 2},
        nn::Padding::Valid);
  };
  slow = pool(slow);
  fast = pool(fast);
  out.slow_stages.push_back(slow);
  out.fast_stages.push_back(fast);
  auto fuse = [&](int i) {
    auto lateral = unit("fusion" + std::to_string(i), fast, 2 * fast_in, 5, 1, 1, 8);
    out.lateral.push_back(lateral);
    slow = Join(slow, lateral, cfg.lateral_scale);
    slow_in += 2 * fast_in;
  };
  fuse(0);
  const int depths[] = {3, 4, 6, 3};
  for (int stage = 0; stage < 4; ++stage) {
    for (int path = 0; path < 2; ++path) {
      auto& x        = path ? fast : slow;
      auto& incoming = path ? fast_in : slow_in;
      int   width = (cfg.base << stage) / (path ? 8 : 1), channels = width * 4,
          temporal = path || stage >= 2 ? 3 : 1;
      for (int block = 0; block < depths[stage]; ++block) {
        int  stride = stage && block == 0 ? 2 : 1;
        auto name   = std::string(path ? "fast" : "slow") + "/stage" + std::to_string(stage) +
                    "/block" + std::to_string(block);
        auto skip = x;
        auto y    = unit(name + "/reduce", x, width, temporal, 1, 1);
        y         = unit(name + "/spatial", y, width, 1, 3, 3, 1, stride, stride);
        y         = unit(name + "/expand", y, channels, 1, 1, 1, 1, 1, 1, false, true);
        if (incoming != channels || stride != 1)
          skip = unit(name + "/skip", x, channels, 1, 1, 1, 1, stride, stride, false);
        x        = nn::relu(y + skip);
        incoming = channels;
        (path ? out.fast_blocks : out.slow_blocks).push_back(x);
      }
    }
    out.slow_stages.push_back(slow);
    out.fast_stages.push_back(fast);
    if (stage < 3)
      fuse(stage + 1);
  }
  out.slow_final = slow;
  out.fast_final = fast;
  out.pooled     = nn::concat({nn::mean(slow, {1, 2, 3}), nn::mean(fast, {1, 2, 3})}, 1);
  auto feature   = out.pooled;
  if (training)
    feature = nn::dropout(feature, .5, true, *dropout);
  auto w     = p.get("head/W", {cfg.base * 36, cfg.classes}, nn::init::normal(.01));
  out.logits = nn::matmul(feature, w) + p.get("head/b", {cfg.classes}, nn::init::zeros());
  return out;
}
struct Model {
  nn::ParameterStore parameters{751}, state{753};
  nn::Generator      dropout{743};
  Model() {
    nn::NoGradGuard guard;
    Logits(nn::zeros({1, 16, 32, 32, 3}), false, false);
  }
  Tensor Logits(const Tensor& rgb, bool training, bool update_stats) {
    return Forward(parameters, state, rgb, {8, 3, 16}, training, update_stats, &dropout).logits;
  }
};

} // namespace slowfast
