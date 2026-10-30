#pragma once
/// @file slowfast_models.hpp
/// @brief Complete two-path R50 video graph with zero-gated residual and lateral units.
#include "datamunge/nn/factorized_video_models.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Choose endpoint-inclusive integer frame indices for the Slow pathway.
///
/// Index arithmetic is metadata;no pixel computation. At16 frames returns0,15;at32 returns0,10,20,31.
///
/// @param frames Positive time count divisible by alpha,at least two slow frames.
/// @param alpha Positive temporal rate ratio,default8.
/// @return Host index metadata floor(i*(frames-1)/(frames/alpha-1)).
/// @throws std::invalid_argument For invalid alpha/frame divisibility or too few Slow observations.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto ids = m::SlowFastIndices(32);
/// @endcode
/// @par References
/// - Feichtenhofer,Fan,Malik and He(2019),SlowFast Networks for Video Recognition, https://arxiv.org/abs/1812.03982. Source R50 alpha8/beta1/8 adaptation uses endpoint-inclusive integer sampling,explicit symmetric padding,zero-final fused BN gates and four time-strided concatenations. Compact16x32/base8/three-class examples are randomly initialized,not paper benchmarks.
/// @see SlowFast(),FactorUnit()
inline std::vector<int> SlowFastIndices(int frames, int alpha = 8) {
  if (alpha < 1 || frames < 2 * alpha || frames % alpha)
    throw std::invalid_argument(
        "SlowFast frame count divisible by alpha,at least two slow frames");
  int count = frames / alpha;
  std::vector<int> ids;
  for (int i = 0; i < count; ++i)
    ids.push_back(int(std::int64_t(i) * (frames - 1) / (count - 1)));
  return ids;
}
/// @brief Gather the actual Slow pathway observations with differentiable tensor indexing.
///
/// Gradients scatter only to selected original time positions. No pixel copying through host arrays.
///
/// @param rgb NDHWC tensor with exact frames time dimension.
/// @param frames Exact original time count.
/// @param alpha Positive temporal ratio,default8.
/// @return Tensor[N,frames/alpha,H,W,C],owning a live gather graph.
/// @throws std::invalid_argument For invalid sampling geometry or rank/time mismatch.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::SlowFastSample(n::ones({1, 16, 2, 2, 3}), 16);
/// @endcode
/// @par References
/// - Feichtenhofer,Fan,Malik and He(2019),SlowFast Networks for Video Recognition, https://arxiv.org/abs/1812.03982. Source R50 alpha8/beta1/8 adaptation uses endpoint-inclusive integer sampling,explicit symmetric padding,zero-final fused BN gates and four time-strided concatenations. Compact16x32/base8/three-class examples are randomly initialized,not paper benchmarks.
/// @see SlowFast(),FactorUnit()
inline Tensor SlowFastSample(const Tensor &rgb, int frames, int alpha = 8) {
  if (rgb.dim() != 5 || rgb.size(1) != frames)
    throw std::invalid_argument("NDHWC input time must match frame count");
  return nn::gather(rgb, nn::from_vector(SlowFastIndices(frames, alpha)), 1);
}
/// @brief Concatenate Slow and scaled lateral features along the channel axis.
///
/// Tensor concatenation preserves both input graphs and scales lateral adjoints.
///
/// @param slow Rank5 Slow tensor.
/// @param lateral Matching rank5 batch/time/spatial tensor.
/// @param scale Finite lateral multiplier,default1;0 is a declared ablation.
/// @return Rank5 tensor with both channel groups.
/// @throws std::invalid_argument For invalid ranks,nonfinite scale,or incompatible axes/dtypes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::SlowFastJoin(n::ones({1, 2, 1, 1, 3}), n::ones({1, 2, 1, 1, 2}));
/// @endcode
/// @par References
/// - Feichtenhofer,Fan,Malik and He(2019),SlowFast Networks for Video Recognition, https://arxiv.org/abs/1812.03982. Source R50 alpha8/beta1/8 adaptation uses endpoint-inclusive integer sampling,explicit symmetric padding,zero-final fused BN gates and four time-strided concatenations. Compact16x32/base8/three-class examples are randomly initialized,not paper benchmarks.
/// @see SlowFast(),FactorUnit()
inline Tensor SlowFastJoin(const Tensor &slow, const Tensor &lateral, double scale = 1) {
  if (slow.dim() != 5 || lateral.dim() != 5 || !std::isfinite(scale))
    throw std::invalid_argument("finite scale and rank5 pathway tensors required");
  return nn::concat({slow, lateral * scale}, 4);
}
/// @brief Configure full two-path R50 widths,time geometry and lateral transfer.
///
/// Full defaults32frames/base64/400classes;compact16/base8/3 retains
/// all32bottlenecks,110normalized convolutions,332learned and220moving leaves.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SlowFastConfig c{8, 3, 16};
/// @endcode
/// @par References
/// - Feichtenhofer,Fan,Malik and He(2019),SlowFast Networks for Video Recognition, https://arxiv.org/abs/1812.03982. Source R50 alpha8/beta1/8 adaptation uses endpoint-inclusive integer sampling,explicit symmetric padding,zero-final fused BN gates and four time-strided concatenations. Compact16x32/base8/three-class examples are randomly initialized,not paper benchmarks.
/// @see SlowFast(),FactorUnit()
struct SlowFastConfig {
  int base = 64;            ///< Slow stem width>=8,divisible8;Fast width base/8.
  int classes = 400;        ///< Categorical output count>=2.
  int frames = 32;          ///< Exact time count>=16,divisible8.
  double lateral_scale = 1; ///< Finite lateral multiplier;0 disables transfer as a control.
};
/// @brief Own all SlowFast pathway,block,fusion and head endpoints.
///
/// Owning tensor handles retain storage and the differentiable graph.
/// Separate caller stores own learned parameters and moving BN arrays.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(751), state(753);
/// const auto out = m::SlowFast(p, state, n::zeros({1, 16, 32, 32, 3}), {8, 3, 16});
/// @endcode
/// @par References
/// - Feichtenhofer,Fan,Malik and He(2019),SlowFast Networks for Video Recognition, https://arxiv.org/abs/1812.03982. Source R50 alpha8/beta1/8 adaptation uses endpoint-inclusive integer sampling,explicit symmetric padding,zero-final fused BN gates and four time-strided concatenations. Compact16x32/base8/three-class examples are randomly initialized,not paper benchmarks.
/// @see SlowFast(),FactorUnit()
struct SlowFastOutput {
  Tensor logits;                   ///< Raw[N,classes] categorical head scores.
  Tensor slow_input;               ///< Endpoint-inclusive time subsample[N,frames/8,H,W,3].
  Tensor slow_final;               ///< Last Slow residual endpoint,base*32 channels.
  Tensor fast_final;               ///< Last Fast residual endpoint,base*4 channels.
  Tensor pooled;                   ///< Concatenated spatial-temporal means[N,base*36].
  std::vector<Tensor> slow_stages; ///< Stem and four Slow stage endpoints before fusion.
  std::vector<Tensor> fast_stages; ///< Stem and four Fast stage endpoints.
  std::vector<Tensor> slow_blocks; ///< All16 Slow residual block endpoints.
  std::vector<Tensor> fast_blocks; ///< All16 Fast residual block endpoints.
  std::vector<Tensor> lateral; ///< Four temporal5/stride8 Fast-to-Slow normalized projections.
};
/// @brief Apply a symmetric3D convolution and fused BN with optional zero residual gate.
///
/// W uses fan-out normal sqrt(2/(kt*kh*kw*out)). Fused BN eps1e-5,
/// decay.9 uses population batch normalization and UNBIASED moving variance.
/// Zero gamma initially blocks upstream branch kernels while gamma/beta can learn.
/// Creates parameters/state;never performs an optimizer update.
///
/// @param p Mutable Float32 learned store,fan-out normal W and BN gamma/beta.
/// @param state Mutable Float32 fused-BN moving mean/variance store.
/// @param name Unique unit prefix.
/// @param x Float32 NDHWC unit input.
/// @param out Positive output channels.
/// @param kt Positive temporal kernel width.
/// @param kh Positive height kernel width.
/// @param kw Positive width kernel width.
/// @param st Positive temporal stride.
/// @param sh Positive height stride.
/// @param sw Positive width stride.
/// @param relu Apply final ReLU iftrue.
/// @param training Use batch BN moments iftrue;moving moments iffalse.
/// @param zero_final Initialize BN gamma0 iftrue,otherwise1;existing values unchanged.
/// @param update_stats Update moving arrays only when training&&true.
/// @return Normalized output,optionally ReLU-activated.
/// @throws std::invalid_argument For invalid geometry,dtype,or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p, state;
/// const auto y = m::SlowFastUnit(p, state, "u", n::ones({2, 2, 2, 2, 1}), 2, 1, 1, 1, 1, 1, 1,
///                                false, true, true, false);
/// @endcode
/// @par References
/// - Feichtenhofer,Fan,Malik and He(2019),SlowFast Networks for Video Recognition, https://arxiv.org/abs/1812.03982. Source R50 alpha8/beta1/8 adaptation uses endpoint-inclusive integer sampling,explicit symmetric padding,zero-final fused BN gates and four time-strided concatenations. Compact16x32/base8/three-class examples are randomly initialized,not paper benchmarks.
/// @see SlowFast(),FactorUnit()
inline Tensor SlowFastUnit(nn::ParameterStore &p, nn::ParameterStore &state,
                           const std::string &name, const Tensor &x, int out, int kt, int kh,
                           int kw, int st, int sh, int sw, bool relu, bool training,
                           bool zero_final, bool update_stats) {
  if (x.dim() != 5 || out < 1 || kt < 1 || kh < 1 || kw < 1 || st < 1 || sh < 1 || sw < 1)
    throw std::invalid_argument("positive rank5 SlowFast unit geometry required");
  p.get(name + "/W", {kt, kh, kw, x.size(4), out},
        nn::init::normal(std::sqrt(2. / (kt * kh * kw * out))));
  p.get(name + "/bn/gamma", {out}, nn::init::constant(zero_final ? 0 : 1));
  return FactorUnit(p, state, name, x, out, kt, kh, kw, st, sh, sw, relu, training,
                    update_stats);
}
/// @brief Classify genuine clips through both complete R50 paths and four lateral fusions.
///
/// Scale RGB*2-1;Slow uses endpoint-inclusive samples,Fast all observations.
/// Two3/4/6/3bottleneck paths,Fast width1/8,temporal5/stride8 lateral projections
/// at stem and stages0..2. All32expand BN gammas initialize0. Head normal.01/b0,
/// GAP concatenation and keep.5dropout. Full reference geometry is available;
/// compact tensor execution alone does not establish full benchmark accuracy.
/// Only lazy initialization,moving-stat updates and dropout RNG consumption mutate
/// caller state;loss and tensor optimizer loops belong to the calling example.
///
/// @param p Mutable Float32 learned store;warm all332leaves before optimizer construction.
/// @param state Mutable Float32 store for220moving arrays;save/restore paired with p.
/// @param rgb Float32[N,frames,H,W,3],Npositive,H/W>=32,normally[0,1].
/// @param cfg Full defaults32/base64/400;compact16/base8/3 retains every source block.
/// @param training Use batch BN and keep.5head dropout iftrue;moving moments otherwise.
/// @param update_stats Update moving arrays only when training&&true,defaultfalse.
/// @param dropout Required explicit mutable generator for training,unused at inference.
/// @return All pathway/block/fusion endpoints,pooled features and raw categorical logits.
/// @throws std::invalid_argument For invalid width/classes/time/rank/dtype/scale,missing training RNG or stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(751), state(753);
/// const auto out = m::SlowFast(p, state, n::zeros({1, 16, 32, 32, 3}), {8, 3, 16});
/// @endcode
/// @par References
/// - Feichtenhofer,Fan,Malik and He(2019),SlowFast Networks for Video Recognition, https://arxiv.org/abs/1812.03982. Source R50 alpha8/beta1/8 adaptation uses endpoint-inclusive integer sampling,explicit symmetric padding,zero-final fused BN gates and four time-strided concatenations. Compact16x32/base8/three-class examples are randomly initialized,not paper benchmarks.
/// @see SlowFast(),FactorUnit()
inline SlowFastOutput SlowFast(nn::ParameterStore &p, nn::ParameterStore &state,
                               const Tensor &rgb, const SlowFastConfig &cfg = {},
                               bool training = false, bool update_stats = false,
                               nn::Generator *dropout = nullptr) {
  if (cfg.base < 8 || cfg.base % 8 || cfg.classes < 2 || !std::isfinite(cfg.lateral_scale))
    throw std::invalid_argument(
        "finite SlowFast scale,base divisible8 and classes>=2 required");
  SlowFastIndices(cfg.frames);
  if (rgb.dim() != 5 || rgb.size(0) < 1 || rgb.size(1) != cfg.frames || rgb.size(2) < 32 ||
      rgb.size(3) < 32 || rgb.size(4) != 3 || rgb.dtype() != nn::DType::Float32 ||
      p.dtype() != rgb.dtype() || state.dtype() != rgb.dtype() || (training && !dropout))
    throw std::invalid_argument("Float32 NDHWC RGB,matching frames,spatial>=32 and explicit "
                                "training dropout required");
  SlowFastOutput out;
  out.slow_input = SlowFastSample(rgb, cfg.frames);
  auto unit = [&](const std::string &name, const Tensor &x, int channels, int kt, int kh,
                  int kw, int st = 1, int sh = 1, int sw = 1, bool relu = true,
                  bool zero = false) {
    return SlowFastUnit(p, state, name, x, channels, kt, kh, kw, st, sh, sw, relu, training,
                        zero, training && update_stats);
  };
  auto slow = out.slow_input * 2. - 1., fast = rgb * 2. - 1.;
  int slow_in = cfg.base, fast_in = cfg.base / 8;
  slow = unit("slow/stem", slow, slow_in, 1, 7, 7, 1, 2, 2);
  fast = unit("fast/stem", fast, fast_in, 5, 7, 7, 1, 2, 2);
  auto pool = [&](const Tensor &x) {
    return nn::max_pool3d(
        nn::pad(x, {{0, 0}, {0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant, -INFINITY),
        {1, 3, 3}, {1, 2, 2}, nn::Padding::Valid);
  };
  slow = pool(slow);
  fast = pool(fast);
  out.slow_stages.push_back(slow);
  out.fast_stages.push_back(fast);
  auto fuse = [&](int i) {
    auto lateral = unit("fusion" + std::to_string(i), fast, 2 * fast_in, 5, 1, 1, 8);
    out.lateral.push_back(lateral);
    slow = SlowFastJoin(slow, lateral, cfg.lateral_scale);
    slow_in += 2 * fast_in;
  };
  fuse(0);
  const int depths[] = {3, 4, 6, 3};
  for (int stage = 0; stage < 4; ++stage) {
    for (int path = 0; path < 2; ++path) {
      auto &x = path ? fast : slow;
      auto &incoming = path ? fast_in : slow_in;
      int width = (cfg.base << stage) / (path ? 8 : 1), channels = width * 4,
          temporal = path || stage >= 2 ? 3 : 1;
      for (int block = 0; block < depths[stage]; ++block) {
        int stride = stage && block == 0 ? 2 : 1;
        auto name = std::string(path ? "fast" : "slow") + "/stage" + std::to_string(stage) +
                    "/block" + std::to_string(block);
        auto skip = x;
        auto y = unit(name + "/reduce", x, width, temporal, 1, 1);
        y = unit(name + "/spatial", y, width, 1, 3, 3, 1, stride, stride);
        y = unit(name + "/expand", y, channels, 1, 1, 1, 1, 1, 1, false, true);
        if (incoming != channels || stride != 1)
          skip = unit(name + "/skip", x, channels, 1, 1, 1, 1, stride, stride, false);
        x = nn::relu(y + skip);
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
  out.pooled = nn::concat({nn::mean(slow, {1, 2, 3}), nn::mean(fast, {1, 2, 3})}, 1);
  auto feature = out.pooled;
  if (training)
    feature = nn::dropout(feature, .5, true, *dropout);
  auto w = p.get("head/W", {cfg.base * 36, cfg.classes}, nn::init::normal(.01));
  out.logits = nn::matmul(feature, w) + p.get("head/b", {cfg.classes}, nn::init::zeros());
  return out;
}
} // namespace datamunge::nn::models
