#pragma once
/// @file factorized_video_models.hpp
/// @brief Complete R(2+1)D-18/P3D-199 and differentiable symmetric-border factor primitives.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Choose the source parameter-balanced spatial/temporal intermediate width.
///
/// Integer architecture arithmetic only;3Dkernel weights balanced against
/// 1x3x3spatial plus3x1x1temporal,excluding BN.
///
/// @param in Incoming architecture channels1..1048576.
/// @param out Outgoing architecture channels1..1048576.
/// @return max(1,floor(27*in*out/(9*in+3*out))).
/// @throws std::invalid_argument For widths outside the supported architecture range.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const int middle = m::FactorMidplanes(64, 128);
/// @endcode
/// @par References
/// - Tran et al.(2018),A Closer Look at Spatiotemporal Convolutions for Action Recognition, https://arxiv.org/abs/1711.11248;Qiu,Yao and Mei(2017),Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks, https://arxiv.org/abs/1711.10305,authors released Caffe topology https://github.com/ZhaofanQiu/pseudo-3d-residual-networks. Source random/compact initialization and custom KTH fit are teaching adaptations,not pretrained benchmark results.
/// @see FactorVideo()
inline int FactorMidplanes(int in, int out) {
  if (in < 1 || out < 1 || in>1048576 || out>1048576)
    throw std::invalid_argument("positive factorized widths");
  return std::max(1, static_cast<int>(std::int64_t{27} * in * out /
                                      (std::int64_t{9} * in + std::int64_t{3} * out)));
}

/// @brief Convolve with explicit symmetric temporal and spatial borders.
///
/// Pad floor(kernel/2) on BOTH sides even at stride2,matching source
/// PyTorch/Caffe borders rather than TensorFlow asymmetric SAME. No fitting.
///
/// @param x Floating[N,T,H,W,C] input.
/// @param kernel Matching floating[kt,kh,kw,C,O],positive kernel axes.
/// @param st Positive temporal stride,default1.
/// @param sh Positive height stride,default1.
/// @param sw Positive width stride,default1.
/// @return Raw floating NDHWC convolution retaining input/kernel derivatives.
/// @throws std::invalid_argument For invalid shape,channels,dtype or stride.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::FactorConv(n::ones({1, 4, 4, 4, 1}), n::ones({3, 3, 3, 1, 2}), 2, 2, 2);
/// @endcode
/// @par References
/// - Tran et al.(2018),A Closer Look at Spatiotemporal Convolutions for Action Recognition, https://arxiv.org/abs/1711.11248;Qiu,Yao and Mei(2017),Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks, https://arxiv.org/abs/1711.10305,authors released Caffe topology https://github.com/ZhaofanQiu/pseudo-3d-residual-networks. Source random/compact initialization and custom KTH fit are teaching adaptations,not pretrained benchmark results.
/// @see FactorVideo()
inline Tensor FactorConv(const Tensor &x, const Tensor &kernel, int st = 1, int sh = 1,
                         int sw = 1) {
  if (x.dim() != 5 || kernel.dim() != 5 || x.size(4) != kernel.size(3) || st < 1 || sh < 1 ||
      sw < 1 ||
      std::any_of(
          kernel.shape().begin(), kernel.shape().end(),
          [](auto n) {
            return n < 1;
          }))
    throw std::invalid_argument(
        "valid NDHWC kernel/channel geometry and positive strides required");
  const auto &k = kernel.shape();
  nn::Conv3DOptions o;
  o.strides = {st, sh, sw};
  o.padding = nn::Padding::Explicit;
  o.explicit_paddings = {k[0] / 2, k[0] / 2, k[1] / 2, k[1] / 2, k[2] / 2, k[2] / 2};
  return nn::conv3d(x, kernel, o);
}

/// @brief Select the source released serial,parallel or averaged-shortcut P3D mix.
///
/// Callbacks include their own normalization/activation. C uses the released
/// source0.5average,explicitly distinguished from an unscaled sum.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto kind = m::P3dKind::C;
/// @endcode
/// @par References
/// - Tran et al.(2018),A Closer Look at Spatiotemporal Convolutions for Action Recognition, https://arxiv.org/abs/1711.11248;Qiu,Yao and Mei(2017),Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks, https://arxiv.org/abs/1711.10305,authors released Caffe topology https://github.com/ZhaofanQiu/pseudo-3d-residual-networks. Source random/compact initialization and custom KTH fit are teaching adaptations,not pretrained benchmark results.
/// @see FactorVideo()
enum class P3dKind {
  A, ///< Serial temporal(spatial(x)),no extra ReLU.
  B, ///< Parallel ReLU(spatial(x)+temporal(x)).
  C  ///< Shortcut ReLU(.5*(spatial(x)+temporal(spatial(x)))).
};

/// @brief Combine differentiable spatial and temporal callbacks with the source P3D topology.
///
/// Does not own or fit parameters;callbacks may lazily create learned leaves
/// or update caller state according to their own contracts.
///
/// @param x Floating callback input.
/// @param spatial Callable spatial operator,including its normalization/activation.
/// @param temporal Callable temporal operator,including its normalization/activation.
/// @param kind A serial,Bparallel,Caveragedshortcut.
/// @return Serial output or ReLU of the specified sum/average,with callback graphs retained.
/// @throws std::invalid_argument For missing callbacks,invalid enum or incompatible callback shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::tensor({-2., 1., 3.});
/// const auto y = m::P3dMix(
///     x,
///     [](const auto &v) {
///       return v * 2.;
///     },
///     [](const auto &v) {
///       return v + 1.;
///     },
///     m::P3dKind::C);
/// @endcode
/// @par References
/// - Tran et al.(2018),A Closer Look at Spatiotemporal Convolutions for Action Recognition, https://arxiv.org/abs/1711.11248;Qiu,Yao and Mei(2017),Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks, https://arxiv.org/abs/1711.10305,authors released Caffe topology https://github.com/ZhaofanQiu/pseudo-3d-residual-networks. Source random/compact initialization and custom KTH fit are teaching adaptations,not pretrained benchmark results.
/// @see FactorVideo()
inline Tensor P3dMix(const Tensor &x, const std::function<Tensor(const Tensor &)> &spatial,
                     const std::function<Tensor(const Tensor &)> &temporal, P3dKind kind) {
  if (!spatial || !temporal ||
      (kind != P3dKind::A && kind != P3dKind::B && kind != P3dKind::C))
    throw std::invalid_argument("valid P3D kind and callbacks required");
  const Tensor a = spatial(x);
  const Tensor b = temporal(kind == P3dKind::B ? x : a);
  if (kind == P3dKind::A)
    return b;
  Tensor combined = a + b;
  if (kind == P3dKind::C)
    combined = combined * .5;
  return nn::relu(combined);
}

/// @brief Configure full-depth R(2+1)D-18 or P3D-199.
///
/// Width scaling preserves8basic or50bottleneck blocks. P3D stages3/8/36/3
/// with the final stage spatial-only;not a shortened surrogate.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::FactorVideoConfig c{true, 4, 3};
/// @endcode
/// @par References
/// - Tran et al.(2018),A Closer Look at Spatiotemporal Convolutions for Action Recognition, https://arxiv.org/abs/1711.11248;Qiu,Yao and Mei(2017),Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks, https://arxiv.org/abs/1711.10305,authors released Caffe topology https://github.com/ZhaofanQiu/pseudo-3d-residual-networks. Source random/compact initialization and custom KTH fit are teaching adaptations,not pretrained benchmark results.
/// @see FactorVideo()
struct FactorVideoConfig {
  bool p3d = false;  ///< FalseR(2+1)D-18,trueP3D-199.
  int base = 64;     ///< Base width>=2;compact4.
  int classes = 400; ///< Head classes>=2;compact3.
};

/// @brief Own full-depth factorized video logits,pooled features and residual endpoints.
///
/// Handles retain graph/storage;BN moving state lives in the separate caller store.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(727), state(729);
/// const auto out = m::FactorVideo(p, state, n::zeros({1, 16, 32, 32, 3}), {false, 4, 3});
/// @endcode
/// @par References
/// - Tran et al.(2018),A Closer Look at Spatiotemporal Convolutions for Action Recognition, https://arxiv.org/abs/1711.11248;Qiu,Yao and Mei(2017),Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks, https://arxiv.org/abs/1711.10305,authors released Caffe topology https://github.com/ZhaofanQiu/pseudo-3d-residual-networks. Source random/compact initialization and custom KTH fit are teaching adaptations,not pretrained benchmark results.
/// @see FactorVideo()
struct FactorVideoOutput {
  Tensor logits;              ///< Raw[N,classes] head scores.
  Tensor pooled;              ///< Global time/height/width mean before optional P3D dropout.
  std::vector<Tensor> stages; ///< Stem and every residual block endpoint,9or51 tensors.
};

/// @brief Apply source fan-out Kaiming convolution and explicit-state batch normalization.
///
/// No convolution bias. Normal kernel std=sqrt(2/(kt*kh*kw*out)),
/// BN eps1e-5,momentum.9,gamma1/beta0,moving mean0/variance1.
/// Training uses population variance;the library fused-BN moving variance
/// applies its unbiased sample correction. Lazy creation is not optimization.
/// Inference must not update moving state.
///
/// @param p Mutable learned W/gamma/beta store.
/// @param state Mutable moving mean/variance store,same dtype as p.
/// @param name Prefix;W plus/bn learned/state names.
/// @param x FloatingNDHWC,dtype matching stores.
/// @param out Positive output channels.
/// @param kt Positive temporal taps.
/// @param kh Positive height taps.
/// @param kw Positive width taps.
/// @param st Positive temporal stride.
/// @param sh Positive height stride.
/// @param sw Positive width stride.
/// @param relu Enable output ReLU.
/// @param training Use batch moments iftrue,moving moments otherwise.
/// @param update_stats Update moving state only when training&&true,caller model enforces this.
/// @return Normalized optional-ReLU convolution graph.
/// @throws std::invalid_argument For invalid shape,width/kernel/strides,dtype or stored tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(727), s(729);
/// const auto y = m::FactorUnit(p, s, "u", n::ones({2, 4, 4, 4, 2}), 3, 1, 3, 3, 1, 1, 1, true,
///                              false, false);
/// @endcode
/// @par References
/// - Tran et al.(2018),A Closer Look at Spatiotemporal Convolutions for Action Recognition, https://arxiv.org/abs/1711.11248;Qiu,Yao and Mei(2017),Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks, https://arxiv.org/abs/1711.10305,authors released Caffe topology https://github.com/ZhaofanQiu/pseudo-3d-residual-networks. Source random/compact initialization and custom KTH fit are teaching adaptations,not pretrained benchmark results.
/// @see FactorVideo()
inline Tensor FactorUnit(nn::ParameterStore &p, nn::ParameterStore &state,
                         const std::string &name, const Tensor &x, int out, int kt, int kh,
                         int kw, int st, int sh, int sw, bool relu, bool training,
                         bool update_stats) {
  if (x.dim() != 5 || out < 1 || kt < 1 || kh < 1 || kw < 1)
    throw std::invalid_argument("positive factorized unit shape/width/kernel required");
  const std::int64_t in = x.shape().back();
  const Tensor w = p.get(name + "/W", {kt, kh, kw, in, out},
                         nn::init::normal(std::sqrt(2.0 / (kt * kh * kw * out))));
  const Tensor y = FactorConv(x, w, st, sh, sw);
  nn::BatchNormOptions bn;
  bn.momentum = .9;
  bn.eps = 1e-5;
  bn.update_stats = update_stats;
  const Tensor normalized = nn::batch_norm(p, state, name + "/bn", y, training, bn);
  return relu ? nn::relu(normalized) : normalized;
}

/// @brief Classify RGB clips with complete R(2+1)D-18 or P3D-199 residual topology.
///
/// RGB*2-1. R(2+1)D stem spatial7/temporal3,2/2/2/2basic blocks,
/// parameter-balanced spatial3/temporal3 factors,symmetric stride2 borders.
/// P3D stem spatial7+ceilpool,3/8/36/3bottleneck blocks,cyclic A/B/C in
/// first3stages,final spatial-only stage and temporal pooling between stages.
/// All units have source fan-out Kaiming/BN. HeadN(0,.01),bias0. No pretraining,
/// optimizer or hidden state calibration. Restore learned AND moving arrays.
///
/// @param p Mutable learned store;warm every leaf before constructing the optimizer.
/// @param state Mutable paired BN moving store;checkpoint together with p.
/// @param rgb Floating[N,T,H,W,3],T>=16,H/W>=32,positiveN,dtype matching stores.
/// @param cfg Architecture,base>=2/classes>=2;fullR(2+1)Ddefaults.
/// @param training False uses moving moments;true uses batch moments and P3D keep.5head dropout.
/// @param update_stats False by default;only training&&true changes moving moments.
/// @param dropout_gen Required mutable generator only for P3D training.
/// @return Raw head scores,global features,and all9R(2+1)D or51P3D endpoints.
/// @throws std::invalid_argument For invalid architecture,geometry,dtype,dropout or stored tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(727), state(729);
/// const auto out = m::FactorVideo(p, state, n::zeros({1, 16, 32, 32, 3}), {false, 4, 3});
/// @endcode
/// @par References
/// - Tran et al.(2018),A Closer Look at Spatiotemporal Convolutions for Action Recognition, https://arxiv.org/abs/1711.11248;Qiu,Yao and Mei(2017),Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks, https://arxiv.org/abs/1711.10305,authors released Caffe topology https://github.com/ZhaofanQiu/pseudo-3d-residual-networks. Source random/compact initialization and custom KTH fit are teaching adaptations,not pretrained benchmark results.
/// @see FactorVideo()
inline FactorVideoOutput FactorVideo(nn::ParameterStore &p, nn::ParameterStore &state,
                                     const Tensor &rgb, const FactorVideoConfig &cfg = {},
                                     bool training = false, bool update_stats = false,
                                     nn::Generator *dropout_gen = nullptr) {
  if (cfg.base < 2 || cfg.classes < 2)
    throw std::invalid_argument("factorized video configuration");
  if (rgb.dim() != 5 || rgb.size(0) < 1 || rgb.size(1) < 16 || rgb.size(2) < 32 ||
      rgb.size(3) < 32 || rgb.size(4) != 3 || rgb.dtype() != p.dtype() ||
      state.dtype() != p.dtype() || (cfg.p3d && training && !dropout_gen))
    throw std::invalid_argument(
        "positive RGB video geometry,matching stores and explicit P3D dropout required");
  FactorVideoOutput result;
  const auto unit = [&](const std::string &name, const Tensor &x, int out, int kt, int kh,
                        int kw, int st, int sh, int sw, bool relu = true) {
    return FactorUnit(p, state, name, x, out, kt, kh, kw, st, sh, sw, relu, training,
                      training && update_stats);
  };
  Tensor x = rgb * 2.0 - 1.0;
  int incoming = cfg.base;
  if (!cfg.p3d) {
    const int middle = std::max(1, 45 * cfg.base / 64);
    x = unit("stem/spatial", x, middle, 1, 7, 7, 1, 2, 2);
    x = unit("stem/temporal", x, cfg.base, 3, 1, 1, 1, 1, 1);
    result.stages.push_back(x);
    for (int stage = 0; stage < 4; ++stage)
      for (int block = 0; block < 2; ++block) {
        const int out = cfg.base << stage, stride = stage && block == 0 ? 2 : 1,
                  mid = FactorMidplanes(incoming, out);
        const std::string name =
            "stage" + std::to_string(stage) + "/block" + std::to_string(block);
        Tensor skip = x;
        Tensor y = unit(name + "/first/spatial", x, mid, 1, 3, 3, 1, stride, stride);
        y = unit(name + "/first/temporal", y, out, 3, 1, 1, stride, 1, 1);
        y = unit(name + "/second/spatial", y, mid, 1, 3, 3, 1, 1, 1);
        y = unit(name + "/second/temporal", y, out, 3, 1, 1, 1, 1, 1, false);
        if (stride != 1 || incoming != out)
          skip = unit(name + "/skip", x, out, 1, 1, 1, stride, stride, stride, false);
        x = nn::relu(y + skip);
        incoming = out;
        result.stages.push_back(x);
      }
  } else {
    x = unit("stem", x, cfg.base, 1, 7, 7, 1, 2, 2);
    // Caffe CEIL spatial 3/stride 2 pool from an even stem grid: partial end window.
    x = nn::pad(x, {{0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 0}}, nn::PadMode::Constant,
                -std::numeric_limits<double>::infinity());
    x = nn::max_pool3d(x, {2, 3, 3}, {2, 2, 2}, nn::Padding::Valid);
    result.stages.push_back(x);
    const int depth[] = {3, 8, 36, 3};
    int index = 0;
    for (int stage = 0; stage < 4; ++stage) {
      if (stage)
        x = nn::max_pool3d(x, {2, 1, 1}, {2, 1, 1}, nn::Padding::Valid);
      for (int block = 0; block < depth[stage]; ++block, ++index) {
        const int width = cfg.base << stage, out = 4 * width,
                  stride = stage && block == 0 ? 2 : 1;
        const std::string name =
            "stage" + std::to_string(stage) + "/block" + std::to_string(block);
        Tensor skip = x;
        Tensor y = unit(name + "/reduce", x, width, 1, 1, 1, 1, stride, stride);
        if (stage < 3) {
          const auto kind = static_cast<P3dKind>(index % 3);
          const auto spatial = [&](const Tensor &v) {
            return unit(name + "/spatial", v, width, 1, 3, 3, 1, 1, 1);
          };
          const auto temporal = [&](const Tensor &v) {
            return unit(name + "/temporal", v, width, 3, 1, 1, 1, 1, 1, kind == P3dKind::A);
          };
          y = P3dMix(y, spatial, temporal, kind);
        } else {
          y = unit(name + "/spatial", y, width, 1, 3, 3, 1, 1,
                   1); // T = 1: the authors' final 2D stage.
        }
        y = unit(name + "/expand", y, out, 1, 1, 1, 1, 1, 1, false);
        if (incoming != out || stride != 1)
          skip = unit(name + "/skip", x, out, 1, 1, 1, 1, stride, stride, false);
        x = nn::relu(skip + y);
        incoming = out;
        result.stages.push_back(x);
      }
    }
  }
  result.pooled = nn::mean(x, {1, 2, 3});
  x = result.pooled;
  if (cfg.p3d && training)
    x = nn::dropout(x, .5, true, *dropout_gen);
  const Tensor w = p.get("head/W", {incoming, cfg.classes}, nn::init::normal(.01));
  const Tensor b = p.get("head/b", {cfg.classes}, nn::init::zeros());
  result.logits = nn::matmul(x, w) + b;
  return result;
}

} // namespace datamunge::nn::models
