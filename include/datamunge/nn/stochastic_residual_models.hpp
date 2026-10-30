#pragma once
/// @file stochastic_residual_models.hpp
/// @brief Original/inverted DropPath and complete CIFAR residual tensor forward model.
#include "datamunge/nn/resnet_models.hpp"
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Choose original or inverted stochastic residual scaling.
///
/// Original training uses b*F and inference p*F; inverted training uses b*F/p
/// and inference F. At p=0, training outputs zero under both conventions and
/// inference inverted still returns F; no division by zero occurs.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::Generator gen(71);
/// const auto y = n::models::DropPath(n::ones({4, 3}), .5, true, true,
///                                    n::models::DropPathScaling::kInverted, gen);
/// @endcode
/// @par References
/// - Huang et al. (2016), https://arxiv.org/abs/1603.09382; inverted convention https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py
/// @see DropPath()
enum class DropPathScaling {
  kOriginal, ///< Bernoulli gate in training, survival factor in inference.
  kInverted  ///< Gate/survival in training when survival>0, unity in inference.
};

/// @brief Return the linear survival schedule p_l=1-l/L*(1-p_last).
///
/// Architecture metadata only, not sampled masks. Sum equals
/// L-(1-last)*(L+1)/2; for L=54,last=.5, expected active blocks is 40.25.
/// This expectation counts nonzero masks, not executed blocks. O(L) work/storage.
///
/// @param blocks Positive number L of residual blocks, excluding the stem/head.
/// @param last Final survival probability in [0,1], default chosen by caller.
/// @return Float vector of length blocks, indexed l-1 for l=1..L.
/// @throws std::invalid_argument For nonpositive blocks or invalid/nonfinite probability.
/// @par Example
/// @code
/// const auto p = dm::nn::models::SurvivalProbabilities(54, .5f);
/// @endcode
/// @par References
/// - Huang et al. (2016), https://arxiv.org/abs/1603.09382; inverted convention https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py
/// @see StochasticCifarResNet()
inline std::vector<float> SurvivalProbabilities(int blocks, float last) {
  if (blocks < 1 || !(last >= 0 && last <= 1))
    throw std::invalid_argument("invalid survival probability schedule");
  std::vector<float> probabilities(static_cast<std::size_t>(blocks));
  for (int i = 0; i < blocks; ++i)
    probabilities[static_cast<std::size_t>(i)] = 1 - float(i + 1) / blocks * (1 - last);
  return probabilities;
}

/// @brief Masked branch, sampled gate and broadcast scaling multiplier.
///
/// Value retains the branch graph; gate and multiplier are nondifferentiable
/// Float32 constants broadcast over branch shape. Handles own storage references.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::Generator gen(71);
/// const auto y = n::models::DropPath(n::ones({4, 3}), .5, true, true,
///                                    n::models::DropPathScaling::kInverted, gen);
/// @endcode
/// @see DropPath()
struct DropPathOutputs {
  nn::Tensor value; ///< branch*multiplier, same broadcast shape, retaining branch derivatives.
  nn::Tensor gate;  ///< Bernoulli scalar or [N,1,...,1] in training; scalar one in inference.
  nn::Tensor multiplier; ///< Gate, gate/p, p or one according to mode/scaling.
};
/// @brief Mask an already-computed residual with a batch or per-example Bernoulli gate.
///
/// Computes masking only, with no conditional execution or optimizer. Original
/// expectation is pF; inverted expectation is F when p>0. At p=0 both training
/// conventions return zero, but inverted inference returns F. Gates are constants
/// and are not reparameterized; input/weight derivatives follow the multiplier.
/// No mutation of branch; O(numel(branch)) work/storage. Caller synchronizes gen.
///
/// @param branch Nonempty Float32/64 tensor; per_example requires rank at least one.
/// @param survival Keep probability p in [0,1], including deterministic zero/one endpoints.
/// @param training True samples a mask except at endpoints; false applies deterministic scaling.
/// @param per_example True shares one gate per first-axis row; false shares one scalar gate.
/// @param scaling Original bF/train,pF/test or inverted bF/p/train,F/test.
/// @param gen Caller-owned random generator, advanced only for interior training probabilities.
/// @return Branch value, binary gate and broadcast multiplier with branch autograd retained.
/// @throws std::invalid_argument For invalid dtype, shape, scaling enum or probability.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::Generator gen(71);
/// const auto y = n::models::DropPath(n::ones({4, 3}), .5, true, true,
///                                    n::models::DropPathScaling::kInverted, gen);
/// @endcode
/// @par References
/// - Huang et al. (2016), https://arxiv.org/abs/1603.09382; inverted convention https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py
/// @see DropPathScaling, SurvivalProbabilities()
inline DropPathOutputs DropPath(const nn::Tensor &branch, double survival, bool training,
                                bool per_example, DropPathScaling scaling,
                                nn::Generator &gen) {
  if (!branch.defined() || branch.numel() == 0 ||
      (branch.dtype() != nn::DType::Float32 && branch.dtype() != nn::DType::Float64) ||
      (per_example && branch.dim() < 1) ||
      (scaling != DropPathScaling::kOriginal && scaling != DropPathScaling::kInverted))
    throw std::invalid_argument("DropPath requires floating nonempty input, valid scaling and "
                                "a batch axis for row masks");
  if (!(survival >= 0 && survival <= 1))
    throw std::invalid_argument("survival probability must be in [0,1]");
  DropPathOutputs out;
  if (!training) {
    out.gate = nn::scalar(1);
    out.multiplier = nn::scalar(scaling == DropPathScaling::kOriginal ? survival : 1.);
  } else {
    if (survival == 0 || survival == 1)
      out.gate = nn::scalar(survival);
    else {
      nn::Shape shape;
      if (per_example) {
        shape.assign(static_cast<std::size_t>(branch.dim()), 1);
        shape[0] = branch.shape()[0];
      }
      out.gate = nn::bernoulli(shape, survival, gen);
    }
    out.multiplier =
        scaling == DropPathScaling::kInverted && survival > 0 ? out.gate / survival : out.gate;
  }
  out.value = branch * out.multiplier;
  return out;
}

/// @brief Apply parameter-free CIFAR option-A pooling and trailing zero channels.
///
/// Raw values have no activation; stride-two derivatives spread with weight .25.
/// No learned state, fitting or generator; work/storage O(NHW*out).
///
/// @param input Nonempty floating NHWC tensor, even spatial sides at stride two.
/// @param out Exactly Cin or 2*Cin channels.
/// @param stride One or two, using 2x2 VALID average pooling when two.
/// @return Signed shortcut with unchanged or halved sides, followed by zero channels when doubled.
/// @throws std::invalid_argument For invalid rank, width, stride or odd downsampled sides.
/// @par Example
/// @code
/// const auto y = dm::nn::models::CifarIdentityShortcut(dm::nn::ones({1, 4, 4, 2}), 4, 2);
/// @endcode
/// @par References
/// - Huang et al. (2016), https://arxiv.org/abs/1603.09382; inverted convention https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py
/// @see StochasticCifarResNet()
inline nn::Tensor CifarIdentityShortcut(const nn::Tensor &input, std::int64_t out,
                                        int stride) {
  if (input.dim() != 4 || input.numel() == 0 ||
      (stride == 2 && (input.size(1) % 2 || input.size(2) % 2)))
    throw std::invalid_argument(
        "CIFAR shortcut requires nonempty NHWC and even strided sides");
  const std::int64_t in = input.shape().back();
  if ((stride != 1 && stride != 2) || in < 1 || (out != in && out != 2 * in))
    throw std::invalid_argument("invalid CIFAR option-A shortcut");
  nn::Tensor h = stride == 1 ? input : nn::avg_pool2d(input, 2, 2, nn::Padding::Valid);
  if (out != in)
    h = nn::concat({h, nn::zeros_like(h)}, 3);
  return h;
}

/// @brief CIFAR residual depth, widths and linear DropPath survival settings.
///
/// Defaults use depth 110, width 16, whole-batch masks and final survival .5.
/// Value metadata only; no parameter allocation or fitting.
///
/// @par Example
/// @code
/// dm::nn::models::StochasticCifarConfig cfg;
/// cfg.Validate();
/// @endcode
/// @par References
/// - Huang et al. (2016), https://arxiv.org/abs/1603.09382; inverted convention https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py
/// @see StochasticCifarResNet()
struct StochasticCifarConfig {
  int depth = 110;           ///< Depth 6n+2 with n>=1, three equal n-block stages.
  int base_width = 16;       ///< Positive stem/first-stage width; later stages double it.
  int classes = 10;          ///< At least two output classes.
  double last_survival = .5; ///< Last block survival in [0,1], linear schedule from near one.
  double momentum = .9;      ///< BN moving-moment retention in [0,1).
  bool per_example =
      false; ///< Independent first-axis row masks instead of one whole-batch gate.
  bool update_stats =
      true; ///< Permit BN moving updates during training, including masked branches.

  /// @brief Validate 6n+2 depth, dimensions and survival/BN probabilities.
  ///
  /// Checks value metadata without allocating learned state.
  ///
  /// @throws std::invalid_argument For invalid depth, width, classes, survival or momentum.
  /// @par Example
  /// @code
  /// dm::nn::models::StochasticCifarConfig{}.Validate();
  /// @endcode
  /// @see StochasticCifarConfig
  void Validate() const {
    if (depth < 8 || (depth - 2) % 6 || base_width < 1 || classes < 2 ||
        !(last_survival >= 0 && last_survival <= 1) || !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid 6n+2 stochastic CIFAR model");
  }
};
/// @brief CIFAR classifier logits, stage activations and ordered DropPath gates.
///
/// Output handles retain storage and autograd graphs; stores own learned/state leaves.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::Generator gen(71);
/// n::models::StochasticCifarConfig cfg;
/// cfg.depth = 8;
/// cfg.base_width = 4;
/// const auto y =
///     n::models::StochasticCifarResNet(p, state, n::ones({2, 8, 8, 3}), cfg, false, gen);
/// @endcode
/// @see StochasticCifarResNet()
struct StochasticCifarOutputs {
  nn::Tensor logits;              ///< Unnormalised [N,classes] global-pool/dense scores.
  std::vector<nn::Tensor> stages; ///< Three post-addition ReLU stage ends.
  std::vector<nn::Tensor>
      gates; ///< Every sampled gate in execution order, or scalar ones in inference.
};
/// @brief Build a complete CIFAR residual classifier with inverted stochastic-depth masking.
///
/// Bias-free He-uniform 3x3 stem/BN/ReLU; three stages of two-filter units.
/// Branches use conv/BN/ReLU then conv/BN, inverted DropPath, shortcut addition
/// and ReLU. Later stage entries pool/zero-pad option-A shortcuts. Final global
/// mean and Glorot dense classifier. BN learns scale/shift, epsilon 1e-5.
/// EVERY branch is evaluated and BN moments update even when its mask is zero.
/// This implements regularization, not conditional-execution speedup. One random
/// stream replaces per-block source streams. No preprocessing, optimizer or fitting
/// is hidden. Work does not decrease with survival; storage includes graph tensors.
///
/// @param p Mutable learnable store, lazily creates/reuses all model leaves.
/// @param state Mutable BN moving moments matching p dtype.
/// @param image Nonempty floating NHWC images, spatial sides divisible by four.
/// @param cfg Validated depth, width and survival settings.
/// @param training True uses batch BN moments and random gates; false uses saved moments and all branches.
/// @param gen Caller-owned mask generator, advanced only for interior training probabilities.
/// @return Three stage ends, every gate and classifier logits with gradients.
/// @throws std::invalid_argument For invalid configuration, image geometry, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::Generator gen(71);
/// n::models::StochasticCifarConfig cfg;
/// cfg.depth = 8;
/// cfg.base_width = 4;
/// const auto y =
///     n::models::StochasticCifarResNet(p, state, n::ones({2, 8, 8, 3}), cfg, false, gen);
/// @endcode
/// @par References
/// - Huang et al. (2016), https://arxiv.org/abs/1603.09382; inverted convention https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py
/// @see DropPath(), StochasticCifarConfig
inline StochasticCifarOutputs StochasticCifarResNet(nn::ParameterStore &p,
                                                    nn::ParameterStore &state,
                                                    const nn::Tensor &image,
                                                    const StochasticCifarConfig &cfg,
                                                    bool training, nn::Generator &gen) {
  cfg.Validate();
  if (image.dim() != 4 || image.numel() == 0 || image.size(1) % 4 || image.size(2) % 4)
    throw std::invalid_argument(
        "stochastic CIFAR images require nonempty NHWC and spatial sides divisible by four");
  StochasticCifarOutputs out;
  ResNetNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.epsilon = 1e-5;
  norm.momentum = cfg.momentum;
  norm.update_stats = cfg.update_stats;
  const int repetitions = (cfg.depth - 2) / 6;
  const auto survival = SurvivalProbabilities(3 * repetitions, float(cfg.last_survival));
  int index = 0;
  auto h = ResNetConv(p, state, "stem", image, cfg.base_width, 3, norm);
  for (int stage = 0; stage < 3; ++stage) {
    const std::int64_t channels = std::int64_t(cfg.base_width) << stage;
    for (int block = 0; block < repetitions; ++block, ++index) {
      const int stride = stage > 0 && block == 0 ? 2 : 1;
      const auto name =
          "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      const auto skip = CifarIdentityShortcut(h, channels, stride);
      auto residual = ResNetConv(p, state, name + "/conv1", h, channels, 3, norm, stride);
      residual = ResNetConv(p, state, name + "/conv2", residual, channels, 3, norm, 1, false);
      const auto dropped =
          DropPath(residual, survival[static_cast<std::size_t>(index)], training,
                   cfg.per_example, DropPathScaling::kInverted, gen);
      h = nn::relu(skip + dropped.value);
      out.gates.push_back(dropped.gate);
    }
    out.stages.push_back(h);
  }
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}
} // namespace datamunge::nn::models
