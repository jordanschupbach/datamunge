#pragma once
/// @file dncnn_models.hpp
/// @brief Orthogonal17-layer tensor noise prediction with moving BN and signed residual reconstruction.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own predicted signed noise,restored image and every DnCNN stage.
///
/// Tensor handles preserve the graph. Clean output retains the direct input shortcut.
///
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// const auto r = n::models::DnCNN(p, state, n::zeros({1, 7, 9, 1}), 16);
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
struct DnCNNResult {
  nn::Tensor noise; ///< Signed predicted noise[N,H,W,1],unclipped.
  nn::Tensor clean; ///< Noisy input minus predicted noise,unclipped.
  std::vector<nn::Tensor>
      stages; ///< All17 convolutional stage activations,ending with predicted noise.
};

/// @brief Subtract signed predicted noise without an activation or clamp.
///
/// Negative and greater-than-one values remain available for raw denoising scores.
///
/// @param noisy Floating noisy observations.
/// @param noise Same-shaped signed predicted noise.
/// @return noisy-noise,with input derivative+1 and noise derivative-1.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::DnSubtract(n::tensor({-.1, 1.2}), n::tensor({.1, -.2}));
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline nn::Tensor DnSubtract(const nn::Tensor &noisy, const nn::Tensor &noise) {
  return noisy - noise;
}

/// @brief Normalize DnCNN activations with separately owned moving BN.
///
/// Old-state retention.05(new-batch weight.95),epsilon1e-4. Training uses
/// population variance;moving state uses unbiased sample variance. Save learned
/// and moving stores together. No optimizer or hidden fitting.
///
/// @param p Mutable learned gamma/beta store.
/// @param state Mutable nontrainable moving mean/variance store.
/// @param name BN parameter/state prefix.
/// @param x Floating NHWC activations matching p dtype.
/// @param training Use batch statistics when true,stored moments otherwise.
/// @param update Refresh statistics only in training when true,default true.
/// @return Normalized activation retaining input/affine gradients.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// const auto y = n::models::DnNorm(p, state, "bn", n::ones({2, 3, 3, 4}), true, false);
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline nn::Tensor DnNorm(nn::ParameterStore &p, nn::ParameterStore &state,
                         const std::string &name, const nn::Tensor &x, bool training,
                         bool update = true) {
  nn::BatchNormOptions o;
  o.momentum = .05;
  o.eps = 1e-4;
  o.update_stats = update;
  return nn::batch_norm(p, state, name, x, training, o);
}

/// @brief Predict additive Gaussian noise with a complete17-layer residual denoiser.
///
/// First3x3 convolution/bias/ReLU,15 bias-free Conv/BN/ReLU,final linear
/// bias-free noise convolution. Orthogonal flattened kernels. All17 layers remain
/// at either width;there is no implicit optimizer or target input. Inference
/// uses fixed BN and preserves single/batch independence.
///
/// @param p Mutable learned store;warm every leaf before constructing optimizer.
/// @param state Mutable separate running BN store.
/// @param noisy Floating[N,H,W,1],positive geometry,matching p dtype;no input clipping.
/// @param width Positive hidden channel width,default64;16 is compact teaching width.
/// @param training Batch BN moments when true,default false.
/// @param update Refresh BN only in training and true,default true.
/// @return Predicted noise,unclipped noisy-minus-noise and all17 stages.
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// const auto y = n::models::DnCNN(p, state, n::zeros({2, 19, 21, 1}), 16, true, false);
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline DnCNNResult DnCNN(nn::ParameterStore &p, nn::ParameterStore &state,
                         const nn::Tensor &noisy, int width = 64, bool training = false,
                         bool update = true) {
  if (width < 1 || noisy.dim() != 4 || noisy.size(0) < 1 || noisy.size(1) < 1 ||
      noisy.size(2) < 1 || noisy.size(3) != 1 || noisy.dtype() != p.dtype())
    throw std::invalid_argument("positive DnCNN width");
  DnCNNResult result;
  /// @brief Create or reuse one orthogonally initialized SAME convolution.
  /// @param name Parameter prefix.
  /// @param x Floating NHWC activation.
  /// @param out Output-channel count.
  /// @param bias Whether to include a learned additive bias.
  /// @return Convolved activation before normalization or nonlinearity.
  auto convolution = [&](const std::string &name, const nn::Tensor &x, int out, bool bias) {
    nn::ConvLayerOptions o;
    o.bias = bias;
    o.init = nn::init::orthogonal();
    return nn::conv2d(p, name, x, out, 3, o);
  };
  nn::Tensor x = nn::relu(convolution("conv1", noisy, width, true));
  result.stages.push_back(x);
  for (int layer = 2; layer <= 16; ++layer) {
    const auto name = "conv" + std::to_string(layer);
    x = convolution(name, x, width, false);
    x = nn::relu(DnNorm(p, state, name + "/bn", x, training, update));
    result.stages.push_back(x);
  }
  result.noise = convolution("conv17", x, 1, false);
  result.stages.push_back(result.noise);
  result.clean = DnSubtract(noisy, result.noise);
  return result;
}

/// @brief Compute pixel-averaged residual half squared error.
///
/// Gradient(predicted-true)/numel. Noise targets reflect final Float32 noisy
/// rounding,not an independently rounded Gaussian draw. No optimizer mutation.
///
/// @param predicted_noise Nonempty floating predicted signed noise.
/// @param true_noise Matching actual noisy-clean residual tensor.
/// @return Scalar.5*mean((predicted_noise-true_noise)^2).
/// @throws std::invalid_argument For invalid geometry,dtype,range or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::DnResidualLoss(n::zeros({2, 3, 4, 1}), n::ones({2, 3, 4, 1}));
/// @endcode
/// @par References
/// - Zhang et al.(2017),Beyond a Gaussian Denoiser:Residual Learning of Deep CNN for Image Denoising, https://arxiv.org/abs/1608.03981;author training https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py. Natural CIFAR32/sigma25/255 compact teaching fit is not Train400/BSD68 evaluation.
/// @see DnCNN(), DnCorrupt(), ScoreTensorDn()
inline nn::Tensor DnResidualLoss(const nn::Tensor &predicted_noise,
                                 const nn::Tensor &true_noise) {
  if (predicted_noise.shape() != true_noise.shape() || predicted_noise.numel() < 1)
    throw std::invalid_argument("aligned nonempty residual targets required");
  return .5 * nn::mean(nn::squared_difference(predicted_noise, true_noise));
}

} // namespace datamunge::nn::models
