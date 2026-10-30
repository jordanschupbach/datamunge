#pragma once
/// @file grad_cam.hpp
/// @brief Coarse Grad-CAM and explicit exp-score Grad-CAM++formulas.
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/saliency_models.hpp"
#include <stdexcept>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Own Grad-CAM and explicit locally-affine exp-score Grad-CAM++intermediates.
///
/// Tensor handles retain their graphs. Maps are coarse feature-space quantities,
/// not pixel-level object masks. No resizing or normalization hidden in this value.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto out = m::ClassActivationMaps(n::ones({1, 2, 2, 3}), n::ones({1, 2, 2, 3}));
/// @endcode
/// @par References
/// - Selvaraju etal(2017),Grad-CAM:Visual Explanations from Deep Networks via Gradient-based Localization, https://arxiv.org/abs/1610.02391;Chattopadhyay etal(2018),Grad-CAM++:Improved Visual Explanations for Deep Convolutional Networks, https://arxiv.org/abs/1710.11063. Grad-CAM++Eq19here follows exp(S-c)with cfixed at current rawS and locally affine downstream rawscore,nonnegative post-ReLUfeatures;g squared/cubed are exp-score higher-derivative identities,NOT rawlogit second/third derivatives.
/// @see ClassActivationMaps(),NormalizeCam()
struct CamOutput {
  Tensor grad_weights;   ///< Spatial-mean raw-score derivatives[N,C].
  Tensor grad_signed;    ///< Unrectified weighted activation sum[N,H,W].
  Tensor grad_map;       ///< ReLU of the signed Grad-CAM sum[N,H,W].
  Tensor positive_alpha; ///< Nonnegative Eq19positive-gradient alpha[N,H,W,C],zero for g<=0.
  Tensor plus_weights;   ///< Spatial sum(alpha*ReLU(raw_score_gradient))[N,C].
  Tensor plus_map;       ///< ReLU of the Grad-CAM++weighted activation sum[N,H,W].
};

/// @brief Compute Grad-CAM and paper-equation Grad-CAM++from post-ReLU features and raw-score derivatives.
///
/// Classicweights=mean_spatial(g);classicmap=ReLU(sum_channels(A*weights)).
/// For exp(S-c),c fixed at the current rawscore and locally affine S,at thispoint
/// first/second/thirdexp-score derivatives are g,g squared,g cubed. For g>0,
/// Eq19alpha simplifies1/(2+sum_spatial(A)*g);g<=0contributes0. This does NOT
/// claim rawlogit higher derivatives equal its first derivative powers.
/// Nonnegative A ensures positive relevant denominators. No fitting,softmax,
/// resizing or normalization;caller freezes its fixed targets and learned model.
///
/// @param activation Finite floating[N,H,W,C]nonnegative feature activation.
/// @param raw_score_gradient Matching finite derivative of SUMfixed selected RAWscores wrt activation.
/// @return Channel weights,signed/rectified Grad-CAM,positive-gradient alpha,Grad-CAM++weights/map.
/// @throws std::invalid_argument For empty/nonmatching rank4,dtype,nonfinite values or negative activation.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto out = m::ClassActivationMaps(n::ones({1, 2, 2, 3}), n::ones({1, 2, 2, 3}));
/// @endcode
/// @par References
/// - Selvaraju etal(2017),Grad-CAM:Visual Explanations from Deep Networks via Gradient-based Localization, https://arxiv.org/abs/1610.02391;Chattopadhyay etal(2018),Grad-CAM++:Improved Visual Explanations for Deep Convolutional Networks, https://arxiv.org/abs/1710.11063. Grad-CAM++Eq19here follows exp(S-c)with cfixed at current rawS and locally affine downstream rawscore,nonnegative post-ReLUfeatures;g squared/cubed are exp-score higher-derivative identities,NOT rawlogit second/third derivatives.
/// @see ClassActivationMaps(),NormalizeCam()
inline CamOutput ClassActivationMaps(const Tensor &activation,
                                     const Tensor &raw_score_gradient) {
  if (activation.dim() != 4 || activation.numel() < 1 ||
      activation.shape() != raw_score_gradient.shape() ||
      activation.dtype() != raw_score_gradient.dtype() ||
      !nn::all(nn::is_finite(activation)).item<bool>() ||
      !nn::all(nn::is_finite(raw_score_gradient)).item<bool>() ||
      nn::min(activation).item() < 0)
    throw std::invalid_argument(
        "finite matching nonnegative post-ReLU feature/gradient geometry required");
  CamOutput out;
  auto expand = [](const Tensor &w) {
    return nn::expand_dims(nn::expand_dims(w, 1), 1);
  };
  out.grad_weights = nn::mean(raw_score_gradient, {1, 2});
  out.grad_signed = nn::sum(nn::mul(activation, expand(out.grad_weights)), {3});
  out.grad_map = nn::relu(out.grad_signed);
  // For Y=exp(S-c),c fixed at current S,and locally affine raw S:
  // dY/dA=g,d2Y/dA2=g^2,d3Y/dA3=g^3 at this point. For g>0,Eq19
  // cancels g^2 exactly:alpha=1/(2+sum_spatial(A)*g). Other contributions
  // have ReLU(g)=0 and alpha is set0,avoiding irrelevant singular denominators.
  const Tensor positive = nn::greater(raw_score_gradient, 0.0);
  const Tensor sums = nn::sum(activation, {1, 2}, true);
  const Tensor denominator =
      nn::add(nn::scalar_like(2.0, sums), nn::mul(sums, raw_score_gradient));
  const Tensor safe = nn::where(positive, denominator, 1.0);
  out.positive_alpha = nn::div(nn::cast(positive, activation.dtype()), safe);
  out.plus_weights =
      nn::sum(nn::mul(out.positive_alpha, nn::relu(raw_score_gradient)), {1, 2});
  out.plus_map = nn::relu(nn::sum(nn::mul(activation, expand(out.plus_weights)), {3}));
  return out;
}

/// @brief Normalize each nonnegative coarse map by its spatial maximum with a zero-safe floor.
///
/// Per-map normalization is display scaling and discards relative amplitude
/// between observations/controls. Preserve rawmaps when comparing magnitude.
///
/// @param map Finite nonnegative floating[N,H,W]coarse CAM.
/// @return map/max(spatial_max,1e-20),allzero maps remain exactlyzero.
/// @throws std::invalid_argument For empty/non-rank3,nonfinite or negative map values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto zero = m::NormalizeCam(n::zeros({2, 3, 3}));
/// @endcode
/// @par References
/// - Selvaraju etal(2017),Grad-CAM:Visual Explanations from Deep Networks via Gradient-based Localization, https://arxiv.org/abs/1610.02391;Chattopadhyay etal(2018),Grad-CAM++:Improved Visual Explanations for Deep Convolutional Networks, https://arxiv.org/abs/1710.11063. Grad-CAM++Eq19here follows exp(S-c)with cfixed at current rawS and locally affine downstream rawscore,nonnegative post-ReLUfeatures;g squared/cubed are exp-score higher-derivative identities,NOT rawlogit second/third derivatives.
/// @see ClassActivationMaps(),NormalizeCam()
inline Tensor NormalizeCam(const Tensor &map) {
  if (map.dim() != 3 || map.numel() < 1 || !nn::all(nn::is_finite(map)).item<bool>() ||
      nn::min(map).item() < 0)
    throw std::invalid_argument("finite nonnegative NHW CAM required");
  return nn::div(map, nn::maximum(nn::max(map, {1, 2}, true), nn::scalar_like(1e-20, map)));
}

/// @brief Own last-convolution derivatives and both frozen-digit class activation maps.
///
/// Owning handles retain actual raw14x14features/gradients and both map types.
/// Per-map normalization discards relative amplitudes;rawmaps remain available.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(1103);
/// const auto out = m::SaliencyCam(p, n::zeros({1, 28, 28, 1}), n::zeros({1}, n::DType::Int32));
/// @endcode
/// @par References
/// - Selvaraju etal(2017),Grad-CAM, https://arxiv.org/abs/1610.02391;Chattopadhyay etal(2018),Grad-CAM++:Improved Visual Explanations for Deep Convolutional Networks, https://arxiv.org/abs/1710.11063. Frozen26826parameter MNIST teaching classifier,not pretrained paper benchmark.
/// @see SaliencyCam(),ClassActivationMaps()
struct SaliencyCamOutput {
  Tensor activation; ///< Detached nonnegative last-conv[N,14,14,16]features.
  Tensor gradient;   ///< Detached fixed raw-logit derivative wrt every last-conv coordinate.
  CamOutput maps;    ///< All raw Grad-CAM/Grad-CAM++formulas and channel weights.
  Tensor normalized; ///< Per-case maximum-normalized classic[N,14,14]map.
  Tensor plus_normalized; ///< Per-case maximum-normalized plus[N,14,14]map.
  Tensor resized;         ///< Classic map half-pixel bilinear display[N,28,28].
  Tensor plus_resized;    ///< Plus map half-pixel bilinear display[N,28,28].
  Tensor logits;          ///< Detached unchanged classifier raw[N,10]scores.
};
/// @brief Explain fixed raw digit scores at the actual last post-ReLU convolution.
///
/// Full shared classifier forward;ordinary SUMraw-score reverse wrt non-leaf
/// conv2,then detach. All14x14x16coordinates retained. Compute classic and
/// locally-affine exp-score CAM++formulas,zero-safe per-case max normalization,
/// half-pixel bilinear28pxdisplay. No classifier training or weight/RNG mutation
/// after warming;maps are coarse sensitivity summaries,not segmentation.
///
/// @param p Mutable shared classifier store;restore selected weights before use,no optimizer applied.
/// @param image Float32[N,28,28,1]actual normalized original images.
/// @param targets Int32[N]fixed selected raw categories0..9.
/// @return Actual features/derivatives,all raw formulas,normalized/resized maps and unchanged logits.
/// @throws std::invalid_argument For invalid image/target/feature or stored-parameter contracts.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(1103);
/// const auto out = m::SaliencyCam(p, n::zeros({1, 28, 28, 1}), n::zeros({1}, n::DType::Int32));
/// @endcode
/// @par References
/// - Selvaraju etal(2017),Grad-CAM, https://arxiv.org/abs/1610.02391;Chattopadhyay etal(2018),Grad-CAM++:Improved Visual Explanations for Deep Convolutional Networks, https://arxiv.org/abs/1710.11063. Frozen26826parameter MNIST teaching classifier,not pretrained paper benchmark.
/// @see SaliencyCam(),ClassActivationMaps()
inline SaliencyCamOutput SaliencyCam(nn::ParameterStore &p, const Tensor &image,
                                     const Tensor &targets) {
  const auto out = SaliencyCnn(p, image);
  SaliencyCamOutput result;
  result.gradient =
      nn::grad(nn::sum(SaliencyScore(out.logits, targets)), {out.conv2})[0].detach();
  nn::NoGradGuard guard;
  result.activation = out.conv2.detach();
  result.maps = ClassActivationMaps(result.activation, result.gradient);
  result.normalized = NormalizeCam(result.maps.grad_map);
  result.plus_normalized = NormalizeCam(result.maps.plus_map);
  result.resized =
      nn::resize_bilinear(nn::expand_dims(result.normalized, 3), {28, 28}, false, true)
          .reshape({image.size(0), 28, 28});
  result.plus_resized =
      nn::resize_bilinear(nn::expand_dims(result.plus_normalized, 3), {28, 28}, false, true)
          .reshape({image.size(0), 28, 28});
  result.logits = out.logits.detach();
  return result;
}
/// @brief Evaluate the frozen classifier tail from explicit last-conv feature coordinates.
///
/// VALIDavg2,flatten784,existingGlorotFC32ReLU and existinglinear10head.
/// No fitting,BN/dropout or new leaves after warming the full classifier.
///
/// @param p Shared classifier store containing hidden and head leaves.
/// @param activation Float32[N,14,14,16]last post-ReLU features,possibly perturbed for derivative checks.
/// @return Raw[N,10]tail logits retaining the activation graph.
/// @throws std::invalid_argument For invalid activation or incompatible stored leaf geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(1103);
/// m::SaliencyCnn(p, n::zeros({1, 28, 28, 1}));
/// const auto z = m::SaliencyCamTail(p, n::zeros({1, 14, 14, 16}));
/// @endcode
/// @par References
/// - Selvaraju etal(2017),Grad-CAM, https://arxiv.org/abs/1610.02391;Chattopadhyay etal(2018),Grad-CAM++:Improved Visual Explanations for Deep Convolutional Networks, https://arxiv.org/abs/1710.11063. Frozen26826parameter MNIST teaching classifier,not pretrained paper benchmark.
/// @see SaliencyCam(),ClassActivationMaps()
inline Tensor SaliencyCamTail(nn::ParameterStore &p, const Tensor &activation) {
  if (activation.dim() != 4 || activation.shape() != nn::Shape{activation.size(0), 14, 14, 16})
    throw std::invalid_argument("lastconv14x14x16feature geometry required");
  return nn::dense(p, "head",
                   nn::relu(nn::dense(p, "hidden",
                                      nn::avg_pool2d(activation, 2, 2, nn::Padding::Valid)
                                          .reshape({activation.size(0), 784}),
                                      32)),
                   10);
}
} // namespace datamunge::nn::models
