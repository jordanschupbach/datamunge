#pragma once
/// @file medical_volume_models.hpp
/// @brief Custom residual3Dmedical-volume classifier with explicit fitting-only normalization state.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Own custom residual CT activations and categorical head scores.
///
/// Tensor handles retain graph/storage. Fitting-only scalar moments live in
/// the separate state store;no hidden patient metadata or optimizer.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(1049), s(1051);
/// const auto out = m::MedicalVolume(p, s, n::zeros({1, 28, 28, 28, 1}));
/// @endcode
/// @par References
/// - Yang et al.(2023),MedMNIST v2:A large-scale lightweight benchmark for2Dand3Dbiomedical image classification, https://doi.org/10.1038/s41597-022-01721-8;corrected NoduleMNIST3D v2.1 archive https://zenodo.org/records/6496656,official original label/split description https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py. The three-stage residual model is a custom source teaching CNN,not a named published medical architecture or clinical predictor.
/// @see MedicalVolume()
struct MedicalVolumeOutput {
  Tensor normalized;          ///< Input standardized with caller fitting-only scalar moments.
  Tensor global;              ///< GlobalD/H/Wmean after three residual stages.
  Tensor logits;              ///< Raw binary[N,2] scores,benign0/malignant1.
  std::vector<Tensor> stages; ///< ThreeReLU stage endpoints,28->14->7->4.
};

/// @brief Apply the custom stride2three-dimensional residual block.
///
/// Both3x3x3convolutions and1x1x1stride2skip have He-uniform kernels and
/// zero biases. SAME padding preserves odd/even source alignment. No BN/dropout
/// in a block;the head alone has dropout. Output spatial axes are ceil(input/2).
///
/// @param p Mutable learnedW/bstore,no fitting here.
/// @param name Caller prefix with/first,/second,/projection suffixes.
/// @param input FloatingNDHWCvolume,positive channel count.
/// @param out Positive output channels.
/// @return ReLU(conv3(ReLU(conv3stride2(input)))+conv1stride2(input)).
/// @throws std::invalid_argument For invalid input/channel geometry or stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(1049);
/// const auto y = m::MedicalResidual(p, "r", n::ones({1, 5, 5, 5, 2}), 4);
/// @endcode
/// @par References
/// - Yang et al.(2023),MedMNIST v2:A large-scale lightweight benchmark for2Dand3Dbiomedical image classification, https://doi.org/10.1038/s41597-022-01721-8;corrected NoduleMNIST3D v2.1 archive https://zenodo.org/records/6496656,official original label/split description https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py. The three-stage residual model is a custom source teaching CNN,not a named published medical architecture or clinical predictor.
/// @see MedicalVolume()
inline Tensor MedicalResidual(nn::ParameterStore &p, const std::string &name,
                              const Tensor &input, std::int64_t out) {
  if (input.dim() != 5 || input.size(4) < 1 || out < 1)
    throw std::invalid_argument("positive NDHWC residual channels required");
  nn::ConvLayerOptions opt;
  opt.init = nn::init::he_uniform();
  opt.stride = 2;
  const Tensor first = nn::relu(nn::conv3d(p, name + "/first", input, out, 3, opt));
  opt.stride = 1;
  const Tensor second = nn::conv3d(p, name + "/second", first, out, 3, opt);
  opt.stride = 2;
  const Tensor shortcut = nn::conv3d(p, name + "/projection", input, out, 1, opt);
  return nn::relu(second + shortcut);
}

/// @brief Classify28cube intensities with the complete custom three-stage residual model.
///
/// If moments are absent,lazily initialize mean0/stddev1;no population estimation
/// is hidden. All20learned leaves are six3x3x3convs,three1x1x1projections and
/// oneGlorotdensehead. Head dropout follows global average pool. No BN.
/// Restore learned arrays AND the unchanged fitting-only scalar moments.
///
/// @param p Mutable learned store;warm all20leaves before optimizer construction.
/// @param state Matching floating store with scalar normalization/mean and/stddev;fit these only on fitting voxels.
/// @param volume Floating[N,28,28,28,1] intensity,source[0,1],Npositive.
/// @param base Positive stage width,default4;stagesbase/2base/4base.
/// @param training False by default;true enables head keep.8dropout.
/// @param dropout_gen Required mutable generator for training,unused at inference.
/// @return Normalized input,three residual stages,global means and binary logits.
/// @throws std::invalid_argument For invalid shape,dtype,base,dropout,normalization or stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(1049), s(1051);
/// const auto out = m::MedicalVolume(p, s, n::zeros({1, 28, 28, 28, 1}));
/// @endcode
/// @par References
/// - Yang et al.(2023),MedMNIST v2:A large-scale lightweight benchmark for2Dand3Dbiomedical image classification, https://doi.org/10.1038/s41597-022-01721-8;corrected NoduleMNIST3D v2.1 archive https://zenodo.org/records/6496656,official original label/split description https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py. The three-stage residual model is a custom source teaching CNN,not a named published medical architecture or clinical predictor.
/// @see MedicalVolume()
inline MedicalVolumeOutput MedicalVolume(nn::ParameterStore &p, nn::ParameterStore &state,
                                         const Tensor &volume, int base = 4,
                                         bool training = false,
                                         nn::Generator *dropout_gen = nullptr) {
  if (volume.dim() != 5 || volume.size(0) < 1 ||
      volume.shape() != nn::Shape{volume.size(0), 28, 28, 28, 1} ||
      volume.dtype() != p.dtype() || state.dtype() != p.dtype() || (training && !dropout_gen))
    throw std::invalid_argument(
        "matching floating28cube intensity and explicit head dropout required");
  if (base < 1)
    throw std::invalid_argument("positive medical volume width");
  MedicalVolumeOutput out;
  const Tensor mean = state.get("normalization/mean", {}, nn::init::zeros()).detach();
  const Tensor stddev = state.get("normalization/stddev", {}, nn::init::ones()).detach();
  if (!nn::all(nn::is_finite(mean)).item<bool>() ||
      !nn::all(nn::is_finite(stddev)).item<bool>() || stddev.item() <= 0)
    throw std::invalid_argument(
        "finite fitting-only scalar mean and positive stddev required");
  out.normalized = (volume - mean) / stddev;
  Tensor x = out.normalized;
  for (int i = 0; i < 3; ++i) {
    x = MedicalResidual(p, "stage" + std::to_string(i), x, std::int64_t{base} << i);
    out.stages.push_back(x);
  }
  out.global = nn::mean(x, {1, 2, 3});
  x = out.global;
  if (training)
    x = nn::dropout(x, .2, true, *dropout_gen);
  out.logits = nn::dense(p, "head", x, 2);
  return out;
}

} // namespace datamunge::nn::models
