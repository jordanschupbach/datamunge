#pragma once

/// @file audio_tagger.hpp
/// @brief Convolutional multilabel tagging on log-mel spectrograms.

#include <stdexcept>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Shared spectrogram CNN implementation details.
namespace audio_detail {

/// @brief Apply the shared two-block spectrogram CNN and dense head.
/// @param model Mutable trainable parameter store.
/// @param features Model-ready floating [batch, frames, bands] features.
/// @param classes Positive output width.
/// @param average_time Whether to average over pooled time frames.
/// @return Class logits [batch, classes].
/// @throws std::invalid_argument For incompatible input dimensions or parameters.
inline Tensor spectrogram_cnn_logits(ParameterStore& model, const Tensor& features,
                                     int classes, bool average_time) {
  if (!features.defined() || features.dim() != 3 || features.size(0) < 1 ||
      features.size(1) < 4 || features.size(2) < 4 || features.dtype() != model.dtype() ||
      classes < 1) {
    throw std::invalid_argument("Invalid spectrogram CNN features");
  }
  const auto image = expand_dims(features,3);
  const ConvLayerOptions options{.activation = Activation::Relu};
  const auto first = max_pool2d(conv2d(model, "conv1", image, 8, 3, options), 2, 2);
  const auto second = max_pool2d(conv2d(model, "conv2", first, 12, 3, options), 2, 2);
  const auto pooled = reshape(average_time ? mean(second,{1}) : second, {features.size(0),-1});
  return dense(model, "head", dense(model, "hidden", pooled, 24, Activation::Relu), classes);
}

}  // namespace audio_detail

/// @brief Score sound classes independently from batched log-mel features.
///
/// Applies two ReLU convolutions and two 2×2 max pools. A time average
/// preserves pooled frequency position for the dense classification head.
/// Use sigmoid scores for multilabel inference.
/// @param model Store whose parameters initialize on first use.
/// @param features Log-mel values [batch, frames, bands] matching the store dtype,
/// with a positive batch and at least four frames and bands.
/// @param classes Positive number of independent labels.
/// @return Unnormalized logits [batch, classes].
/// @throws std::invalid_argument If dimensions, dtype or stored parameters are incompatible.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// auto scores = dm::nn::audio_tag_logits(model, dm::nn::zeros({2,49,32}), 4);
/// @endcode
/// @par References
/// - K. J. Piczak (2015), Environmental Sound Classification with Convolutional Neural Networks.
///   doi:10.1109/MLSP.2015.7324337
/// @see log_mel_spectrogram(), sigmoid_cross_entropy()
inline Tensor audio_tag_logits(ParameterStore& model, const Tensor& features, int classes) {
  if (!features.defined() || features.dtype()!=model.dtype())
    throw std::invalid_argument("Invalid audio tagging dtype");
  return audio_detail::spectrogram_cnn_logits(model, (features + 6) * .25, classes, true);
}

/// @brief Classify a clip from standardized log-mel features with a keyword CNN.
/// Uses the full pooled time–frequency map to retain word timing information.
/// @param model Store whose parameters initialize on first use.
/// @param features Standardized [batch, frames, bands] features matching the store dtype.
/// Requires a positive batch and at least four frames and bands.
/// @param classes Number of keyword classes, default eight.
/// @return Logits [batch, classes] for softmax classification.
/// @throws std::invalid_argument For incompatible sizes, dtype or stored parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// auto words = dm::nn::keyword_cnn_logits(model, dm::nn::zeros({2,49,32}));
/// @endcode
/// @par References
/// - Warden (2018), Speech Commands. https://arxiv.org/abs/1804.03209
/// @see audio_tag_logits(), sparse_softmax_cross_entropy()
inline Tensor keyword_cnn_logits(ParameterStore& model, const Tensor& features, int classes = 8) {
  return audio_detail::spectrogram_cnn_logits(model, features, classes, false);
}

}  // namespace datamunge::nn
