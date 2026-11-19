/// @file
/// @brief Compact image pretext transformations and classification.
#pragma once

#include <datamunge/nn/layers.hpp>

#include <array>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Select a four-class image pretext transformation.
/// @see image_pretext_features()
enum class ImagePretextTask {
  Rotation,         ///< Counterclockwise quarter turns.
  Jigsaw,           ///< Four predefined orders of corner tiles.
  RelativePosition  ///< Center patch followed by one of four neighboring patches.
};

/// @brief Prepare flattened inputs for the compact image pretext tasks.
///
/// Rotation uses all pixels. Jigsaw concatenates four 3-by-3 corner tiles,
/// and relative position concatenates two 2-by-2 patches starting with the
/// central patch. The latter two outputs are zero-padded to 64 features.
/// The operation materializes features on the host and detaches the result.
///
/// @param images Floating grayscale images [N,8,8] with N positive.
/// @param labels Integral transformation labels [N] in [0,4).
/// @param task Transformation to apply.
/// @return Floating [N,64] features in the image dtype.
/// @throws std::invalid_argument For incompatible input shapes, dtypes or labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::image_pretext_features(n::ones({4,8,8}),
///     n::arange(0,4,1,n::DType::Int64), n::ImagePretextTask::Rotation);
/// @endcode
/// @par References
/// - Gidaris et al. (2018), https://arxiv.org/abs/1803.07728
/// - Noroozi and Favaro (2016), https://arxiv.org/abs/1603.09246
/// - Doersch et al. (2015), https://arxiv.org/abs/1505.05192
/// @see image_pretext_logits()
inline Tensor image_pretext_features(const Tensor &images, const Tensor &labels,
                                    ImagePretextTask task) {
  if (images.dim() != 3 || images.shape()[0] < 1 || images.shape()[1] != 8 ||
      images.shape()[2] != 8 || !is_floating(images.dtype()) ||
      labels.shape() != Shape{images.shape()[0]} || !is_integral(labels.dtype()) ||
      (task != ImagePretextTask::Rotation && task != ImagePretextTask::Jigsaw &&
       task != ImagePretextTask::RelativePosition))
    throw std::invalid_argument("image_pretext_features: invalid image, label or task geometry");

  constexpr int orders[4][4]{{0,1,2,3}, {2,0,3,1}, {3,2,1,0}, {1,3,0,2}};
  constexpr std::array<std::array<int, 2>, 4> offsets{{{0,3}, {3,0}, {3,6}, {6,3}}};
  const auto pixels = images.to_vector<double>();
  const auto classes = labels.to_vector<std::int64_t>();
  std::vector<double> result(pixels.size(), 0.);
  for (std::size_t row = 0; row < classes.size(); ++row) {
    const auto label = classes[row];
    if (label < 0 || label >= 4)
      throw std::invalid_argument("image_pretext_features: label outside [0,4)");
    auto dst = row * 64;
    const auto at = [&](int i, int j) { return pixels[row * 64 + i * 8 + j]; };
    if (task == ImagePretextTask::Rotation) {
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
          int y = i, x = j;
          for (int k = 0; k < label; ++k) {
            const int next_x = 7 - y;
            y = x;
            x = next_x;
          }
          result[dst++] = at(y, x);
        }
    } else if (task == ImagePretextTask::Jigsaw) {
      for (int tile : orders[label])
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            result[dst++] = at(5 * (tile / 2) + i, 5 * (tile % 2) + j);
    } else {
      for (const auto &patch : {std::array{3, 3}, offsets[label]})
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 2; ++j)
            result[dst++] = at(patch[0] + i, patch[1] + j);
    }
  }
  return from_vector(result, {images.shape()[0], 64}).to(images.dtype());
}

/// @brief Predict transformation labels with the compact four-class pretext MLP.
///
/// @param store Owns the trainable parameters.
/// @param name Layer prefix, reused on subsequent calls.
/// @param features Floating [N,64] output of image_pretext_features().
/// @return Logits [N,4] from a 32-unit ReLU hidden layer and linear head.
/// @throws std::invalid_argument For incompatible features or existing weights.
/// @par Example
/// @code
/// dm::nn::ParameterStore store(7);
/// const auto logits = dm::nn::image_pretext_logits(store, "rotation", dm::nn::ones({4,64}));
/// @endcode
/// @see image_pretext_features()
inline Tensor image_pretext_logits(ParameterStore &store, const std::string &name,
                                   const Tensor &features) {
  if (features.dim() != 2 || features.shape()[1] != 64 || !is_floating(features.dtype()))
    throw std::invalid_argument("image_pretext_logits: expected floating [N,64] features");
  const auto hidden = dense(store, name + "/hidden", features, 32, Activation::Relu);
  return dense(store, name + "/head", hidden, 4);
}

}  // namespace datamunge::nn
