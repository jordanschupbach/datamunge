/// @file
/// @brief Shared patch insertion and fitting for image robustness studies.
#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Construct a linear map from rectangular patch pixels to a flattened image.
/// @param image_height Image height in pixels.
/// @param image_width Image width in pixels.
/// @param patch_height Patch height in pixels.
/// @param patch_width Patch width in pixels.
/// @param top Zero-based insertion row.
/// @param left Zero-based insertion column.
/// @return Float32 CPU matrix [patch pixels,image pixels], with row-major indexing.
/// @throws std::invalid_argument Invalid dimensions or placement.
inline Tensor patch_insertion(std::int64_t image_height, std::int64_t image_width,
    std::int64_t patch_height, std::int64_t patch_width, std::int64_t top, std::int64_t left) {
  if (patch_height < 1 || patch_width < 1 || patch_height > image_height ||
      patch_width > image_width || top < 0 || left < 0 ||
      top > image_height - patch_height || left > image_width - patch_width) {
    throw std::invalid_argument("patch_insertion: invalid patch geometry");
  }
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (image_height > limit / image_width) {
    throw std::invalid_argument("patch_insertion: image size exceeds the tensor index range");
  }
  const auto pixels = image_height * image_width, count = patch_height * patch_width;
  if (count > limit / pixels) {
    throw std::invalid_argument("patch_insertion: map size exceeds the tensor index range");
  }
  std::vector<float> map(count * pixels);
  for (std::int64_t j = 0; j < count; ++j) {
    const auto i = (top + j / patch_width) * image_width + left + j % patch_width;
    map[j * pixels + i] = 1.f;
  }
  return from_vector(map, {count,pixels});
}

/// @brief Replace the selected pixels in each row with the same patch.
/// @param images Row-major flattened images [batch,image pixels].
/// @param patch Shared pixel values [1,patch pixels].
/// @param insertion Map returned by patch_insertion().
/// @return Patched images with gradients through both inputs.
inline Tensor insert_shared_patch(const Tensor& images, const Tensor& patch, const Tensor& insertion) {
  return images * (1.0 - sum(insertion, {0})) + matmul(patch, insertion);
}

/// @brief Minimize a differentiable image loss over a shared unit-range patch.
/// @param images Training images [batch,image pixels].
/// @param insertions Placement maps returned by patch_insertion(), all for the same geometry.
/// @param initial Initial patch [1,patch pixels] in [0,1].
/// @param objective Maps a batch of patched images to a scalar differentiable loss.
/// @param steps Number of projected gradient steps.
/// @param rate Positive learning rate.
/// @return Final patch and detached trajectory [steps+1,patch pixels].
/// @throws std::invalid_argument Invalid optimization settings.
inline std::pair<Tensor,Tensor> fit_shared_patch(const Tensor& images,
    const std::vector<Tensor>& insertions, const Tensor& initial,
    const std::function<Tensor(const Tensor&)>& objective, int steps, double rate) {
  if (insertions.empty() || !objective || steps < 0 || !std::isfinite(rate) || rate <= 0) {
    throw std::invalid_argument("fit_shared_patch: invalid optimization settings");
  }
  auto patch = initial.detach();
  std::vector<Tensor> trajectory{patch};
  for (int step = 0; step < steps; ++step) {
    auto leaf = patch.detach();
    leaf.requires_grad_(true);
    auto loss = objective(insert_shared_patch(images, leaf, insertions.front()));
    for (std::size_t i = 1; i < insertions.size(); ++i) {
      loss = loss + objective(insert_shared_patch(images, leaf, insertions[i]));
    }
    const auto gradient = grad(loss / double(insertions.size()), {leaf}).at(0);
    patch = clip(patch - rate * gradient, 0.0, 1.0).detach();
    trajectory.push_back(patch);
  }
  return {patch, concat(trajectory, 0)};
}

} // namespace datamunge::nn
