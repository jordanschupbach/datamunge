#pragma once
#include "common/detection_scenes.hpp"
#include "datamunge/nn/detection_augmentation.hpp"
namespace detector {
struct DetectionTransform {
  double scale = 1, shift_y = 0, shift_x = 0, exposure = 1;
};
inline DetectionScenes
TransformDetectionScenes(const DetectionScenes &source, const std::vector<int> &rows,
                         const std::vector<DetectionTransform> &transforms) {
  if (source.height < 1 || source.width < 1 || source.labels.size() != source.boxes.size() ||
      source.pixels.size() != std::size_t(source.size()) * std::size_t(source.height) *
                                  std::size_t(source.width) ||
      rows.size() != transforms.size())
    throw std::invalid_argument("invalid affine scene storage");
  namespace nn = datamunge::nn;
  nn::models::TensorDetectionScenes data;
  std::vector<int> values(source.pixels.begin(), source.pixels.end());
  data.images = nn::from_vector(values, {source.size(), source.height, source.width, 1})
                    .to(nn::DType::Float32) /
                255.;
  for (int row = 0; row < source.size(); ++row) {
    data.boxes.push_back(nn::boxes_from_vector(source.boxes[row]));
    data.labels.push_back(nn::from_vector(
        std::vector<std::int64_t>(source.labels[row].begin(), source.labels[row].end())));
  }
  std::vector<double> settings;
  for (const auto &t : transforms)
    settings.insert(settings.end(), {t.scale, t.shift_y, t.shift_x, t.exposure});
  const auto out = nn::models::TransformTensorDetectionScenes(
      data, std::vector<std::int64_t>(rows.begin(), rows.end()),
      nn::from_vector(settings, {std::int64_t(rows.size()), 4}));
  DetectionScenes result;
  result.height = source.height;
  result.width = source.width;
  const auto bytes = nn::floor(out.images.to(nn::DType::Float64) * 255. + .5).to_vector<int>();
  for (int v : bytes)
    result.pixels.push_back(std::uint8_t(v));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    result.boxes.push_back(nn::boxes_to_vector(out.boxes[i]));
    result.labels.push_back(out.labels[i].to_vector<int>());
  }
  return result;
}
} // namespace detector
