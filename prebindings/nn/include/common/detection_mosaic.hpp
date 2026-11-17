#pragma once
#include "common/detection_scenes.hpp"
#include "datamunge/nn/detection_mosaic.hpp"
namespace detector {
inline nn::models::TensorDetectionScenes TensorMosaicRows(const DetectionScenes &source,
                                                          const std::vector<int> &rows) {
  if (source.height < 1 || source.width < 1 || source.labels.size() != source.boxes.size() ||
      source.pixels.size() !=
          std::size_t(source.size()) * std::size_t(source.height) * std::size_t(source.width))
    throw std::invalid_argument("invalid mosaic source storage");
  nn::models::TensorDetectionScenes out;
  std::vector<nn::Tensor> images;
  for (auto row : rows) {
    if (row < 0 || row >= source.size())
      throw std::invalid_argument("invalid mosaic source row");
    const auto *p = source.image(row);
    images.push_back(nn::from_vector(std::vector<int>(p, p + source.height * source.width),
                                     {source.height, source.width, 1})
                         .to(nn::DType::Float32) /
                     255.);
    out.boxes.push_back(nn::boxes_from_vector(source.boxes[row]));
    out.labels.push_back(nn::from_vector(
        std::vector<std::int64_t>(source.labels[row].begin(), source.labels[row].end())));
  }
  out.images =
      images.empty() ? nn::zeros({0, source.height, source.width, 1}) : nn::stack(images, 0);
  return out;
}
inline DetectionScenes LegacyMosaicRows(const nn::models::TensorDetectionScenes &data) {
  DetectionScenes out;
  out.height = int(data.images.shape()[1]);
  out.width = int(data.images.shape()[2]);
  const auto bytes =
      nn::floor(data.images.to(nn::DType::Float64) * 255. + .5).to_vector<int>();
  for (int value : bytes)
    out.pixels.push_back(std::uint8_t(value));
  for (std::size_t row = 0; row < data.boxes.size(); ++row) {
    out.boxes.push_back(nn::boxes_to_vector(data.boxes[row]));
    out.labels.push_back(data.labels[row].to_vector<int>());
  }
  return out;
}
inline DetectionScenes MosaicScenes(const DetectionScenes &source,
                                    const std::vector<std::array<int, 4>> &selections) {
  if (source.height < 2 || source.width < 2 || source.height % 2 || source.width % 2)
    throw std::invalid_argument("mosaic requires positive even image dimensions");
  std::vector<int> rows;
  for (const auto &s : selections)
    rows.insert(rows.end(), s.begin(), s.end());
  if (rows.empty()) {
    DetectionScenes out;
    out.height = source.height;
    out.width = source.width;
    return out;
  }
  const auto data = TensorMosaicRows(source, rows);
  const auto chosen = nn::arange(0, double(rows.size()), 1, nn::DType::Int64)
                          .reshape({std::int64_t(selections.size()), 4});
  return LegacyMosaicRows(nn::models::MosaicTensorDetectionScenes(data, chosen));
}
inline DetectionScenes CopyDetectionRows(const DetectionScenes &source,
                                         const std::vector<int> &rows) {
  for (auto row : rows)
    if (row < 0 || row >= source.size())
      throw std::out_of_range("detection row");
  if (rows.empty()) {
    DetectionScenes out;
    out.height = source.height;
    out.width = source.width;
    return out;
  }
  const auto data = TensorMosaicRows(source, rows);
  std::vector<std::int64_t> local(rows.size());
  std::iota(local.begin(), local.end(), 0);
  return LegacyMosaicRows(nn::models::SelectTensorDetectionScenes(data, local));
}
inline void FlipDetectionScene(DetectionScenes &source, int row) {
  if (row < 0 || row >= source.size())
    throw std::out_of_range("detection flip row");
  const auto data = TensorMosaicRows(source, {row});
  const auto flipped = LegacyMosaicRows(nn::models::FlipTensorDetectionScenes(data, {0}));
  const auto start =
      source.pixels.begin() + std::ptrdiff_t(row * source.height * source.width);
  std::copy(flipped.pixels.begin(), flipped.pixels.end(), start);
  source.boxes[row] = flipped.boxes[0];
  source.labels[row] = flipped.labels[0];
}
} // namespace detector
