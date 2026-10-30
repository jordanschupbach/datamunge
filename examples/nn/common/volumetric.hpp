#pragma once
// Compatibility names and containers;all model,data,loss,metric and preview numerics use canonical tensors.
#include "common/checks.hpp"
#include "common/paths.hpp"
#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/volumetric_data.hpp"
#include "datamunge/nn/volumetric_models.hpp"
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
namespace volumetric {
namespace nn = datamunge::nn;
using nn::Padding;
using nn::Tensor;
using nn::models::SparseVolumeCE;
using nn::models::SquaredVolumeDice;
using nn::models::Unet3D;
using nn::models::Unet3DConfig;
using nn::models::Unet3DOutputSide;
using nn::models::VNet;
using nn::models::VNetConfig;
using nn::models::VolumeCrop;
using nn::models::VolumeOutputs;
using nn::models::VolumePRelu;
using nn::models::VolumeResidual;
using nn::models::VolumeTranspose;
struct VolumeData {
  int count = 0, side = 16;
  std::vector<float> pixels;
  std::vector<int> labels;
};
inline nn::models::TensorVolumeData TensorData(const VolumeData &data) {
  return {nn::from_vector(data.pixels, {data.count, data.side, data.side, data.side, 1}),
          nn::from_vector(std::vector<std::int64_t>(data.labels.begin(), data.labels.end()),
                          {data.count, data.side, data.side, data.side})};
}
inline VolumeData EllipsoidVolumes(int count, int side, int seed) {
  const auto data = nn::models::MakeTensorEllipsoidVolumes(count, side, std::uint64_t(seed));
  return {count, side, data.images.to_vector<float>(), data.labels.to_vector<int>()};
}
inline std::pair<Tensor, Tensor>
VolumeBatch(const VolumeData &data, const std::vector<int> &rows, bool sparse = false) {
  for (const int row : rows)
    if (row < 0 || row >= data.count)
      throw std::out_of_range("volume batch row");
  const auto batch = nn::models::TensorVolumeBatch(
      TensorData(data), std::vector<std::int64_t>(rows.begin(), rows.end()), sparse);
  return {batch.first, batch.second.to(nn::DType::Int32)};
}
struct VolumeMetrics {
  double dice = 0, iou = 0, accuracy = 0;
  std::int64_t tp = 0, fp = 0, fn = 0, tn = 0;
};
using LogitsFn = std::function<Tensor(const Tensor &images)>;
inline nn::Tensor DensePredictions(const LogitsFn &logits_of, const VolumeData &data) {
  nn::NoGradGuard guard;
  std::vector<nn::Tensor> predictions;
  for (int row = 0; row < data.count; ++row) {
    const auto logits = logits_of(VolumeBatch(data, {row}).first);
    if (logits.shape() != nn::Shape{1, data.side, data.side, data.side, 2})
      throw std::invalid_argument("binary volume logits geometry mismatch");
    const auto probability = nn::softmax(logits, 4);
    if (!nn::all(nn::logical_and(nn::greater_equal(probability, 0.),
                                 nn::less_equal(probability, 1.)))
             .item<bool>() ||
        !nn::all(nn::less_equal(nn::abs(nn::sum(probability, {4}) - 1.), 1e-6)).item<bool>())
      throw std::invalid_argument("invalid volume probabilities");
    predictions.push_back(nn::argmax(logits, 4));
  }
  if (predictions.empty())
    throw std::invalid_argument("positive volume count required");
  return nn::concat(predictions, 0);
}
inline VolumeMetrics ScoreVolumes(const LogitsFn &logits_of, const VolumeData &data) {
  const auto metrics = nn::models::VolumePredictionMetrics(DensePredictions(logits_of, data),
                                                           TensorData(data).labels);
  return {metrics.dice.item(),
          metrics.iou.item(),
          metrics.accuracy.item(),
          metrics.tp.item<std::int64_t>(),
          metrics.fp.item<std::int64_t>(),
          metrics.fn.item<std::int64_t>(),
          metrics.tn.item<std::int64_t>()};
}
inline void VolumePreview(const LogitsFn &logits_of, const VolumeData &data,
                          const std::filesystem::path &filename) {
  const auto prediction = DensePredictions(logits_of, data);
  const auto rows = nn::models::VolumeSlicePreview(TensorData(data), prediction);
  nn::save_image_grid(rows,filename.string(),1,0,1,0,0,4);
}
} // namespace volumetric
