#pragma once
#include "common/segmentation.hpp"
#include "datamunge/nn/dense_crf_models.hpp"
namespace seg {
using nn::models::DenseCrfConfig;
using nn::models::DenseCrfKernel;
using nn::models::DenseCrfMeanField;
inline void DenseCrfPreview(const LogitsFn &logits_of, const SegmentationData &data,
                            const std::filesystem::path &directory) {
  nn::NoGradGuard guard;
  const int count = std::min(4, data.count), C = data.classes;
  auto before = nn::zeros({C, C}, nn::DType::Int64),
       after = nn::zeros({C, C}, nn::DType::Int64);
  std::vector<nn::Tensor> gallery;
  for (int i = 0; i < count; ++i) {
    const auto batch = SegmentationBatch(data, {i});
    const auto image = nn::models::SampleSegmentationGrid(batch.first),
               unary = nn::models::SampleSegmentationGrid(logits_of(batch.first))
                           .reshape({1024, C}),
               truth = nn::models::SampleSegmentationGrid(
                           batch.second.reshape({1, data.height, data.width, 1}))
                           .reshape({1, 32, 32});
    const auto raw = nn::argmax(unary, 1).reshape({1, 32, 32}),
               refined =
                   nn::argmax(
                       DenseCrfMeanField(unary, DenseCrfKernel(image.reshape({32, 32, 3}))), 1)
                       .reshape({1, 32, 32});
    before = before + nn::models::SegmentationPredictionMetrics(raw, truth, C).confusion;
    after = after + nn::models::SegmentationPredictionMetrics(refined, truth, C).confusion;
    gallery.push_back(image);
    gallery.push_back(nn::models::ColorSegmentationLabels(truth));
    gallery.push_back(nn::models::ColorSegmentationLabels(raw));
    gallery.push_back(nn::models::ColorSegmentationLabels(refined));
  }
  const auto path = directory / "crf_preview.png";
  nn::save_image_grid(nn::concat(gallery, 0), path.string(), 4, 0., 1., 0, 0., 1);
  std::cout << "  fixed CRF on first" << count << " held-out scenes sampled32x32: mIoU "
            << nn::models::SegmentationConfusionMetrics(before).mean_iou.item() << " -> "
            << nn::models::SegmentationConfusionMetrics(after).mean_iou.item()
            << "; image/truth/raw/CRF preview: " << path.string() << std::endl;
}
} // namespace seg
