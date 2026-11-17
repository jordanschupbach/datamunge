#pragma once
#include "common/checks.hpp"
#include "datamunge/nn/metric_learning_data.hpp"
namespace metric {
namespace nn = datamunge::nn;
using datamunge::nn::models::DigitBuckets;
using datamunge::nn::models::MakeSiameseBatch;
using datamunge::nn::models::OneShotTask;
using datamunge::nn::models::SiameseBatch;
using datamunge::nn::models::SiameseBuckets;
using datamunge::nn::models::SiameseClasses;
using datamunge::nn::models::SiameseImages;
using datamunge::nn::models::SiamesePairs;
using datamunge::nn::models::SiameseTasks;
using datamunge::nn::models::VerificationPair;
inline std::vector<int> Labels(const nn::ImageDataset &data) {
  return nn::models::MetricDatasetLabels(data);
}
inline int BestOneShot(const std::vector<double> &scores) {
  if (scores.empty())
    throw std::invalid_argument("nonempty one-shot candidates required");
  const auto values = nn::from_vector(scores);
  if (!nn::all(nn::less(nn::abs(values), std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("finite one-shot scores required");
  return int(nn::argmax(values, 0).item<std::int64_t>());
}
} // namespace metric
