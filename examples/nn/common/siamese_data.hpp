#pragma once
#include "common/checks.hpp"
#include "datamunge/nn/metric_learning_data.hpp"
namespace metric {
namespace nn = datamunge::nn;
using nn::models::DigitBuckets;
using nn::models::MakeSiameseBatch;
using nn::models::OneShotTask;
using nn::models::SiameseBatch;
using nn::models::SiameseBuckets;
using nn::models::SiameseClasses;
using nn::models::SiameseImages;
using nn::models::SiamesePairs;
using nn::models::SiameseTasks;
using nn::models::VerificationPair;
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
