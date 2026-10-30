#pragma once
#include "common/checks.hpp"
#include "datamunge/nn/flower_data.hpp"
namespace fine {
namespace nn = datamunge::nn;
using nn::models::FlowerData;
using nn::models::FlowerRows;
using nn::models::FlowerSplits;
using nn::models::LoadFlowers;
struct FlowerMetrics {
  double nll = 0, accuracy = 0, macro_accuracy = 0, brier = 0;
  std::array<int, 36> confusion{};
};

inline FlowerMetrics ScoreFlowers(const nn::Tensor &probability, const nn::Tensor &labels) {
  const auto value = nn::models::ScoreFlowers(probability, labels);
  FlowerMetrics result;
  result.nll = value.nll.item();
  result.accuracy = value.accuracy.item();
  result.macro_accuracy = value.macro_accuracy.item();
  result.brier = value.brier.item();
  const auto counts = value.confusion.to_vector<int>();
  std::copy(counts.begin(), counts.end(), result.confusion.begin());
  return result;
}
inline std::vector<float> FlowerColorMeans(const FlowerData &data) {
  return nn::models::FlowerColorMeans(data).to_vector<float>();
}
using nn::models::FlowerColorBaseline;
} // namespace fine
