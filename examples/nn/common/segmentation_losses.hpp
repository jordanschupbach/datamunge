#pragma once
#include "datamunge/nn/segmentation_losses.hpp"
namespace seg {
namespace nn = datamunge::nn;
using datamunge::nn::models::BoundaryLoss;
using datamunge::nn::models::LovaszSoftmaxLoss;
using datamunge::nn::models::SignedBoundaryDistances;
using datamunge::nn::models::SoftDiceLoss;
using datamunge::nn::models::SoftTverskyLoss;
inline std::vector<double> SquaredDistance1D(const std::vector<double> &f) {
  return nn::models::SquaredDistance1D(nn::from_vector(f)).to_vector<double>();
}
inline std::vector<double> SquaredDistance2D(const std::vector<bool> &seeds, int height,
                                             int width) {
  if (height < 1 || width < 1 || seeds.size() != std::size_t(height) * std::size_t(width))
    throw std::invalid_argument("invalid distance grid");
  return nn::models::SquaredDistance2D(nn::from_vector(seeds, {height, width}))
      .to_vector<double>();
}
} // namespace seg
