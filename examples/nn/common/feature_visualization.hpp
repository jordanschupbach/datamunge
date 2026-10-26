// Selected-channel activation maximization with explicit L2 and squared-TV priors:
// a port of tflow's common/nn/feature_visualization.h (the ascent itself is
// dream::DreamAscent, common/dream.hpp).
#pragma once

#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/dream.hpp"

namespace featurevis {

namespace nn = datamunge::nn;
using nn::Tensor;

struct FeatureObjective {
  Tensor feature, l2, tv, per_image, mean;  // [B] x4, scalar
};

/// Per image: spatial mean of the selected channel of `activations` [B, H, W, count]
/// minus l2_weight * mean((x - .5)^2) minus tv_weight * (mean dy^2 + mean dx^2) of
/// `image` [B, H, W, 1]; `mean` averages over the batch (so d mean / dx has a 1/B factor).
inline FeatureObjective VisualizeFeature(const Tensor& image, const Tensor& activations, const Tensor& channels, int count,
                                         float l2_weight, float tv_weight) {
  FeatureObjective out;
  const Tensor     spatial = nn::mean(activations, {1, 2});
  out.feature = nn::sum(nn::mul(spatial, nn::one_hot(channels, count, 1.0, 0.0, spatial.dtype())), {1});
  out.l2      = nn::mean(nn::square(nn::sub(image, double(.5f))), {1, 2, 3});
  const auto h = image.size(1), w = image.size(2);
  const Tensor dy = nn::sub(nn::slice_axis(image, 1, 1, h), nn::slice_axis(image, 1, 0, h - 1));
  const Tensor dx = nn::sub(nn::slice_axis(image, 2, 1, w), nn::slice_axis(image, 2, 0, w - 1));
  out.tv          = nn::add(nn::mean(nn::square(dy), {1, 2, 3}), nn::mean(nn::square(dx), {1, 2, 3}));
  out.per_image   = nn::sub(out.feature, nn::add(nn::mul(out.l2, double(l2_weight)), nn::mul(out.tv, double(tv_weight))));
  out.mean        = nn::mean(out.per_image, {0});
  return out;
}

/// n independent 28x28 uniform (0,1) noise images from mt19937(seed): (u + .5) / 2^32.
inline Tensor FeatureNoise(int n, std::uint32_t seed) {
  check::Require(n > 0, "positive number of noise seeds");
  std::mt19937       rng(seed);
  std::vector<float> x(std::size_t(n) * 784);
  for (auto& v : x) v = float((double(rng()) + .5) / 4294967296.);
  return nn::from_vector(x, {n, 28, 28, 1});
}

}  // namespace featurevis
