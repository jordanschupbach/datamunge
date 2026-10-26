// PointNet (Qi et al. 2017) with both identity-initialized T-Nets, and
// PointNet++ SSG (Qi et al. 2017b) hierarchical set abstraction.
// A port of tflow's common/nn/point_models.h (models only; tflow's
// common/ml/point_models.h training harness belongs to the pointnet ports).
//
// TF -> datamunge: models are functions of an nn::ParameterStore plus a
// "state" store for batch-norm moving statistics. Each unit is nn::dense then
// tflow's hand-written batch norm (learned gamma/beta, population moments over
// every axis but the last, epsilon .001, moving averages with momentum .9 of
// the population variance), so it is not nn::batch_norm. Moving averages are
// updated only with `update_stats` (TF runs them with the training step only).
// TF's Dropout takes a keep probability, nn::dropout a rate.
#pragma once

#include <algorithm>
#include <cstdint>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/point_geometry.hpp"

namespace point_models {

namespace nn = datamunge::nn;
using nn::Tensor;

struct PointConfig {
  int divisor = 1, classes = 40, points = 1024, first_centers = 512, second_centers = 128, first_neighbors = 32,
      second_neighbors = 64;
};

struct PointOutput {
  Tensor logits, global, input_transform, feature_transform, centroids1, centroids2, neighbors1, neighbors2, counts1,
      counts2;
};

class PointLayers {
 public:
  PointLayers(nn::ParameterStore& parameters, nn::ParameterStore& stats, bool train, bool update, int div,
              nn::Generator* dropout_gen)
      : p(parameters), state(stats), training(train), update_stats(update), divisor(div), dropout(dropout_gen) {
    if (divisor < 1 || divisor > 16) throw std::invalid_argument("point model width divisor");
    if (training && !dropout) throw std::invalid_argument("point model training needs a dropout generator");
  }
  [[nodiscard]] int Width(int full) const { return std::max(1, full / divisor); }

  /// relu(BN(x W + b)) over the last axis: "<name>/W", "<name>/b", "<name>/bn/{gamma,beta}",
  /// moving "<name>/bn/{mean,variance}" in the state store.
  Tensor Unit(const std::string& name, const Tensor& x, int out) {
    const Tensor y        = nn::dense(p, name, x, out);
    const Tensor gamma    = p.get(name + "/bn/gamma", {out}, nn::init::ones());
    const Tensor beta     = p.get(name + "/bn/beta", {out}, nn::init::zeros());
    const Tensor mean     = state.get(name + "/bn/mean", {out}, nn::init::zeros());
    const Tensor variance = state.get(name + "/bn/variance", {out}, nn::init::ones());
    Tensor       mu = mean.detach(), var = variance.detach();
    if (training) {
      nn::Axes axes;
      for (std::int64_t i = 0; i + 1 < y.dim(); ++i) axes.push_back(i);
      mu                    = nn::mean(y, axes);
      const Tensor centered = y - mu;
      var                   = nn::mean(centered * centered, axes);
      if (update_stats) {
        nn::NoGradGuard guard;
        state.set(name + "/bn/mean", mean.detach() * .9 + mu.detach() * .1);
        state.set(name + "/bn/variance", variance.detach() * .9 + var.detach() * .1);
      }
    }
    return nn::relu((y - mu) * (gamma * nn::rsqrt(var + .001)) + beta);
  }

  /// T-Net: shared MLP 64/128/1024, max over points, FC 512/256 and a zero-initialized
  /// head plus the identity, so it starts as exactly I. Returns [B, d, d].
  Tensor Transform(const std::string& name, const Tensor& features, int dimension) {
    Tensor x = Unit(name + "/mlp1", features, Width(64));
    x        = Unit(name + "/mlp2", x, Width(128));
    x        = Unit(name + "/mlp3", x, Width(1024));
    x        = nn::max(x, {1});
    x        = Unit(name + "/fc1", x, Width(512));
    x        = Unit(name + "/fc2", x, Width(256));
    const Tensor w        = p.get(name + "/head/W", {Width(256), dimension * dimension}, nn::init::zeros());
    const Tensor b        = p.get(name + "/head/b", {dimension * dimension}, nn::init::zeros());
    const Tensor identity = nn::reshape(nn::eye(dimension), {dimension * dimension});
    return nn::reshape(nn::bias_add(nn::matmul(x, w), b) + identity, {-1, dimension, dimension});
  }

  /// FC 512 / dropout / FC 256 / dropout / linear head "classifier/head".
  Tensor Classifier(const Tensor& global, int classes, double keep) {
    Tensor x = Unit("classifier/fc1", global, Width(512));
    if (training) x = nn::dropout(x, 1 - keep, true, *dropout);
    x = Unit("classifier/fc2", x, Width(256));
    if (training) x = nn::dropout(x, 1 - keep, true, *dropout);
    return nn::dense(p, "classifier/head", x, classes);
  }

 private:
  nn::ParameterStore& p;
  nn::ParameterStore& state;
  bool                training, update_stats;
  int                 divisor;
  nn::Generator*      dropout;
};

/// PointNet classifier on xyz [B, N, 3] (classifier dropout keep .7 while training).
inline PointOutput PointNet(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& xyz,
                            const PointConfig& cfg = {}, bool training = false, bool update_stats = false,
                            nn::Generator* dropout_gen = nullptr) {
  if (cfg.classes < 2) throw std::invalid_argument("PointNet classes");
  PointOutput out;
  PointLayers layer(p, state, training, training && update_stats, cfg.divisor, dropout_gen);
  out.input_transform   = layer.Transform("input_tnet", xyz, 3);
  Tensor x              = nn::matmul(xyz, out.input_transform);
  x                     = layer.Unit("point_mlp1", x, layer.Width(64));
  x                     = layer.Unit("point_mlp2", x, layer.Width(64));
  out.feature_transform = layer.Transform("feature_tnet", x, layer.Width(64));
  x                     = nn::matmul(x, out.feature_transform);
  x                     = layer.Unit("point_mlp3", x, layer.Width(64));
  x                     = layer.Unit("point_mlp4", x, layer.Width(128));
  x                     = layer.Unit("point_mlp5", x, layer.Width(1024));
  out.global            = nn::max(x, {1});
  out.logits            = layer.Classifier(out.global, cfg.classes, .7);
  return out;
}

/// Feature-transform orthogonality penalty .0005 sum((A A^T - I)^2) (= .001 L2Loss) of A [B, d, d].
inline Tensor PointTransformPenalty(const Tensor& transform, int dimension) {
  const Tensor error = nn::matmul(transform, transform, false, true) - nn::eye(dimension, transform.dtype());
  return nn::sum(error * error) * .0005;
}

/// PointNet++ SSG classifier on xyz [B, N, 3]: two FPS/ball-query set abstractions
/// (radius .2 and .4), a global abstraction and the classifier (dropout keep .5).
inline PointOutput PointNetPlusPlus(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& xyz,
                                    const PointConfig& cfg = {}, bool training = false, bool update_stats = false,
                                    nn::Generator* dropout_gen = nullptr) {
  if (cfg.classes < 2 || cfg.points < cfg.first_centers || cfg.first_centers < cfg.second_centers)
    throw std::invalid_argument("PointNet++ dimensions");
  PointOutput out;
  PointLayers layer(p, state, training, training && update_stats, cfg.divisor, dropout_gen);
  const auto  first = point_geometry::PointGroup(xyz, cfg.first_centers, .2f, cfg.first_neighbors);
  out.centroids1    = first.centers;
  out.neighbors1    = first.indices;
  out.counts1       = first.counts;
  Tensor x          = layer.Unit("sa1/mlp1", first.relative, layer.Width(64));
  x                 = layer.Unit("sa1/mlp2", x, layer.Width(64));
  x                 = layer.Unit("sa1/mlp3", x, layer.Width(128));
  x                 = nn::max(x, {2});
  const auto second = point_geometry::PointGroup(first.centers, cfg.second_centers, .4f, cfg.second_neighbors);
  out.centroids2    = second.centers;
  out.neighbors2    = second.indices;
  out.counts2       = second.counts;
  x                 = nn::concat({second.relative, point_geometry::PointGather(x, second.indices)}, 3);
  x                 = layer.Unit("sa2/mlp1", x, layer.Width(128));
  x                 = layer.Unit("sa2/mlp2", x, layer.Width(128));
  x                 = layer.Unit("sa2/mlp3", x, layer.Width(256));
  x                 = nn::max(x, {2});
  x                 = nn::expand_dims(nn::concat({second.centers, x}, 2), 1);
  x                 = layer.Unit("sa3/mlp1", x, layer.Width(256));
  x                 = layer.Unit("sa3/mlp2", x, layer.Width(512));
  x                 = layer.Unit("sa3/mlp3", x, layer.Width(1024));
  out.global        = nn::reshape(nn::max(x, {2}), {-1, layer.Width(1024)});
  out.logits        = layer.Classifier(out.global, cfg.classes, .5);
  return out;
}

}  // namespace point_models
