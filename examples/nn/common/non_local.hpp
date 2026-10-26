// Non-local blocks: every position attends to every other position (port of
// tflow's common/nn/non_local.h).
// https://arxiv.org/abs/1711.07971
// https://github.com/facebookresearch/video-nonlocal-net (lib/models/nonlocal_helper.py)
//
// y_i = (1/C(x)) sum_j f(x_i, x_j) g(x_j) and z = BN(W_z y) + x, where g, the
// embeddings theta and phi, and W_z are biased 1x1 convolutions with half the
// input width inside. The pairwise function f is one of the paper's four:
// embedded Gaussian (softmax over j of theta_i . phi_j, scaled by
// 1/sqrt(inner width) as in the release), Gaussian (softmax of x_i . x_j,
// no embeddings), dot product (theta_i . phi_j / N) and concatenation
// (ReLU(w . [theta_i, phi_j]) / N). Optional 2x2 max-pooling of phi and g
// cuts the pairwise cost by four. BN's scale starts at zero, so a new block
// is the identity. Kernels are He-uniform fan-in (tflow's initialization
// adaptation); BN epsilon is 1e-5.
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>

namespace non_local {

namespace nn = datamunge::nn;

enum class Pairwise { kEmbeddedGaussian, kGaussian, kDotProduct, kConcatenation };

struct NonLocalOutputs {
  nn::Tensor attention, output;  // attention [N, P, M]
};

/// Non-local block on x [N, H, W, C]. While `training` its BN uses batch
/// statistics (updating the moving averages in `state` when `update_stats`).
inline NonLocalOutputs NonLocalBlock(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                     const nn::Tensor& x, Pairwise f, bool subsample, bool training,
                                     double momentum = .9, bool update_stats = true) {
  const std::int64_t n = x.shape()[0], height = x.shape()[1], width = x.shape()[2], channels = x.shape()[3],
                     inner = channels / 2;
  if (inner < 1) throw std::invalid_argument("non-local blocks need at least two channels");
  nn::ConvLayerOptions cfg;
  cfg.init  = nn::init::he_uniform();
  auto conv = [&](const std::string& part, const nn::Tensor& in, std::int64_t to) {
    return nn::conv2d(p, name + "/" + part, in, to, 1, cfg);
  };
  auto flatten = [&](const nn::Tensor& t) { return nn::reshape(t, {n, -1, t.shape()[3]}); };  // [N, H*W, C]
  const nn::Tensor pooled = subsample ? nn::max_pool2d(x, 2, 2, nn::Padding::Valid) : x;
  const nn::Tensor g      = flatten(conv("g", pooled, inner));  // [N, M, inner]
  nn::Tensor       affinity;
  if (f == Pairwise::kGaussian) {
    affinity = nn::matmul(flatten(x), flatten(pooled), false, true);
  } else {
    const nn::Tensor theta = flatten(conv("theta", x, inner)), phi = flatten(conv("phi", pooled, inner));
    if (f == Pairwise::kConcatenation) {
      // w . [theta_i, phi_j] = w1 . theta_i + w2 . phi_j for every pair (i, j);
      // He-uniform with the fan-in 2*inner of the concatenated vector.
      const auto       init = nn::init::uniform(std::sqrt(6.0 / double(2 * inner)));
      const nn::Tensor w1 = p.get(name + "/concat/W1", {inner, 1}, init), w2 = p.get(name + "/concat/W2", {inner, 1}, init);
      const nn::Tensor a = nn::matmul(theta, w1), b = nn::matmul(phi, w2);  // [N, P, 1], [N, M, 1]
      affinity           = nn::relu(a + nn::transpose(b, {0, 2, 1}));
    } else {
      affinity = nn::matmul(theta, phi, false, true);  // [N, P, M]
      if (f == Pairwise::kEmbeddedGaussian) affinity = affinity * (1 / std::sqrt(double(inner)));
    }
  }
  NonLocalOutputs r;
  if (f == Pairwise::kEmbeddedGaussian || f == Pairwise::kGaussian)
    r.attention = nn::softmax(affinity, -1);
  else
    r.attention = affinity / double(affinity.shape()[2]);
  const nn::Tensor y = nn::reshape(nn::matmul(r.attention, g), {n, height, width, inner});
  const nn::Tensor w = conv("W", y, channels);
  p.get(name + "/bn/gamma", {channels}, nn::init::zeros());  // zero scale: the block starts as the identity
  nn::BatchNormOptions bn;
  bn.momentum     = momentum;
  bn.eps          = 1e-5;
  bn.update_stats = update_stats;
  r.output        = nn::batch_norm(p, state, name + "/bn", w, training, bn) + x;
  return r;
}

}  // namespace non_local
