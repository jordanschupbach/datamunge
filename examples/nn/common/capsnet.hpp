// Capsule networks with dynamic routing between capsules (CapsNet): a port of
// tflow's common/nn/capsnet.h.
// https://arxiv.org/abs/1710.09829
//
// squash(s) = |s|^2 / (1 + |s|^2) * s / |s| keeps a vector's direction and
// maps its length into [0, 1). Routing by agreement: with logits b_ij = 0, r
// times compute c_ij = softmax_j(b_ij), s_j = sum_i c_ij u_hat_j|i,
// v_j = squash(s_j) and b_ij += u_hat_j|i . v_j, where u_hat_j|i = W_ij u_i
// are the votes of input capsule i for output capsule j. Gradients flow
// through every iteration. The margin loss is
// sum_k T_k max(0, 0.9 - |v_k|)^2 + 0.5 (1 - T_k) max(0, |v_k| - 0.1)^2.
// The MNIST network is a 9x9 convolution to 256 channels with ReLU, a 9x9
// stride-two convolution forming 32 types of 8-D primary capsules (6x6 each,
// 1152 in all), and 10 16-D digit capsules from three routing iterations. A
// decoder (fully connected 512, 1024 and 784 units, ReLU, ReLU, sigmoid)
// reconstructs the image from the digit capsules masked to the true class
// (the predicted one at inference); its summed squared error, scaled by
// 0.0005, joins the margin loss. Kernels are He-uniform fan-in, the decoder
// Glorot-uniform and the vote matrices N(0, 0.01), as in tflow.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace capsnet {

namespace nn = datamunge::nn;

/// Squash along the last axis.
inline nn::Tensor Squash(const nn::Tensor& x, double epsilon = 1e-9) {
  const nn::Tensor squared = nn::sum(nn::square(x), {-1}, true);
  return squared / (squared + 1.0) * (x * nn::rsqrt(squared + epsilon));
}

struct RoutingOutputs {
  nn::Tensor              votes, couplings, capsules;
  std::vector<nn::Tensor> history;  // couplings of every iteration
};

/// u: [N, I, in_dim] input capsules; W: [I, J*out_dim, in_dim]. Returns capsules [N, J, out_dim].
inline RoutingOutputs DynamicRouting(const nn::Tensor& u, const nn::Tensor& weights, std::int64_t inputs,
                                     std::int64_t outputs, std::int64_t in_dim, std::int64_t out_dim, int iterations) {
  if (inputs < 1 || outputs < 1 || in_dim < 1 || out_dim < 1 || iterations < 1)
    throw std::invalid_argument("invalid capsule routing");
  RoutingOutputs r;
  // Votes [N, I, J, out_dim] from one batched product per input capsule.
  const nn::Tensor per_capsule = nn::transpose(u, {1, 2, 0});         // [I, in_dim, N]
  const nn::Tensor products    = nn::matmul(weights, per_capsule);    // [I, J*out, N]
  r.votes = nn::reshape(nn::transpose(products, {2, 0, 1}), {-1, inputs, outputs, out_dim});
  nn::Tensor logits = nn::zeros_like(nn::sum(r.votes, {3}));          // [N, I, J]
  for (int t = 0; t < iterations; ++t) {
    r.couplings = nn::softmax(logits, -1);                            // over j
    const nn::Tensor weighted = nn::sum(nn::expand_dims(r.couplings, 3) * r.votes, {1});  // [N, J, out]
    r.capsules = Squash(weighted);
    r.history.push_back(r.couplings);
    if (t + 1 < iterations) logits = logits + nn::sum(r.votes * nn::expand_dims(r.capsules, 1), {3});
  }
  return r;
}

inline nn::Tensor CapsuleLengths(const nn::Tensor& capsules) {
  return nn::sqrt(nn::sum(nn::square(capsules), {-1}) + 1e-9);
}

/// Mean over the batch of the per-example margin loss; one_hot [N, classes].
inline nn::Tensor MarginLoss(const nn::Tensor& lengths, const nn::Tensor& one_hot, double plus = .9, double minus = .1,
                             double down = .5) {
  const nn::Tensor present   = nn::square(nn::relu(plus - lengths));
  const nn::Tensor absent    = nn::square(nn::relu(lengths - minus));
  const nn::Tensor per_class = one_hot * present + (1.0 - one_hot) * absent * down;
  return nn::mean(nn::sum(per_class, {1}));
}

struct CapsNetOutputs {
  nn::Tensor primary, digits, lengths, logits;
  /// Decoder output [N, 784] in (0, 1) from the digit capsules masked by `mask` [N, 10].
  std::function<nn::Tensor(const nn::Tensor& mask)> reconstruct;
};

inline CapsNetOutputs CapsNet(nn::ParameterStore& p, const nn::Tensor& image, int iterations = 3, bool decoder = true) {
  CapsNetOutputs       out;
  nn::ConvLayerOptions cfg;
  cfg.init       = nn::init::he_uniform();
  cfg.padding    = nn::Padding::Valid;
  cfg.activation = nn::Activation::Relu;
  const nn::Tensor h = nn::conv2d(p, "conv1", image, 256, 9, cfg);  // 20x20x256
  cfg.activation = nn::Activation::None;
  cfg.stride     = 2;
  const nn::Tensor primary = nn::conv2d(p, "primary", h, 256, 9, cfg);  // 6x6x(32x8)
  out.primary = Squash(nn::reshape(primary, {-1, 1152, 8}));
  const nn::Tensor w = p.get("digit/W", {1152, 160, 8}, nn::init::normal(.01));
  out.digits  = DynamicRouting(out.primary, w, 1152, 10, 8, 16, iterations).capsules;
  out.lengths = CapsuleLengths(out.digits);
  out.logits  = nn::log(out.lengths);  // softmax(log |v|) = |v| / sum |v|
  if (decoder) {
    // Create the decoder's variables now (tflow builds it once on the unmasked capsules).
    const std::int64_t widths[4] = {160, 512, 1024, 784};
    for (int i = 0; i < 3; ++i) {
      const std::string name = "decoder/fc" + std::to_string(i + 1);
      p.get(name + "/W", {widths[i], widths[i + 1]}, nn::init::glorot_uniform());
      p.get(name + "/b", {widths[i + 1]}, nn::init::zeros());
    }
    const nn::Tensor digits = out.digits;
    nn::ParameterStore* store = &p;
    out.reconstruct = [store, digits](const nn::Tensor& mask) {
      nn::Tensor d = nn::reshape(digits * nn::expand_dims(mask, 2), {-1, 160});
      d = nn::relu(nn::dense(*store, "decoder/fc1", d, 512));
      d = nn::relu(nn::dense(*store, "decoder/fc2", d, 1024));
      return nn::sigmoid(nn::dense(*store, "decoder/fc3", d, 784));
    };
  }
  return out;
}

}  // namespace capsnet
