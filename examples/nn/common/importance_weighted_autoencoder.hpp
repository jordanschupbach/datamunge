// Port of tflow's common/nn/importance_weighted_autoencoder.h and
// common/ml/importance_weighted_autoencoder.h: the multisample IWAE bound on the
// compact 4-D beta-VAE network (beta 1): sampled log p(x,z) - log q(z|x) per draw,
// stable log-mean-exp over K draws, normalized weights and effective sample size.
#pragma once

#include <numeric>

#include "common/flow_variational_autoencoder.hpp"

namespace ae {

/// log(mean_k exp(w_k)) over axis 0 of [K, n], shifted by the (gradient-stopped) max.
inline nn::Tensor IwaeLogMeanExp(const nn::Tensor& log_weights, std::int64_t samples) {
  auto maximum = nn::stop_gradient(nn::max(log_weights, {0}, true));
  return nn::squeeze(maximum + nn::log(nn::sum(nn::exp(log_weights - maximum), {0}, true)), {0}) -
         double(float(std::log(double(samples))));
}

struct ImportanceWeightedOutput {
  nn::Tensor log_weights, normalized_weights, negative_bound, effective_samples, loss;
};

/// `net` evaluated on K x n rows stacked sample-major.
inline ImportanceWeightedOutput ImportanceWeightedBound(const BetaVaeOutput& net, std::int64_t samples) {
  const double log2pi = FlowLog2Pi();
  auto         logp   = -0.5 * nn::sum(nn::square(net.z) + log2pi, {1});
  auto logq = -0.5 * nn::sum(nn::square(net.z - net.mu) * nn::exp(-net.log_variance) + net.log_variance + log2pi, {1});
  ImportanceWeightedOutput out;
  out.log_weights    = nn::reshape(logp - net.reconstruction_nll - logq, {samples, -1});
  out.negative_bound = -IwaeLogMeanExp(out.log_weights, samples);
  out.loss           = nn::mean(out.negative_bound, {0});
  auto maximum       = nn::stop_gradient(nn::max(out.log_weights, {0}, true));
  auto unnormalized  = nn::exp(out.log_weights - maximum);
  out.normalized_weights = unnormalized / nn::sum(unnormalized, {0}, true);
  out.effective_samples  = 1.0 / nn::sum(nn::square(out.normalized_weights), {0});
  return out;
}

class ImportanceWeightedModel {
 public:
  nn::ParameterStore             parameters{2237};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit ImportanceWeightedModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      BetaVariationalAutoencoder(parameters, nn::zeros({1, 784}), nn::zeros({1, 4}), 1);
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  struct Forward {
    BetaVaeOutput            net;
    ImportanceWeightedOutput importance;
  };
  /// images [n, 28, 28, 1]; noise [K, n, 4]. Images are tiled K times sample-major.
  Forward Run(const nn::Tensor& images, const nn::Tensor& noise) {
    check::Require(noise.dim() == 3 && noise.size(0) > 0 && noise.size(1) == images.size(0) && noise.size(2) == 4,
                   "noise has K,B,4 axes");
    const std::int64_t k = noise.size(0), b = images.size(0);
    auto tiled   = nn::reshape(nn::tile(nn::reshape(Flatten(images), {1, b, 784}), {k, 1, 1}), {k * b, 784});
    auto epsilon = nn::reshape(noise, {k * b, 4});
    Forward f;
    f.net        = BetaVariationalAutoencoder(parameters, tiled.contiguous(), epsilon.contiguous(), 1);
    f.importance = ImportanceWeightedBound(f.net, k);
    return f;
  }
  /// {mu, log_variance, z, probability, nll, kl, loss, negative_bound, log_weights, normalized_weights, ess}
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images, const nn::Tensor& noise) {
    nn::NoGradGuard         guard;
    auto                    f = Run(images, noise);
    std::vector<nn::Tensor> out{f.net.mu,
                                f.net.log_variance,
                                f.net.z,
                                f.net.probability,
                                f.net.reconstruction_nll,
                                f.net.kl,
                                f.importance.loss,
                                f.importance.negative_bound,
                                f.importance.log_weights,
                                f.importance.normalized_weights,
                                f.importance.effective_samples};
    for (auto& t : out) {
      t = t.contiguous();
      for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "ALL posterior/output/objective values finite");
    }
    return out;
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::sigmoid(
               nn::dense(parameters, "output", nn::dense(parameters, "decoder", codes, 32, nn::Activation::Relu), 784))
        .contiguous();
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& noise) {
    return nn::grad(Run(images, noise).importance.loss, parameters.parameters());
  }
  double Update(const nn::Tensor& images, const nn::Tensor& noise) {
    auto         loss  = Run(images, noise).importance.loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value), "finite IWAE objective");
    optimizer->minimize(loss);
    return value;
  }
  void Load(const nn::NamedTensors& arrays) { LoadArrays(parameters, arrays); }
};

/// The first `samples` draws of `bank` (float32 [S, n, 4]) at `rows`, as [samples, rows, 4].
inline nn::Tensor IwaeNoiseRows(const VaeNoiseBank& bank, const std::vector<std::int64_t>& rows, std::int64_t samples) {
  check::Require(samples > 0 && samples <= bank.epsilon.size(0), "available importance samples");
  auto first = nn::slice_axis(bank.epsilon, 0, 0, samples);
  return nn::gather(first, nn::from_vector(rows, {std::int64_t(rows.size())}), 1).contiguous();
}
inline std::vector<std::int64_t> IwaeRows(std::int64_t n) {
  std::vector<std::int64_t> out(static_cast<std::size_t>(n));
  std::iota(out.begin(), out.end(), 0);
  return out;
}

}  // namespace ae
