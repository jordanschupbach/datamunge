// Port of tflow's common/nn/info_variational_autoencoder.h and
// common/ml/info_variational_autoencoder.h: a small Gaussian InfoVAE / MMD-VAE
// (784 -> 16 tanh -> Gaussian 4, decoder 4 -> 16 tanh -> 784 Bernoulli) trained on
// mean(NLL + (1 - alpha) KL) + (alpha + lambda - 1) aggregate IMQ MMD between the
// sampled posterior codes and an explicit independent N(0, I) prior sample.
#pragma once

#include "common/beta_variational_autoencoder.hpp"
#include "common/kernel_discrepancy.hpp"

namespace ae {

/// mean(nll + (1 - alpha) kl) + (alpha + lambda - 1) mmd, alpha in [0, 1], lambda >= 1.
inline nn::Tensor InfoVaeObjective(const nn::Tensor& nll, const nn::Tensor& kl, const nn::Tensor& mmd, float alpha,
                                   float lambda) {
  check::Require(std::isfinite(alpha) && alpha >= 0 && alpha <= 1 && std::isfinite(lambda) && lambda >= 1,
                 "InfoVAE alpha0..1/lambda>=1");
  return nn::mean(nll + double(1 - alpha) * kl, {0}) + double(alpha + lambda - 1) * mmd;
}

struct InfoVaeOutput {
  nn::Tensor mu, log_variance, z, logits, probability, nll, kl, negative_elbo, loss;
  MmdOutput  mmd;
};

inline nn::Tensor InfoVaeDecodeLogits(nn::ParameterStore& p, const nn::Tensor& z) {
  return nn::dense(p, "output", nn::dense(p, "decoder", z, 16, nn::Activation::Tanh), 784);
}

inline InfoVaeOutput InfoVariationalAutoencoder(nn::ParameterStore& p, const nn::Tensor& x, const nn::Tensor& epsilon,
                                                const nn::Tensor& prior, float alpha, float lambda) {
  InfoVaeOutput o;
  auto          h = nn::dense(p, "encoder", x, 16, nn::Activation::Tanh);
  o.mu            = nn::dense(p, "mu", h, 4);
  o.log_variance  = nn::dense(p, "log_variance", h, 4);
  o.z             = VaeReparameterize(o.mu, o.log_variance, epsilon);
  o.logits        = InfoVaeDecodeLogits(p, o.z);
  o.probability   = nn::sigmoid(o.logits);
  o.nll           = VaePixelNll(o.logits, x);
  o.kl            = VaeKl(o.mu, o.log_variance);
  o.negative_elbo = nn::mean(o.nll + o.kl, {0});
  o.mmd           = ImqMmd(o.z, prior);
  o.loss          = InfoVaeObjective(o.nll, o.kl, o.mmd.value, alpha, lambda);
  return o;
}

class InfoVariationalAutoencoderModel {
 public:
  nn::ParameterStore             parameters{3901};
  std::unique_ptr<nn::Optimizer> optimizer;
  float                          alpha, lambda;

  explicit InfoVariationalAutoencoderModel(float alpha_, float lambda_, bool training = true)
      : alpha(alpha_), lambda(lambda_) {
    {
      nn::NoGradGuard guard;
      InfoVariationalAutoencoder(parameters, nn::zeros({2, 784}), nn::zeros({2, 4}), nn::zeros({2, 4}), alpha, lambda);
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  InfoVaeOutput Forward(const nn::Tensor& images, const nn::Tensor& noise, const nn::Tensor& prior) {
    check::Require(images.size(0) >= 2 && prior.size(0) >= 2, "unbiased MMD needs at least2samples EACH");
    return InfoVariationalAutoencoder(parameters, Flatten(images), noise, prior, alpha, lambda);
  }
  /// tflow's named fields, all checked finite.
  nn::NamedTensors Evaluate(const nn::Tensor& images, const nn::Tensor& noise, const nn::Tensor& prior) {
    nn::NoGradGuard  guard;
    auto             n = Forward(images, noise, prior);
    nn::NamedTensors out{{"mu", n.mu},
                         {"log_variance", n.log_variance},
                         {"z", n.z},
                         {"logits", n.logits},
                         {"probability", n.probability},
                         {"nll", n.nll},
                         {"kl", n.kl},
                         {"negative_elbo", n.negative_elbo},
                         {"mmd_xx", n.mmd.xx},
                         {"mmd_yy", n.mmd.yy},
                         {"mmd_xy", n.mmd.xy},
                         {"mmd_within_x", n.mmd.within_x},
                         {"mmd_within_y", n.mmd.within_y},
                         {"mmd_cross", n.mmd.cross},
                         {"mmd", n.mmd.value},
                         {"loss", n.loss}};
    for (auto& [name, t] : out) {
      t = t.contiguous();
      for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "all InfoVAE fields finite");
    }
    return out;
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::sigmoid(InfoVaeDecodeLogits(parameters, codes)).contiguous();
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& noise, const nn::Tensor& prior) {
    return nn::grad(Forward(images, noise, prior).loss, parameters.parameters());
  }
  double Update(const nn::Tensor& images, const nn::Tensor& noise, const nn::Tensor& prior) {
    auto         loss  = Forward(images, noise, prior).loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value), "finite InfoVAE objective");
    optimizer->minimize(loss);
    return value;
  }
  void Load(const nn::NamedTensors& arrays) { LoadArrays(parameters, arrays); }
};

}  // namespace ae
