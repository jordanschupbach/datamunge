// Port of tflow's common/nn/total_correlation_autoencoder.h and
// common/ml/total_correlation_autoencoder.h: the finite empirical Gaussian-mixture
// total correlation (betaTCVAE with ALL 210 fitting posteriors as references) and the
// FactorVAE density-ratio discriminator, both on the compact 4-D beta-VAE of
// common/beta_variational_autoencoder.hpp. Eager: the VAE optimizer only sees VAE
// parameters and the discriminator optimizer only the discriminator, so the TF
// "frozen opposite store" checks hold by construction (and are still checked).
#pragma once

#include "common/beta_variational_autoencoder.hpp"

namespace ae {

struct EmpiricalTcOutput {
  nn::Tensor log_joint, log_product, per_case, mean;
};

/// Exact density of the finite reference mixture (z [B, d]; mu, lv [R, d] references).
/// Both reference heads remain differentiable. Reference images MUST be fitting only.
inline EmpiricalTcOutput EmpiricalTotalCorrelation(const nn::Tensor& z, const nn::Tensor& mu, const nn::Tensor& lv,
                                                   int references) {
  const float log2pi = float(std::log(2 * 3.14159265358979323846));
  auto        delta  = nn::expand_dims(z, 1) - nn::expand_dims(mu, 0);
  auto        lv0    = nn::expand_dims(lv, 0);
  auto components = -0.5 * ((nn::square(delta) * nn::exp(nn::neg(lv0)) + lv0) + double(log2pi));  // [B, R, d]
  const double log_refs = double(float(std::log(references)));
  EmpiricalTcOutput out;
  out.log_joint   = nn::logsumexp(nn::sum(components, {2}), {1}) - log_refs;
  out.log_product = nn::sum(nn::logsumexp(components, {1}) - log_refs, {1});
  out.per_case    = out.log_joint - out.log_product;
  out.mean        = nn::mean(out.per_case, {0});
  return out;
}

/// Positive scalar logit means joint q(z), negative means product of marginals. 4 -> 16 tanh -> 1.
inline nn::Tensor FactorDiscriminator(nn::ParameterStore& p, const nn::Tensor& z) {
  auto h = nn::dense(p, "hidden", z, 16, nn::Activation::Tanh);
  return nn::reshape(nn::dense(p, "ratio", h, 1), {z.size(0)});
}

class TotalCorrelationModel {
 public:
  nn::ParameterStore             parameters{2237}, discriminator{2309};
  nn::Tensor                     reference_images;
  bool                           factor;
  std::unique_ptr<nn::Optimizer> optimizer, disc_optimizer;

  struct Output {
    BetaVaeOutput net;
    nn::Tensor    tc, loss;
  };

  TotalCorrelationModel(bool use_factor, const nn::Tensor& fitting, bool training = true)
      : reference_images(Flatten(fitting).contiguous()), factor(use_factor) {
    {
      nn::NoGradGuard guard;
      BetaVariationalAutoencoder(parameters, nn::zeros({1, 784}), nn::zeros({1, 4}), 1);
      if (factor) FactorDiscriminator(discriminator, nn::zeros({1, 4}));
    }
    if (training) {
      optimizer = MakeAdam(parameters);
      if (factor) disc_optimizer = MakeAdam(discriminator);
    }
  }
  Output Forward(const nn::Tensor& images, const nn::Tensor& noise) {
    Output o;
    o.net = BetaVariationalAutoencoder(parameters, Flatten(images), noise, 1);
    if (factor) {
      o.tc = nn::mean(FactorDiscriminator(discriminator, o.net.z), {0});
    } else {
      auto h  = nn::dense(parameters, "encoder", reference_images, 32, nn::Activation::Relu);
      auto mu = nn::dense(parameters, "mu", h, 4), lv = nn::dense(parameters, "log_variance", h, 4);
      o.tc    = EmpiricalTotalCorrelation(o.net.z, mu, lv, 210).mean;
    }
    // gamma = 3 for FactorVAE; beta_TC = 4 for betaTCVAE (beta_TC - 1 = 3).
    o.loss = o.net.unweighted_loss + 3.0 * o.tc;
    return o;
  }
  /// {mu, log_variance, z, probability, nll, kl, TC-augmented loss, unweighted loss, tc}, all finite.
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images, const nn::Tensor& noise) {
    nn::NoGradGuard         guard;
    auto                    o = Forward(images, noise);
    std::vector<nn::Tensor> out{o.net.mu,  o.net.log_variance, o.net.z,
                                o.net.probability, o.net.reconstruction_nll, o.net.kl,
                                o.loss,    o.net.unweighted_loss,   o.tc};
    for (auto& t : out) {
      t = t.contiguous();
      for (float v : t.to_vector<float>())
        check::Require(std::isfinite(v), "ALL posterior/output/objective values finite");
    }
    return out;
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::sigmoid(nn::dense(parameters, "output",
                                 nn::dense(parameters, "decoder", codes, 32, nn::Activation::Relu), 784))
        .contiguous();
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& noise) {
    return nn::grad(Forward(images, noise).loss, parameters.parameters());
  }
  double Update(const nn::Tensor& images, const nn::Tensor& noise) {
    auto         loss  = Forward(images, noise).loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value), "finite TC-augmented objective");
    optimizer->step(nn::grad(loss, parameters.parameters()));
    return value;
  }
  void Load(const nn::NamedTensors& arrays) { LoadArrays(parameters, arrays); }

  nn::Tensor DiscLoss(const nn::Tensor& joint, const nn::Tensor& product) {
    auto j = FactorDiscriminator(discriminator, joint), p = FactorDiscriminator(discriminator, product);
    return 0.5 * (nn::mean(nn::softplus(nn::neg(j)), {0}) + nn::mean(nn::softplus(p), {0}));
  }
  std::vector<nn::Tensor> DiscGradient(const nn::Tensor& joint, const nn::Tensor& product) {
    return nn::grad(DiscLoss(joint, product), discriminator.parameters());
  }
  /// Detached discriminator-only Adam update; returns the pre-update cross entropy.
  double DiscUpdate(const nn::Tensor& joint, const nn::Tensor& product) {
    auto         loss  = DiscLoss(joint.detach(), product.detach());
    const double value = loss.item<float>();
    disc_optimizer->step(nn::grad(loss, discriminator.parameters()));
    return value;
  }
  nn::NamedTensors DiscArrays() const { return factor ? Arrays(discriminator) : nn::NamedTensors{}; }
};

/// {NLL, KL, negative ELBO, NLL + KL + 3 mean TC} over every sample of `noise`.
inline std::vector<double> TotalCorrelationScore(TotalCorrelationModel& model, const nn::Tensor& images,
                                                 const VaeNoiseBank& noise) {
  double             nll = 0, kl = 0, tc = 0;
  const std::int64_t n = images.size(0), samples = noise.epsilon.size(0);
  for (std::int64_t k = 0; k < samples; ++k) {
    auto       out = model.Evaluate(images, BetaVaeNoiseAt(noise, k));
    const auto a = out[4].to_vector<float>(), b = out[5].to_vector<float>();
    for (std::int64_t i = 0; i < n; ++i) {
      nll += a[i];
      if (k == 0) kl += b[i];
    }
    tc += out[8].item<float>();
  }
  nll /= double(n * samples);
  kl /= double(n);
  tc /= double(samples);
  return {nll, kl, nll + kl, nll + kl + 3 * tc};
}

}  // namespace ae
