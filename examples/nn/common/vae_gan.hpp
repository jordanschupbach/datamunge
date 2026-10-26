// Port of tflow's common/nn/vae_gan.h and common/ml/vae_gan.h: a compact VAE-GAN (Larsen et
// al.) with learned discriminator-feature reconstruction. Three parameter groups (encoder seed
// 3301, generator 3307, discriminator 3313), three independent Adam states (.003) and
// explicitly sequenced, isolated D -> E -> G updates (each recomputed from current weights).
#pragma once

#include "common/beta_variational_autoencoder.hpp"

namespace ae {

/// z [n, 4] -> 16 tanh -> 784 sigmoid probabilities.
inline nn::Tensor VaeGanDecode(nn::ParameterStore& g, const nn::Tensor& z) {
  return nn::sigmoid(nn::dense(g, "output", nn::dense(g, "hidden", z, 16, nn::Activation::Tanh), 784));
}

struct VaeGanDiscriminatorOutput {
  nn::Tensor feature, logit;
};
/// x [n, 784] -> 8 tanh feature -> 1 logit.
inline VaeGanDiscriminatorOutput VaeGanDiscriminator(nn::ParameterStore& d, const nn::Tensor& x) {
  auto feature = nn::dense(d, "feature", x, 8, nn::Activation::Tanh);
  return {feature, nn::dense(d, "logit", feature, 1)};
}

struct VaeGanOutput {
  nn::Tensor mu, log_variance, z, probability, prior_probability, real_feature, reconstruction_feature, prior_feature,
      real_logit, reconstruction_logit, prior_logit, feature_nll, kl, pixel_nll, mse, encoder_loss, generator_loss,
      discriminator_loss, generator_adversarial;
};

inline VaeGanOutput VaeGan(nn::ParameterStore& e, nn::ParameterStore& g, nn::ParameterStore& d, const nn::Tensor& x,
                           const nn::Tensor& epsilon, const nn::Tensor& prior) {
  VaeGanOutput o;
  auto         h     = nn::dense(e, "hidden", x, 16, nn::Activation::Tanh);
  o.mu               = nn::dense(e, "mu", h, 4);
  o.log_variance     = nn::dense(e, "log_variance", h, 4);
  o.z                = VaeReparameterize(o.mu, o.log_variance, epsilon);
  o.probability      = VaeGanDecode(g, o.z);
  o.prior_probability = VaeGanDecode(g, prior);
  auto real = VaeGanDiscriminator(d, x), rec = VaeGanDiscriminator(d, o.probability),
       generated = VaeGanDiscriminator(d, o.prior_probability);
  o.real_feature           = real.feature;
  o.reconstruction_feature = rec.feature;
  o.prior_feature          = generated.feature;
  o.real_logit             = real.logit;
  o.reconstruction_logit   = rec.logit;
  o.prior_logit            = generated.logit;
  o.feature_nll  = 0.5 * nn::sum(nn::square(real.feature.detach() - rec.feature), {1});
  o.kl           = VaeKl(o.mu, o.log_variance);
  o.encoder_loss = nn::mean(o.feature_nll + o.kl, {0});
  o.generator_adversarial = 0.5 * (nn::mean(nn::softplus(nn::neg(rec.logit)), {0, 1}) +
                                   nn::mean(nn::softplus(nn::neg(generated.logit)), {0, 1}));
  o.generator_loss        = nn::mean(o.feature_nll, {0}) + o.generator_adversarial;
  o.discriminator_loss    = nn::mean(nn::softplus(nn::neg(real.logit)), {0, 1}) +
                         0.5 * (nn::mean(nn::softplus(rec.logit), {0, 1}) +
                                nn::mean(nn::softplus(generated.logit), {0, 1}));
  auto clipped = nn::clip(o.probability, double(1e-6f), double(1.f - 1e-6f));
  o.pixel_nll  = nn::neg(nn::sum(x * nn::log(clipped) + (1.0 - x) * nn::log(1.0 - clipped), {1}));
  o.mse        = nn::mean(nn::square(x - o.probability), {1});
  return o;
}

class VaeGanModel {
 public:
  nn::ParameterStore                          encoder{3301}, generator{3307}, discriminator{3313};
  std::vector<std::unique_ptr<nn::Optimizer>> optimizers;  // E, G, D

  explicit VaeGanModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      VaeGan(encoder, generator, discriminator, nn::zeros({1, 784}), nn::zeros({1, 4}), nn::zeros({1, 4}));
    }
    if (training)
      for (auto* group : Groups()) optimizers.push_back(MakeAdam(*group));
  }
  std::vector<nn::ParameterStore*> Groups() { return {&encoder, &generator, &discriminator}; }
  std::int64_t Count() { return encoder.count() + generator.count() + discriminator.count(); }
  std::size_t  Tensors() { return encoder.size() + generator.size() + discriminator.size(); }

  VaeGanOutput Forward(const nn::Tensor& x, const nn::Tensor& e, const nn::Tensor& p) {
    return VaeGan(encoder, generator, discriminator, Flatten(x), e, p);
  }
  static nn::Tensor Objective(const VaeGanOutput& o, int group) {
    return group == 0 ? o.encoder_loss : group == 1 ? o.generator_loss : o.discriminator_loss;
  }
  /// Named fields of tflow's Evaluate, all checked finite.
  nn::NamedTensors Evaluate(const nn::Tensor& x, const nn::Tensor& e, const nn::Tensor& p) {
    nn::NoGradGuard  guard;
    auto             o = Forward(x, e, p);
    nn::NamedTensors out{{"mu", o.mu},
                         {"log_variance", o.log_variance},
                         {"z", o.z},
                         {"probability", o.probability},
                         {"prior_probability", o.prior_probability},
                         {"real_feature", o.real_feature},
                         {"reconstruction_feature", o.reconstruction_feature},
                         {"prior_feature", o.prior_feature},
                         {"real_logit", o.real_logit},
                         {"reconstruction_logit", o.reconstruction_logit},
                         {"prior_logit", o.prior_logit},
                         {"feature_nll", o.feature_nll},
                         {"kl", o.kl},
                         {"pixel_nll", o.pixel_nll},
                         {"mse", o.mse},
                         {"encoder_loss", o.encoder_loss},
                         {"generator_loss", o.generator_loss},
                         {"discriminator_loss", o.discriminator_loss},
                         {"generator_adversarial", o.generator_adversarial}};
    for (auto& [name, t] : out) {
      t = t.contiguous();
      for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "all VAE-GAN fields finite");
    }
    return out;
  }
  /// Isolated group gradient named "<group>/<name>".
  nn::NamedTensors Gradient(int group, const nn::Tensor& x, const nn::Tensor& e, const nn::Tensor& p) {
    auto*            store = Groups()[group];
    auto             g     = nn::grad(Objective(Forward(x, e, p), group), store->parameters());
    nn::NamedTensors out;
    for (std::size_t i = 0; i < g.size(); ++i) out.emplace_back(std::to_string(group) + "/" + store->names()[i], g[i]);
    return out;
  }
  /// One Adam update of ONE group; returns its pre-update objective.
  double Update(int group, const nn::Tensor& x, const nn::Tensor& e, const nn::Tensor& p) {
    auto         loss  = Objective(Forward(x, e, p), group);
    const double value = loss.item<float>();
    optimizers.at(group)->step(nn::grad(loss, Groups()[group]->parameters()));
    return value;
  }
  /// All arrays named "<k>/<name>" (k = 0 encoder, 1 generator, 2 discriminator).
  nn::NamedTensors Arrays() {
    nn::NamedTensors out;
    auto             stores = Groups();
    for (int k = 0; k < 3; ++k)
      for (const auto& name : stores[k]->names())
        out.emplace_back(std::to_string(k) + "/" + name, stores[k]->at(name).detach().clone());
    return out;
  }
  std::vector<std::string> Snapshot() {
    std::vector<std::string> out;
    for (const auto& [key, value] : Arrays()) out.push_back(Bytes(value));
    return out;
  }
  /// Loads "<k>/<name>" arrays (tflow's initial_arrays.npz / our best.npz).
  void Load(const nn::NamedTensors& arrays) {
    check::Require(arrays.size() == Tensors(), "all VAE-GAN tensors supplied");
    auto stores = Groups();
    for (int k = 0; k < 3; ++k)
      for (const auto& name : stores[k]->names()) {
        auto value = Find(arrays, std::to_string(k) + "/" + name);
        check::Require(value.shape() == stores[k]->at(name).shape(), "matching VAE-GAN shapes");
        stores[k]->set(name, value.contiguous().clone());
      }
  }
  /// Code-only generation.
  nn::Tensor Decode(const nn::Tensor& z) {
    nn::NoGradGuard guard;
    return VaeGanDecode(generator, z).contiguous();
  }
};

/// Monte Carlo {pixel MSE, diagnostic pixel NLL, feature NLL, KL} per image over 4 draws.
inline std::vector<double> VaeGanScore(VaeGanModel& m, const nn::Tensor& images, const VaeNoiseBank& noise,
                                       const VaeNoiseBank& prior) {
  std::vector<double> sum(4);
  const std::int64_t  n = images.size(0);
  for (int k = 0; k < 4; ++k) {
    auto f = m.Evaluate(images, BetaVaeNoiseAt(noise, k), BetaVaeNoiseAt(prior, k));
    int  j = 0;
    for (const std::string key : {"mse", "pixel_nll", "feature_nll", "kl"}) {
      for (float v : Find(f, key).to_vector<float>()) sum[j] += v;
      ++j;
    }
  }
  for (auto& value : sum) value /= double(4 * n);
  return sum;
}

}  // namespace ae
