// Port of the Gaussian pieces of tflow's common/nn/ladder_variational_autoencoder.h
// (LadderGaussian, LadderPrecisionFusion, LadderConditionalKl). Used by the
// hierarchical VAE port. Also the two-level ladder VAE model and harness of tflow's
// common/ml/ladder_variational_autoencoder.h (LadderVaeModel, LadderNoise, LadderScore).
#pragma once

#include "common/autoencoder.hpp"
#include "common/beta_variational_autoencoder.hpp"

namespace ae {

struct LadderGaussian {
  nn::Tensor mu, log_variance;
};

/// Precision-weighted fusion of an evidence and a prior Gaussian (stable log-sum-exp of precisions).
inline LadderGaussian LadderPrecisionFusion(const nn::Tensor& evidence_mu, const nn::Tensor& evidence_lv,
                                            const nn::Tensor& prior_mu, const nn::Tensor& prior_lv) {
  auto a = -evidence_lv, b = -prior_lv;
  auto shift = nn::maximum(a, b).detach();
  auto logprecision = shift + nn::log(nn::exp(a - shift) + nn::exp(b - shift));
  LadderGaussian q;
  q.log_variance = -logprecision;
  auto wa = nn::exp(q.log_variance - evidence_lv), wb = nn::exp(q.log_variance - prior_lv);
  q.mu    = wa * evidence_mu + wb * prior_mu;
  return q;
}

/// KL(N(mu, exp(lv)) || N(pm, exp(plv))) summed over axis 1.
inline nn::Tensor LadderConditionalKl(const nn::Tensor& mu, const nn::Tensor& lv, const nn::Tensor& pm,
                                      const nn::Tensor& plv) {
  return 0.5 * nn::sum(((plv - lv) + (nn::exp(lv - plv) + nn::square(mu - pm) * nn::exp(-plv))) - 1.0, {1});
}

// ---- two-level ladder VAE (tflow's LadderVariationalAutoencoder / LadderVaeModel) ----------

/// Decoder 4 -> 32 ReLU -> 784 logits.
inline nn::Tensor LadderDecoder(nn::ParameterStore& p, const nn::Tensor& z) {
  return nn::dense(p, "output", nn::dense(p, "decoder", z, 32, nn::Activation::Relu), 784);
}

struct LadderVaeOutput {
  nn::Tensor evidence_mu, evidence_lv, top_mu, top_lv, top_z, prior_mu, prior_lv, lower_mu, lower_lv, lower_z, logits,
      probability, nll, kl_lower, kl_top, loss;
};

/// Bottom-up 784 -> 32 ReLU (4-D Gaussian evidence) -> 16 ReLU (2-D top posterior); top-down
/// z2 -> 16 ReLU -> p(z1|z2); q(z1) = precision fusion of evidence and conditional prior.
/// epsilon [n, 6]: columns 0..3 lower noise, 4..5 top noise.
inline LadderVaeOutput LadderVariationalAutoencoder(nn::ParameterStore& p, const nn::Tensor& image,
                                                    const nn::Tensor& epsilon) {
  LadderVaeOutput o;
  auto h1       = nn::dense(p, "encoder1", image, 32, nn::Activation::Relu);
  auto h2       = nn::dense(p, "encoder2", h1, 16, nn::Activation::Relu);
  o.evidence_mu = nn::dense(p, "evidence_mu", h1, 4);
  o.evidence_lv = nn::dense(p, "evidence_lv", h1, 4);
  o.top_mu      = nn::dense(p, "top_mu", h2, 2);
  o.top_lv      = nn::dense(p, "top_lv", h2, 2);
  auto e1 = nn::slice_axis(epsilon, 1, 0, 4), e2 = nn::slice_axis(epsilon, 1, 4, 6);
  o.top_z     = VaeReparameterize(o.top_mu, o.top_lv, e2);
  auto hp     = nn::dense(p, "prior_hidden", o.top_z, 16, nn::Activation::Relu);
  o.prior_mu  = nn::dense(p, "prior_mu", hp, 4);
  o.prior_lv  = nn::dense(p, "prior_lv", hp, 4);
  auto q      = LadderPrecisionFusion(o.evidence_mu, o.evidence_lv, o.prior_mu, o.prior_lv);
  o.lower_mu  = q.mu;
  o.lower_lv  = q.log_variance;
  o.lower_z   = VaeReparameterize(o.lower_mu, o.lower_lv, e1);
  o.logits    = LadderDecoder(p, o.lower_z);
  o.probability = nn::sigmoid(o.logits);
  o.nll       = VaePixelNll(o.logits, image);
  o.kl_lower  = LadderConditionalKl(o.lower_mu, o.lower_lv, o.prior_mu, o.prior_lv);
  o.kl_top    = VaeKl(o.top_mu, o.top_lv);
  o.loss      = nn::mean(o.nll + (o.kl_lower + o.kl_top), {0});
  return o;
}

/// tflow's LadderVaeModel: one store (seed 2503), Adam .003, 22 tensors / 52196 parameters.
class LadderVaeModel {
 public:
  nn::ParameterStore             parameters{2503};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit LadderVaeModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      LadderVariationalAutoencoder(parameters, nn::zeros({1, 784}), nn::zeros({1, 6}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  /// Named fields of tflow's Evaluate (all finite).
  nn::NamedTensors Evaluate(const nn::Tensor& images, const nn::Tensor& noise) {
    nn::NoGradGuard  guard;
    auto             o = LadderVariationalAutoencoder(parameters, Flatten(images), noise);
    nn::NamedTensors out{{"evidence_mu", o.evidence_mu}, {"evidence_lv", o.evidence_lv}, {"top_mu", o.top_mu},
                         {"top_lv", o.top_lv},           {"top_z", o.top_z},             {"prior_mu", o.prior_mu},
                         {"prior_lv", o.prior_lv},       {"lower_mu", o.lower_mu},       {"lower_lv", o.lower_lv},
                         {"lower_z", o.lower_z},         {"probability", o.probability}, {"nll", o.nll},
                         {"kl_lower", o.kl_lower},       {"kl_top", o.kl_top},           {"loss", o.loss}};
    for (auto& [name, t] : out) {
      t = t.contiguous();
      for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "all ladder outputs finite");
    }
    return out;
  }
  /// Ancestral p(z1|z2) p(x|z1) from given top codes [n, 2] and lower noise [n, 4]; no image/encoder.
  nn::NamedTensors Prior(const nn::Tensor& top, const nn::Tensor& lower_noise) {
    nn::NoGradGuard guard;
    auto hp = nn::dense(parameters, "prior_hidden", top, 16, nn::Activation::Relu);
    auto pm = nn::dense(parameters, "prior_mu", hp, 4), pl = nn::dense(parameters, "prior_lv", hp, 4);
    auto z  = VaeReparameterize(pm, pl, lower_noise);
    return {{"top_z", top},
            {"lower_noise", lower_noise},
            {"prior_mu", pm.contiguous()},
            {"prior_lv", pl.contiguous()},
            {"lower_z", z.contiguous()},
            {"probability", nn::sigmoid(LadderDecoder(parameters, z)).contiguous()}};
  }
  /// Target-free lower-latent decoder.
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::sigmoid(LadderDecoder(parameters, codes)).contiguous();
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& noise) {
    return nn::grad(LadderVariationalAutoencoder(parameters, Flatten(images), noise).loss, parameters.parameters());
  }
  /// One Adam step; returns the pre-update negative ELBO.
  double Update(const nn::Tensor& images, const nn::Tensor& noise) {
    auto         loss  = LadderVariationalAutoencoder(parameters, Flatten(images), noise).loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value) && value >= 0, "finite negativeELBO");
    optimizer->minimize(loss);
    return value;
  }
  void Load(const nn::NamedTensors& arrays) { LoadArrays(parameters, arrays); }
};

/// Banks of `samples` x [n, 6] two-level standard normals (float64 + float32 cast).
inline VaeNoiseBank LadderNoise(std::int64_t n, std::int64_t samples, std::uint32_t seed) {
  return GaussianBank(n, 6, samples, seed);
}
inline nn::Tensor LadderNoiseAt(const VaeNoiseBank& bank, std::int64_t k) { return BetaVaeNoiseAt(bank, k); }

/// Monte Carlo {NLL, lower KL, top KL, negative ELBO} per image over every sample of `noise`.
inline std::vector<double> LadderScore(LadderVaeModel& model, const nn::Tensor& images, const VaeNoiseBank& noise) {
  double             nll = 0, lower = 0, top = 0;
  const std::int64_t n = images.size(0), samples = noise.epsilon.size(0);
  for (std::int64_t k = 0; k < samples; ++k) {
    auto       f = model.Evaluate(images, LadderNoiseAt(noise, k));
    const auto a = Find(f, "nll").to_vector<float>(), b = Find(f, "kl_lower").to_vector<float>(),
               c = Find(f, "kl_top").to_vector<float>();
    for (std::int64_t i = 0; i < n; ++i) {
      nll += a[i];
      lower += b[i];
      top += c[i];
    }
  }
  const double count = double(n * samples);
  return {nll / count, lower / count, top / count, (nll + lower + top) / count};
}

}  // namespace ae
