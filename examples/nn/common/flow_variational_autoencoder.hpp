// Port of tflow's common/nn/flow_variational_autoencoder.h and
// common/ml/flow_variational_autoencoder.h: a 4-D VAE whose diagonal-Gaussian base
// posterior is pushed through two alternating 2+2 affine couplings conditioned on the
// image encoder context (analytic inverses and log-Jacobians), trained with the sampled
// change-of-variables objective NLL + log q_K(z|x) - log p(z).
#pragma once

#include "common/beta_variational_autoencoder.hpp"

namespace ae {

struct AffinePosteriorFlowOutput {
  nn::Tensor value, log_determinant, scale, shift;
};

/// One conditional affine coupling on z [n, 4]: the conditioning half a (first two
/// coordinates, or the last two with `swap`) and the context feed "<name>/conditioner"
/// (2+context -> 16 ReLU) and "<name>/heads" (16 -> 2 scale + 2 shift); scale = .5 tanh(raw).
/// Forward b' = b exp(scale) + shift (logdet = sum scale); inverse b = (b' - shift) exp(-scale).
inline AffinePosteriorFlowOutput ConditionalAffinePosteriorFlow(nn::ParameterStore& p, const std::string& name,
                                                                const nn::Tensor& z, const nn::Tensor& context,
                                                                bool swap = false, bool inverse = false) {
  auto first = nn::slice_axis(z, 1, 0, 2), second = nn::slice_axis(z, 1, 2, 4);
  auto a = swap ? second : first, b = swap ? first : second;
  auto h     = nn::dense(p, name + "/conditioner", nn::concat({a, context}, 1), 16, nn::Activation::Relu);
  auto heads = nn::dense(p, name + "/heads", h, 4);
  AffinePosteriorFlowOutput o;
  o.scale          = 0.5 * nn::tanh(nn::slice_axis(heads, 1, 0, 2));
  o.shift          = nn::slice_axis(heads, 1, 2, 4);
  auto transformed = inverse ? (b - o.shift) * nn::exp(-o.scale) : b * nn::exp(o.scale) + o.shift;
  o.value          = swap ? nn::concat({transformed, a}, 1) : nn::concat({a, transformed}, 1);
  auto sum         = nn::sum(o.scale, {1});
  o.log_determinant = inverse ? -sum : sum;
  return o;
}

inline nn::Tensor FlowVaeDecoder(nn::ParameterStore& p, const nn::Tensor& code) {
  return nn::dense(p, "output", nn::dense(p, "decoder", code, 32, nn::Activation::Relu), 784);
}

/// float(log(2 pi)), the constant tflow adds.
inline double FlowLog2Pi() { return double(float(std::log(2 * 3.14159265358979323846))); }

struct FlowVaeOutput {
  nn::Tensor context, mu, log_variance, z0, z1, z, logdet0, logdet1, logdet, logq0, logq, logp, logratio, probability,
      nll, loss, inverse_z0, inverse_logdet;
  /// Named fields in tflow's order.
  nn::NamedTensors Named() const {
    return {{"context", context}, {"mu", mu},       {"log_variance", log_variance},
            {"z0", z0},           {"z1", z1},       {"z", z},
            {"logdet0", logdet0}, {"logdet1", logdet1}, {"logdet", logdet},
            {"logq0", logq0},     {"logq", logq},   {"logp", logp},
            {"logratio", logratio}, {"probability", probability}, {"nll", nll},
            {"loss", loss},       {"inverse_z0", inverse_z0}, {"inverse_logdet", inverse_logdet}};
  }
};

inline FlowVaeOutput FlowVariationalAutoencoder(nn::ParameterStore& p, const nn::Tensor& image,
                                                const nn::Tensor& epsilon) {
  FlowVaeOutput o;
  o.context      = nn::dense(p, "encoder", image, 32, nn::Activation::Relu);
  o.mu           = nn::dense(p, "mu", o.context, 4);
  o.log_variance = nn::dense(p, "log_variance", o.context, 4);
  o.z0           = VaeReparameterize(o.mu, o.log_variance, epsilon);
  auto first = ConditionalAffinePosteriorFlow(p, "flow0", o.z0, o.context),
       second = ConditionalAffinePosteriorFlow(p, "flow1", first.value, o.context, true);
  o.z1      = first.value;
  o.z       = second.value;
  o.logdet0 = first.log_determinant;
  o.logdet1 = second.log_determinant;
  o.logdet  = o.logdet0 + o.logdet1;
  const double log2pi = FlowLog2Pi();
  o.logq0 = -0.5 * nn::sum(nn::square(o.z0 - o.mu) * nn::exp(-o.log_variance) + o.log_variance + log2pi, {1});
  o.logq  = o.logq0 - o.logdet;
  o.logp  = -0.5 * nn::sum(nn::square(o.z) + log2pi, {1});
  o.logratio = o.logq - o.logp;
  auto logits   = FlowVaeDecoder(p, o.z);
  o.probability = nn::sigmoid(logits);
  o.nll         = VaePixelNll(logits, image);
  o.loss        = nn::mean(o.nll + o.logratio, {0});
  auto back1 = ConditionalAffinePosteriorFlow(p, "flow1", o.z, o.context, true, true),
       back0 = ConditionalAffinePosteriorFlow(p, "flow0", back1.value, o.context, false, true);
  o.inverse_z0     = back0.value;
  o.inverse_logdet = back1.log_determinant + back0.log_determinant;
  return o;
}

class FlowVariationalModel {
 public:
  nn::ParameterStore             parameters{3001};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit FlowVariationalModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      FlowVariationalAutoencoder(parameters, nn::zeros({1, 784}), nn::zeros({1, 4}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  /// All 18 fields, contiguous and checked finite.
  nn::NamedTensors Evaluate(const nn::Tensor& images, const nn::Tensor& noise) {
    nn::NoGradGuard guard;
    auto            out = FlowVariationalAutoencoder(parameters, Flatten(images), noise).Named();
    for (auto& [name, t] : out) {
      t = t.contiguous();
      for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "all flow fields finite");
    }
    return out;
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::sigmoid(FlowVaeDecoder(parameters, codes)).contiguous();
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& noise) {
    return nn::grad(FlowVariationalAutoencoder(parameters, Flatten(images), noise).loss, parameters.parameters());
  }
  double Update(const nn::Tensor& images, const nn::Tensor& noise) {
    auto         loss  = FlowVariationalAutoencoder(parameters, Flatten(images), noise).loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value), "finite negativeELBO");
    optimizer->minimize(loss);
    return value;
  }
};

/// Monte Carlo {NLL, log q - log p, forward logdet, negative ELBO} per image over every bank sample.
inline std::vector<double> FlowVaeScore(FlowVariationalModel& model, const nn::Tensor& images,
                                        const VaeNoiseBank& noise) {
  double             nll = 0, ratio = 0, det = 0;
  const std::int64_t n = images.size(0), samples = noise.epsilon.size(0);
  for (std::int64_t k = 0; k < samples; ++k) {
    auto       f = model.Evaluate(images, BetaVaeNoiseAt(noise, k));
    const auto a = Find(f, "nll").to_vector<float>(), r = Find(f, "logratio").to_vector<float>(),
               d = Find(f, "logdet").to_vector<float>();
    for (std::int64_t b = 0; b < n; ++b) {
      nll += a[b];
      ratio += r[b];
      det += d[b];
    }
  }
  const double count = double(n * samples);
  return {nll / count, ratio / count, det / count, (nll + ratio) / count};
}

}  // namespace ae
