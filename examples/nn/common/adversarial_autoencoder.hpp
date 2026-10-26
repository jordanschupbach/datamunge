// Port of tflow's common/nn/adversarial_autoencoder.h and common/ml/adversarial_autoencoder.h:
// a compact deterministic adversarial autoencoder (Makhzani et al.) whose aggregate 4-D code is
// matched to N(0, I) by a latent discriminator. Three parameter groups (encoder seed 3501,
// decoder 3511, latent discriminator 3527) and FOUR independent Adam states: E-reconstruction,
// G-reconstruction, D, E-adversarial. Phase 0 applies the encoder and decoder reconstruction
// gradients computed from the SAME snapshot (tflow's gradient barrier), then phase 1 updates D,
// phase 2 the encoder against D.
#pragma once

#include "common/beta_variational_autoencoder.hpp"

namespace ae {

inline nn::Tensor AaeDecode(nn::ParameterStore& decoder, const nn::Tensor& code) {
  return nn::dense(decoder, "output", nn::dense(decoder, "hidden", code, 16, nn::Activation::Tanh), 784);
}
inline nn::Tensor AaeDiscriminate(nn::ParameterStore& discriminator, const nn::Tensor& code) {
  return nn::dense(discriminator, "logit", nn::dense(discriminator, "hidden", code, 16, nn::Activation::Tanh), 1);
}

struct AaeOutput {
  nn::Tensor code, logits, probability, prior_logit, encoded_logit, nll, reconstruction_loss, discriminator_loss,
      adversarial_loss;
};

inline AaeOutput AdversarialAutoencoder(nn::ParameterStore& e, nn::ParameterStore& g, nn::ParameterStore& d,
                                        const nn::Tensor& x, const nn::Tensor& prior) {
  AaeOutput o;
  o.code                = nn::dense(e, "code", nn::dense(e, "hidden", x, 16, nn::Activation::Tanh), 4);
  o.logits              = AaeDecode(g, o.code);
  o.probability         = nn::sigmoid(o.logits);
  o.nll                 = VaePixelNll(o.logits, x);
  o.reconstruction_loss = nn::mean(o.nll, {0});
  o.prior_logit         = AaeDiscriminate(d, prior);
  o.encoded_logit       = AaeDiscriminate(d, o.code);
  o.discriminator_loss =
      nn::mean(nn::softplus(nn::neg(o.prior_logit)), {0, 1}) + nn::mean(nn::softplus(o.encoded_logit), {0, 1});
  o.adversarial_loss = nn::mean(nn::softplus(nn::neg(o.encoded_logit)), {0, 1});
  return o;
}

class AdversarialAutoencoderModel {
 public:
  nn::ParameterStore                          encoder{3501}, decoder{3511}, discriminator{3527};
  std::vector<std::unique_ptr<nn::Optimizer>> optimizers;  // E-recon, G-recon, D, E-adversarial

  explicit AdversarialAutoencoderModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      AdversarialAutoencoder(encoder, decoder, discriminator, nn::zeros({1, 784}), nn::zeros({1, 4}));
    }
    if (training)
      for (auto* group : Groups()) optimizers.push_back(MakeAdam(*group));
  }
  /// The optimizer groups in tflow's order {E, G, D, E}.
  std::vector<nn::ParameterStore*> Groups() { return {&encoder, &decoder, &discriminator, &encoder}; }
  std::vector<nn::ParameterStore*> Stores() { return {&encoder, &decoder, &discriminator}; }
  std::int64_t                     Count() { return encoder.count() + decoder.count() + discriminator.count(); }
  std::size_t                      Tensors() { return encoder.size() + decoder.size() + discriminator.size(); }

  AaeOutput Forward(const nn::Tensor& x, const nn::Tensor& p) {
    return AdversarialAutoencoder(encoder, decoder, discriminator, Flatten(x), p);
  }
  /// Named fields of tflow's Evaluate, all checked finite.
  nn::NamedTensors Evaluate(const nn::Tensor& x, const nn::Tensor& p) {
    nn::NoGradGuard  guard;
    auto             o = Forward(x, p);
    nn::NamedTensors out{{"code", o.code},
                         {"logits", o.logits},
                         {"probability", o.probability},
                         {"prior_logit", o.prior_logit},
                         {"encoded_logit", o.encoded_logit},
                         {"nll", o.nll},
                         {"reconstruction_loss", o.reconstruction_loss},
                         {"discriminator_loss", o.discriminator_loss},
                         {"adversarial_loss", o.adversarial_loss}};
    for (auto& [name, t] : out) {
      t = t.contiguous();
      for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "all AAE values finite");
    }
    return out;
  }
  static nn::Tensor Objective(const AaeOutput& o, int group) {
    return group < 2 ? o.reconstruction_loss : group == 2 ? o.discriminator_loss : o.adversarial_loss;
  }
  std::vector<nn::Tensor> GroupGradient(int group, const nn::Tensor& x, const nn::Tensor& p) {
    return nn::grad(Objective(Forward(x, p), group), Groups()[group]->parameters());
  }
  /// tflow's Gradient: named "<k>/<name>" with k = 0 for both encoder groups.
  nn::NamedTensors Gradient(int group, const nn::Tensor& x, const nn::Tensor& p) {
    auto             g = GroupGradient(group, x, p);
    nn::NamedTensors out;
    const auto&      names = Groups()[group]->names();
    for (std::size_t i = 0; i < g.size(); ++i)
      out.emplace_back(std::to_string(group == 3 ? 0 : group) + "/" + names[i], g[i]);
    return out;
  }
  /// Phase 0: joint reconstruction step of E and G from ONE gradient snapshot; 1: D; 2: E adversarial.
  void Update(int phase, const nn::Tensor& x, const nn::Tensor& p) {
    if (phase == 0) {
      auto o  = Forward(x, p);
      auto ge = encoder.parameters(), gd = decoder.parameters();
      auto vars = ge;
      vars.insert(vars.end(), gd.begin(), gd.end());
      auto g = nn::grad(o.reconstruction_loss, vars);
      optimizers[0]->step(std::vector<nn::Tensor>(g.begin(), g.begin() + std::ptrdiff_t(ge.size())));
      optimizers[1]->step(std::vector<nn::Tensor>(g.begin() + std::ptrdiff_t(ge.size()), g.end()));
    } else {
      const int k = phase == 1 ? 2 : 3;
      optimizers[k]->step(GroupGradient(k, x, p));
    }
  }
  /// All arrays named "<k>/<name>" (k = 0 encoder, 1 decoder, 2 discriminator).
  nn::NamedTensors Arrays() {
    nn::NamedTensors out;
    auto             stores = Stores();
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
    check::Require(arrays.size() == Tensors(), "all AAE tensors supplied");
    auto stores = Stores();
    for (int k = 0; k < 3; ++k)
      for (const auto& name : stores[k]->names()) {
        auto value = Find(arrays, std::to_string(k) + "/" + name);
        check::Require(value.shape() == stores[k]->at(name).shape(), "matching AAE shapes");
        stores[k]->set(name, value.contiguous().clone());
      }
  }
  /// AAE target-free code-only decoder.
  nn::Tensor Decode(const nn::Tensor& z) {
    nn::NoGradGuard guard;
    return nn::sigmoid(AaeDecode(decoder, z)).contiguous();
  }
};

}  // namespace ae
