// Port of tflow's common/nn/wasserstein_autoencoder.h and
// common/ml/wasserstein_autoencoder.h: a compact WAE-MMD with a deterministic
// 784 -> 16 tanh -> 4 encoder, 4 -> 16 tanh -> 784 sigmoid decoder, trained on
// mean summed squared pixel cost + 10 unbiased IMQ MMD(codes, explicit N(0, I) sample).
#pragma once

#include "common/beta_variational_autoencoder.hpp"
#include "common/kernel_discrepancy.hpp"

namespace ae {

struct WaeOutput {
  nn::Tensor code, logits, probability, squared_cost, reconstruction_loss, loss;
  MmdOutput  mmd;
};

inline nn::Tensor WaeDecodeLogits(nn::ParameterStore& p, const nn::Tensor& code) {
  return nn::dense(p, "output", nn::dense(p, "decoder", code, 16, nn::Activation::Tanh), 784);
}

inline WaeOutput WassersteinAutoencoder(nn::ParameterStore& p, const nn::Tensor& image, const nn::Tensor& prior) {
  WaeOutput o;
  o.code                = nn::dense(p, "code", nn::dense(p, "encoder", image, 16, nn::Activation::Tanh), 4);
  o.logits              = WaeDecodeLogits(p, o.code);
  o.probability         = nn::sigmoid(o.logits);
  o.squared_cost        = nn::sum(nn::square(image - o.probability), {1});
  o.reconstruction_loss = nn::mean(o.squared_cost, {0});
  o.mmd                 = ImqMmd(o.code, prior);
  o.loss                = o.reconstruction_loss + 10.0 * o.mmd.value;
  return o;
}

class WassersteinAutoencoderModel {
 public:
  nn::ParameterStore             parameters{3701};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit WassersteinAutoencoderModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      WassersteinAutoencoder(parameters, nn::zeros({2, 784}), nn::zeros({2, 4}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  WaeOutput Forward(const nn::Tensor& images, const nn::Tensor& prior) {
    check::Require(images.size(0) >= 2 && prior.size(0) >= 2, "unbiased MMD needs at least2samples EACH");
    return WassersteinAutoencoder(parameters, Flatten(images), prior);
  }
  nn::NamedTensors Evaluate(const nn::Tensor& images, const nn::Tensor& prior) {
    nn::NoGradGuard  guard;
    auto             n = Forward(images, prior);
    nn::NamedTensors out{{"code", n.code},
                         {"logits", n.logits},
                         {"probability", n.probability},
                         {"squared_cost", n.squared_cost},
                         {"reconstruction_loss", n.reconstruction_loss},
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
      for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "all WAE fields finite");
    }
    return out;
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::sigmoid(WaeDecodeLogits(parameters, codes)).contiguous();
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& prior) {
    return nn::grad(Forward(images, prior).loss, parameters.parameters());
  }
  double Update(const nn::Tensor& images, const nn::Tensor& prior) {
    auto         loss  = Forward(images, prior).loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value), "finite WAE objective");
    optimizer->minimize(loss);
    return value;
  }
  void Load(const nn::NamedTensors& arrays) { LoadArrays(parameters, arrays); }
};

}  // namespace ae
