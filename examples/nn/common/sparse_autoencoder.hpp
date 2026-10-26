// Port of tflow's common/{nn,ml}/sparse_autoencoder.h: activation L1 and mean-unit Bernoulli
// KL penalties (explicit batch aggregation and clamp) on a 784->64 sigmoid->784 sigmoid
// autoencoder. Training steps (loss, nn::grad, optimizer) are written out in the example.
#pragma once

#include "common/autoencoder.hpp"

namespace ae {

struct SparsityPenalty {
  nn::Tensor l1, kl, unit_mean, clipped_mean, total;
};

/// l1 = mean |code|; unit_mean = batch mean per unit, clamped to [eps, 1 - eps];
/// kl = mean over units of KL(Bernoulli(rho) || Bernoulli(clipped mean));
/// total = l1_weight l1 + kl_weight kl. All constants are float32 values as in tflow.
inline SparsityPenalty ActivationSparsity(const nn::Tensor& code, float l1_weight, float kl_weight, float rho = .05f,
                                          float epsilon = 1e-5f) {
  check::Require(std::isfinite(l1_weight) && l1_weight >= 0 && std::isfinite(kl_weight) && kl_weight >= 0 && rho > 0 &&
                     rho < 1 && epsilon > 0 && epsilon < .5f,
                 "valid sparsity weights/target/clamp");
  SparsityPenalty out;
  out.l1           = nn::mean(nn::abs(code));
  out.unit_mean    = nn::mean(code, {0});
  out.clipped_mean = nn::clip(out.unit_mean, double(epsilon), double(1.f - epsilon));
  auto positive    = double(rho) * nn::log(double(rho) / out.clipped_mean);
  auto negative    = double(1.f - rho) * nn::log(double(1.f - rho) / (1.0 - out.clipped_mean));
  out.kl           = nn::mean(positive + negative, {0});
  out.total        = double(l1_weight) * out.l1 + double(kl_weight) * out.kl;
  return out;
}

struct SparseAeOutput {
  nn::Tensor      code, reconstruction, mse, loss;
  SparsityPenalty penalty;
};

inline SparseAeOutput SparseAutoencoder(nn::ParameterStore& p, const nn::Tensor& image, float l1_weight,
                                        float kl_weight) {
  SparseAeOutput out;
  out.code           = nn::dense(p, "encoder", image, 64, nn::Activation::Sigmoid);
  out.reconstruction = nn::dense(p, "decoder", out.code, 784, nn::Activation::Sigmoid);
  out.mse            = ReconstructionMse(out.reconstruction, image);
  out.penalty        = ActivationSparsity(out.code, l1_weight, kl_weight);
  out.loss           = out.mse + out.penalty.total;
  return out;
}

class SparseAutoencoderModel {
 public:
  nn::ParameterStore             parameters{1601};
  std::unique_ptr<nn::Optimizer> optimizer;
  float                          l1_weight, kl_weight;

  SparseAutoencoderModel(float l1, float kl, bool training = true) : l1_weight(l1), kl_weight(kl) {
    {
      nn::NoGradGuard guard;
      SparseAutoencoder(parameters, nn::zeros({1, 784}), l1_weight, kl_weight);
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  SparseAeOutput Net(const nn::Tensor& images) { return SparseAutoencoder(parameters, Flatten(images), l1_weight, kl_weight); }
  /// {code, reconstruction, loss, mse, l1, kl, unit_mean}
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            n = Net(images);
    return {n.code, n.reconstruction, n.loss, n.mse, n.penalty.l1, n.penalty.kl, n.penalty.unit_mean};
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::dense(parameters, "decoder", codes, 784, nn::Activation::Sigmoid);
  }
};

}  // namespace ae
