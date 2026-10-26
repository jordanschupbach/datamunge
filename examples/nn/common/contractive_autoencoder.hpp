// Port of tflow's common/ml/contractive_autoencoder.h + common/nn/contractive_autoencoder.h:
// a 784->32 sigmoid->784 sigmoid autoencoder with the exact analytic encoder
// Jacobian Frobenius penalty (differentiated through BOTH the activation slopes
// and the encoder weights) and an independent autodiff Jacobian. Training steps
// (Forward loss, nn::grad, optimizer) are written out in the example.
#pragma once

#include "common/autoencoder.hpp"

namespace ae {

struct ContractivePenalty {
  nn::Tensor per_example, mean;  // [n], scalar
};

/// sum_j (h_j (1 - h_j))^2 sum_i W_ij^2 per example, and its batch mean.
inline ContractivePenalty SigmoidEncoderContraction(const nn::Tensor& code, const nn::Tensor& encoder_weight) {
  auto slope               = code * (1.0 - code);
  auto squared_column_norm = nn::sum(nn::square(encoder_weight), {0});
  auto per                 = nn::sum(nn::square(slope) * squared_column_norm, {1});
  return {per, nn::mean(per, {0})};
}

struct ContractiveOutput {
  nn::Tensor code, reconstruction, loss, mse;
  ContractivePenalty penalty;
};

class ContractiveAutoencoderModel {
 public:
  nn::ParameterStore             parameters{1753};
  std::unique_ptr<nn::Optimizer> optimizer;
  float                          weight;

  explicit ContractiveAutoencoderModel(float weight_, bool training = true) : weight(weight_) {
    check::Require(std::isfinite(weight) && weight >= 0, "finite nonnegative Jacobian weight");
    {
      nn::NoGradGuard guard;
      Forward(nn::zeros({1, 784}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  nn::Tensor Decoder(const nn::Tensor& code) {
    return nn::dense(parameters, "decoder", code, 784, nn::Activation::Sigmoid);
  }
  ContractiveOutput Forward(const nn::Tensor& image) {
    ContractiveOutput out;
    out.code           = nn::dense(parameters, "encoder", image, 32, nn::Activation::Sigmoid);
    out.reconstruction = Decoder(out.code);
    out.mse            = ReconstructionMse(out.reconstruction, image);
    out.penalty        = SigmoidEncoderContraction(out.code, parameters.at("encoder/W"));
    out.loss           = out.mse + double(weight) * out.penalty.mean;
    return out;
  }
  /// {code, reconstruction, loss, mse, penalty mean, penalty per example}
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            n = Forward(Flatten(images));
    return {n.code, n.reconstruction, n.loss, n.mse, n.penalty.mean, n.penalty.per_example};
  }
  /// [n, 32, 784] ordinary image Jacobian of the code, one reverse pass per code unit
  /// (independent of the analytic norm formula).
  nn::Tensor Jacobian(const nn::Tensor& images) {
    auto x    = Flatten(images).contiguous().detach().requires_grad_();
    auto code = nn::dense(parameters, "encoder", x, 32, nn::Activation::Sigmoid);
    std::vector<nn::Tensor> rows;
    nn::GradOptions         keep;
    keep.retain_graph = true;
    for (int j = 0; j < 32; ++j) rows.push_back(nn::grad(nn::sum(nn::slice_axis(code, 1, j, j + 1)), {x}, keep)[0]);
    return nn::stack(rows, 1).contiguous();
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return Decoder(codes);
  }
};

}  // namespace ae
