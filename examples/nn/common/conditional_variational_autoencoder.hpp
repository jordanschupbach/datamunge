// Port of tflow's common/nn/conditional_variational_autoencoder.h and
// common/ml/conditional_variational_autoencoder.h: a class-conditioned Gaussian VAE
// where the supplied one-hot class enters BOTH the posterior and the decoder.
#pragma once

#include "common/autoencoder.hpp"

namespace ae {

struct ConditionalVaeOutput : VaeOutput {
  nn::Tensor condition, encoder_input, decoder_input;
};

inline nn::Tensor ConditionalVaeDecodeLogits(nn::ParameterStore& p, const nn::Tensor& decoder_input) {
  return nn::dense(p, "output", nn::dense(p, "decoder", decoder_input, 64, nn::Activation::Relu), 784);
}

/// concat(image 784, onehot 10) -> 64 ReLU -> (mu, log variance) [2]; concat(z, onehot) -> 64 ReLU -> 784 logits.
inline ConditionalVaeOutput ConditionalVariationalAutoencoder(nn::ParameterStore& p, const nn::Tensor& image,
                                                              const nn::Tensor& labels, const nn::Tensor& epsilon) {
  ConditionalVaeOutput out;
  out.condition          = nn::one_hot(labels, 10);
  out.encoder_input      = nn::concat({image, out.condition}, 1);
  auto h                 = nn::dense(p, "encoder", out.encoder_input, 64, nn::Activation::Relu);
  out.mu                 = nn::dense(p, "mu", h, 2);
  out.log_variance       = nn::dense(p, "log_variance", h, 2);
  out.z                  = VaeReparameterize(out.mu, out.log_variance, epsilon);
  out.decoder_input      = nn::concat({out.z, out.condition}, 1);
  out.logits             = ConditionalVaeDecodeLogits(p, out.decoder_input);
  out.probability        = nn::sigmoid(out.logits);
  out.reconstruction_nll = VaePixelNll(out.logits, image);
  out.kl                 = VaeKl(out.mu, out.log_variance);
  out.loss               = nn::mean(out.reconstruction_nll + out.kl, {0});
  return out;
}

class ConditionalVariationalAutoencoderModel {
 public:
  nn::ParameterStore             parameters{2111};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit ConditionalVariationalAutoencoderModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      ConditionalVariationalAutoencoder(parameters, nn::zeros({1, 784}), nn::zeros({1}, nn::DType::Int32),
                                        nn::zeros({1, 2}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  static void ValidateLabels(const nn::Tensor& classes, std::int64_t n) {
    check::Require(classes.dtype() == nn::DType::Int32 && classes.shape() == nn::Shape({n}),
                   "one supplied class per example");
    for (int c : classes.contiguous().to_vector<int>())
      check::Require(c >= 0 && c < 10, "supplied digit condition must be in0..9");
  }
  ConditionalVaeOutput Forward(const nn::Tensor& images, const nn::Tensor& classes, const nn::Tensor& noise) {
    ValidateLabels(classes, images.size(0));
    return ConditionalVariationalAutoencoder(parameters, Flatten(images), classes, noise);
  }
  /// {mu, log_variance, z, probability, reconstruction_nll, kl, loss}, all checked finite.
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images, const nn::Tensor& classes, const nn::Tensor& noise) {
    nn::NoGradGuard         guard;
    auto                    n = Forward(images, classes, noise);
    std::vector<nn::Tensor> out{n.mu, n.log_variance, n.z, n.probability, n.reconstruction_nll, n.kl, n.loss};
    for (auto& t : out) {
      t = t.contiguous();
      for (float v : t.to_vector<float>())
        check::Require(std::isfinite(v), "ALL posterior/output/objective values finite");
    }
    return out;
  }
  /// Decoder with supplied class, WITHOUT image/noise/target.
  nn::Tensor Decode(const nn::Tensor& codes, const nn::Tensor& classes) {
    ValidateLabels(classes, codes.size(0));
    nn::NoGradGuard guard;
    return nn::sigmoid(ConditionalVaeDecodeLogits(parameters, nn::concat({codes, nn::one_hot(classes, 10)}, 1)))
        .contiguous();
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& classes, const nn::Tensor& noise) {
    return nn::grad(Forward(images, classes, noise).loss, parameters.parameters());
  }
  double Update(const nn::Tensor& images, const nn::Tensor& classes, const nn::Tensor& noise) {
    auto         loss  = Forward(images, classes, noise).loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value) && value >= 0, "finite negativeELBO");
    optimizer->minimize(loss);
    return value;
  }
  /// {one_hot, encoder_input, decoder_input, z}
  std::vector<nn::Tensor> Probe(const nn::Tensor& images, const nn::Tensor& classes, const nn::Tensor& noise) {
    nn::NoGradGuard guard;
    auto            n = Forward(images, classes, noise);
    return {n.condition.contiguous(), n.encoder_input.contiguous(), n.decoder_input.contiguous(), n.z.contiguous()};
  }
};

inline std::vector<double> ConditionalVaeScore(ConditionalVariationalAutoencoderModel& model, const nn::Tensor& images,
                                               const nn::Tensor& labels, const VaeNoiseBank& bank) {
  double             reconstruction = 0, kl = 0;
  const std::int64_t n = images.size(0), samples = bank.epsilon.size(0);
  for (std::int64_t k = 0; k < samples; ++k) {
    auto       out = model.Evaluate(images, labels, VaeNoiseAt(bank, k));
    const auto nll = out[4].to_vector<float>(), kls = out[5].to_vector<float>();
    for (std::int64_t b = 0; b < n; ++b) {
      reconstruction += nll[b];
      if (k == 0) kl += kls[b];
    }
  }
  reconstruction /= double(n * samples);
  kl /= double(n);
  return {reconstruction, kl, reconstruction + kl};
}

}  // namespace ae
