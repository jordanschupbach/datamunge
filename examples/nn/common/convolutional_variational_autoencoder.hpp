// Port of tflow's common/nn/convolutional_variational_autoencoder.h and
// common/ml/convolutional_variational_autoencoder.h: pooled spatial encoder,
// two-dimensional Gaussian posterior, nearest-repeat convolutional decoder, no skips.
#pragma once

#include "common/autoencoder.hpp"

namespace ae {

struct ConvVaeOutput : VaeOutput {
  nn::Tensor conv1, pool1, conv2, pool2, decoder_base, decoder1, spatial_logits;
};

inline nn::ConvLayerOptions ConvVaeRelu() {
  nn::ConvLayerOptions relu;
  relu.init       = nn::init::he_uniform();
  relu.activation = nn::Activation::Relu;
  return relu;
}

/// 2 -> 196 ReLU -> 7x7x4 -> repeat2 -> Conv3 4->8 ReLU -> repeat2 -> Conv3 8->1 logits.
inline void ConvVaeDecode(nn::ParameterStore& p, const nn::Tensor& z, ConvVaeOutput& out) {
  out.decoder_base =
      nn::reshape(nn::dense(p, "decoder_projection", z, 196, nn::Activation::Relu), {z.size(0), 7, 7, 4});
  out.decoder1       = nn::conv2d(p, "decoder1", nn::upsample_nearest(out.decoder_base, 2), 8, 3, ConvVaeRelu());
  out.spatial_logits = nn::conv2d(p, "decoder2", nn::upsample_nearest(out.decoder1, 2), 1, 3, nn::ConvLayerOptions{});
  out.logits         = nn::reshape(out.spatial_logits, {z.size(0), 784});
  out.probability    = nn::sigmoid(out.logits);
}

/// image [n, 28, 28, 1]: Conv3 1->8 ReLU, avg2, Conv3 8->4 ReLU, avg2 (7x7x4), mu/log variance 196->2.
inline ConvVaeOutput ConvolutionalVariationalAutoencoder(nn::ParameterStore& p, const nn::Tensor& image,
                                                         const nn::Tensor& epsilon) {
  ConvVaeOutput out;
  out.conv1              = nn::conv2d(p, "encoder1", image, 8, 3, ConvVaeRelu());
  out.pool1              = nn::avg_pool2d(out.conv1, 2, 2);
  out.conv2              = nn::conv2d(p, "encoder2", out.pool1, 4, 3, ConvVaeRelu());
  out.pool2              = nn::avg_pool2d(out.conv2, 2, 2);
  auto encoded           = nn::reshape(out.pool2, {image.size(0), 196});
  out.mu                 = nn::dense(p, "mu", encoded, 2);
  out.log_variance       = nn::dense(p, "log_variance", encoded, 2);
  out.z                  = VaeReparameterize(out.mu, out.log_variance, epsilon);
  ConvVaeDecode(p, out.z, out);
  out.reconstruction_nll = VaePixelNll(out.logits, nn::reshape(image, {image.size(0), 784}));
  out.kl                 = VaeKl(out.mu, out.log_variance);
  out.loss               = nn::mean(out.reconstruction_nll + out.kl, {0});
  return out;
}

class ConvolutionalVariationalAutoencoderModel {
 public:
  nn::ParameterStore             parameters{2011};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit ConvolutionalVariationalAutoencoderModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      ConvolutionalVariationalAutoencoder(parameters, nn::zeros({1, 28, 28, 1}), nn::zeros({1, 2}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  /// {mu, log_variance, z, probability, reconstruction_nll, kl, loss}, all checked finite.
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images, const nn::Tensor& noise) {
    nn::NoGradGuard         guard;
    auto                    n = ConvolutionalVariationalAutoencoder(parameters, images, noise);
    std::vector<nn::Tensor> out{n.mu, n.log_variance, n.z, n.probability, n.reconstruction_nll, n.kl, n.loss};
    for (auto& t : out) {
      t = t.contiguous();
      for (float v : t.to_vector<float>())
        check::Require(std::isfinite(v), "ALL posterior/output/objective values finite");
    }
    return out;
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    ConvVaeOutput   out;
    ConvVaeDecode(parameters, codes, out);
    return out.probability.contiguous();
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& noise) {
    return nn::grad(ConvolutionalVariationalAutoencoder(parameters, images, noise).loss, parameters.parameters());
  }
  double Update(const nn::Tensor& images, const nn::Tensor& noise) {
    auto         loss  = ConvolutionalVariationalAutoencoder(parameters, images, noise).loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value) && value >= 0, "finite negativeELBO");
    optimizer->minimize(loss);
    return value;
  }
  /// The seven convolution/pool/repeat-decoder stages.
  std::vector<nn::Tensor> Stages(const nn::Tensor& images, const nn::Tensor& noise) {
    nn::NoGradGuard guard;
    auto            n = ConvolutionalVariationalAutoencoder(parameters, images, noise);
    return {n.conv1.contiguous(),        n.pool1.contiguous(),    n.conv2.contiguous(),
            n.pool2.contiguous(),        n.decoder_base.contiguous(), n.decoder1.contiguous(),
            n.spatial_logits.contiguous()};
  }
};

inline std::vector<double> ConvolutionalVaeScore(ConvolutionalVariationalAutoencoderModel& model,
                                                 const nn::Tensor& images, const VaeNoiseBank& bank) {
  double             reconstruction = 0, kl = 0;
  const std::int64_t n = images.size(0), samples = bank.epsilon.size(0);
  for (std::int64_t k = 0; k < samples; ++k) {
    auto       out = model.Evaluate(images, VaeNoiseAt(bank, k));
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
