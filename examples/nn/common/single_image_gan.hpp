// Port of tflow's common/nn/single_image_gan.h and common/ml/single_image_gan.h:
// one SinGAN scale. Fully convolutional residual generator G(previous, noise,
// amplitude) = previous + tanh(conv(leaky(conv(leaky(conv(previous + amplitude
// noise)))))) and a local 3-layer patch discriminator (fan-in uniform SRConv
// kernels, width 4). LSGAN objectives; G also minimises 10 x the MSE of the
// fixed-noise reconstruction. Groups: 0 = G (seed 7307), 1 = D (7309);
// separate Adam (.0005, beta1 .5).
#pragma once

#include "common/adversarial_super_resolution.hpp"

namespace gan {

inline Tensor SingleImageGenerate(nn::ParameterStore& p, const Tensor& previous, const Tensor& noise, const Tensor& amplitude) {
  const Tensor input = previous + noise * amplitude;
  Tensor       h     = nn::leaky_relu(srgan::SRConv(p, "head", input, 3, 4, 3), .2);
  h                  = nn::leaky_relu(srgan::SRConv(p, "body", h, 4, 4, 3), .2);
  return previous + nn::tanh(srgan::SRConv(p, "output", h, 4, 3, 3));
}
inline Tensor SingleImageDiscriminate(nn::ParameterStore& p, const Tensor& image) {
  Tensor h = nn::leaky_relu(srgan::SRConv(p, "head", image, 3, 4, 3), .2);
  h        = nn::leaky_relu(srgan::SRConv(p, "body", h, 4, 4, 3), .2);
  return srgan::SRConv(p, "output", h, 4, 1, 3);
}

class SingleImageGAN : public GroupedGan {
 public:
  explicit SingleImageGAN(bool training = true, std::vector<std::uint64_t> seeds = {7307, 7309}) : GroupedGan(seeds) {
    {
      nn::NoGradGuard guard;
      const Tensor    zero = nn::zeros({1, 4, 4, 3});
      SingleImageDiscriminate(Group(1), SingleImageGenerate(Group(0), zero, zero, Scalar(1)));
    }
    if (training) MakeAdam(.0005, .5);
  }
  static Tensor Scalar(float x) { return nn::from_vector(std::vector<float>{x}, {}); }

  /// Batch keys: image, previous, noise, rec_previous, rec_noise, amplitude.
  Arrays Forward(const Arrays& b) {
    const Tensor amplitude      = b.at("amplitude");
    const Tensor fake           = SingleImageGenerate(Group(0), b.at("previous"), b.at("noise"), amplitude);
    const Tensor reconstruction = SingleImageGenerate(Group(0), b.at("rec_previous"), b.at("rec_noise"), amplitude);
    const Tensor real_score = SingleImageDiscriminate(Group(1), b.at("image")), fake_score = SingleImageDiscriminate(Group(1), fake);
    const Tensor adversarial_loss    = nn::mean(nn::square(fake_score - 1.0));
    const Tensor reconstruction_loss = nn::mean(nn::square(reconstruction - b.at("image")));
    return {{"fake", fake},
            {"reconstruction", reconstruction},
            {"real_score", real_score},
            {"fake_score", fake_score},
            {"d_loss", .5 * (nn::mean(nn::square(real_score - 1.0)) + nn::mean(nn::square(fake_score)))},
            {"g_loss", adversarial_loss + 10.0 * reconstruction_loss},
            {"adversarial_loss", adversarial_loss},
            {"reconstruction_loss", reconstruction_loss}};
  }
  Arrays Evaluate(const Arrays& b) {
    nn::NoGradGuard guard;
    Arrays          out = Forward(b);
    RequireFinite(out, "finite all scale outputs");
    return out;
  }
  /// Noise + previous ONLY fully convolutional generation (any H x W).
  Tensor Generate(const Tensor& previous, const Tensor& z, float amplitude) {
    check::Require(previous.shape() == z.shape() && previous.dim() == 4 && previous.size(3) == 3,
                   "matching NHWC previous and noise");
    nn::NoGradGuard guard;
    return SingleImageGenerate(Group(0), previous, z, Scalar(amplitude));
  }
  Arrays Gradient(int group, const Arrays& b) { return GradientOf(group, Forward(b).at(group == 0 ? "g_loss" : "d_loss")); }
  void   Update(int group, const Arrays& b) { Apply(group, Gradient(group, b)); }
};

}  // namespace gan
