// Port of tflow's common/nn/adversarial_super_resolution.h and
// common/ml/adversarial_super_resolution.h (plus the SRConv / SRPixelShuffle
// pieces of common/nn/super_resolution.h and VggImageNetInput of
// common/nn/perceptual.h it uses): compact paired x2 SRGAN (BN/PReLU residual
// blocks, CRD pixel shuffle) and ESRGAN (residual-in-residual dense blocks, no
// BN, relativistic average GAN), both predicting a residual over the bicubic
// image, with a frozen pretrained VGG19 block1_conv1 content loss.
// Groups: 0 = generator (seed 7001), 1 = discriminator (7013), 2 = SRGAN BN
// moving statistics; Adam (.0001, beta1 .9) for 0 and 1.
#pragma once

#include "common/gan.hpp"
#include "common/sr_data.hpp"

namespace srgan {

namespace nn = datamunge::nn;
using nn::Tensor;
using gan::Arrays;

/// tflow SRConv: fan-in uniform U(+-1/sqrt(k k in)) kernel (Torch Conv2d default), zero bias, SAME.
inline Tensor SRConv(nn::ParameterStore& p, const std::string& name, const Tensor& x, std::int64_t in, std::int64_t out,
                     std::int64_t kernel, std::int64_t stride = 1, bool bias = true) {
  p.get(name + "/W", {kernel, kernel, in, out}, nn::init::uniform(1 / std::sqrt(float(kernel * kernel * in))));
  nn::ConvLayerOptions o;
  o.stride = stride;
  o.bias   = bias;
  return nn::conv2d(p, name, x, out, kernel, o);
}

/// tflow SRPixelShuffle (PyTorch/paper CRD order: input channel c r r + dy r + dx).
inline Tensor SRPixelShuffle(const Tensor& x, std::int64_t channels, std::int64_t scale) {
  const std::int64_t b = x.size(0), h = x.size(1), w = x.size(2);
  return nn::reshape(nn::transpose(nn::reshape(x, {b, h, w, channels, scale, scale}), {0, 1, 4, 2, 5, 3}),
                     {b, h * scale, w * scale, channels});
}

/// tflow VggImageNetInput: RGB [0, 1] -> BGR [0, 255] minus the ImageNet means.
inline Tensor VggImageNetInput(const Tensor& rgb) {
  const Tensor bgr = nn::gather(rgb, nn::tensor({2, 1, 0}, {3}, nn::DType::Int64), 3);
  return bgr * 255.0 - nn::tensor({103.939f, 116.779f, 123.68f}, {3});
}

struct AdversarialSRPath {
  Tensor image, residual, packed, shuffled;
  Arrays moments;  // "<bn>/mean", "<bn>/unbiased_var" (training mode only)
};

inline Tensor SRGanPrelu(nn::ParameterStore& p, const std::string& name, const Tensor& x) {
  const Tensor alpha = p.get(name + "/alpha", {4}, nn::init::constant(.25));
  return nn::where(nn::greater_equal(x, 0.0), x, x * alpha);
}

/// tflow SRGanBatchNorm: batch moments (eps 1e-5) when training, moving statistics otherwise.
inline Tensor SRGanBatchNorm(nn::ParameterStore& p, nn::ParameterStore& stats, AdversarialSRPath& path,
                             const std::string& name, const Tensor& x, bool training) {
  const Tensor gamma = p.get(name + "/gamma", {4}, nn::init::ones()), beta = p.get(name + "/beta", {4}, nn::init::zeros());
  Tensor       mean = stats.get(name + "/moving_mean", {4}, nn::init::zeros());
  Tensor       var  = stats.get(name + "/moving_var", {4}, nn::init::ones());
  if (training) {
    mean               = nn::mean(x, {0, 1, 2});
    var                = nn::mean(nn::square(x - mean), {0, 1, 2});
    const double count = double(float(x.numel()) / 4.f);
    path.moments[name + "/mean"]         = mean.detach();
    path.moments[name + "/unbiased_var"] = (var * double(float(count) / float(count - 1))).detach();
  }
  return (x - mean) / nn::sqrt(var + double(1e-5f)) * gamma + beta;
}

inline Tensor SRGanResidual(nn::ParameterStore& p, nn::ParameterStore& stats, AdversarialSRPath& path,
                            const std::string& name, const Tensor& input, bool training) {
  Tensor h = SRConv(p, name + "/conv1", input, 4, 4, 3, 1, false);
  h        = SRGanBatchNorm(p, stats, path, name + "/bn1", h, training);
  h        = SRGanPrelu(p, name + "/prelu", h);
  h        = SRConv(p, name + "/conv2", h, 4, 4, 3, 1, false);
  return input + SRGanBatchNorm(p, stats, path, name + "/bn2", h, training);
}

/// ESRGAN dense block: four growth-2 leaky convs on the running concatenation, conv5 -> 4, x + .2 tail.
inline Tensor ESRGanDenseBlock(nn::ParameterStore& p, const std::string& name, const Tensor& input) {
  std::vector<Tensor> features{input};
  for (int i = 0; i < 4; ++i) {
    const Tensor joined = features.size() == 1 ? input : nn::concat(features, 3);
    features.push_back(nn::leaky_relu(SRConv(p, name + "/conv" + std::to_string(i + 1), joined, 4 + 2 * i, 2, 3), .2));
  }
  return input + .2 * SRConv(p, name + "/conv5", nn::concat(features, 3), 12, 4, 3);
}

inline AdversarialSRPath AdversarialSRGenerate(nn::ParameterStore& p, nn::ParameterStore& stats, const Tensor& lr,
                                               const Tensor& bicubic, bool esrgan, bool training) {
  AdversarialSRPath path;
  Tensor            head = SRConv(p, "head", lr, 3, 4, 3), h;
  if (esrgan) {
    h = head;
    for (int i = 0; i < 3; ++i) h = ESRGanDenseBlock(p, "rrdb/dense" + std::to_string(i), h);
    h = head + SRConv(p, "body_tail", head + .2 * h, 4, 4, 3);
    h = nn::resize_nearest(h, {32, 32});
    h = nn::leaky_relu(SRConv(p, "up", h, 4, 4, 3), .2);
    h = nn::leaky_relu(SRConv(p, "hr", h, 4, 4, 3), .2);
    path.packed = path.shuffled = h;
  } else {
    head = SRGanPrelu(p, "head_prelu", head);
    h    = head;
    for (int i = 0; i < 2; ++i) h = SRGanResidual(p, stats, path, "residual" + std::to_string(i), h, training);
    h = head + SRGanBatchNorm(p, stats, path, "body_bn", SRConv(p, "body_tail", h, 4, 4, 3, 1, false), training);
    path.packed   = SRConv(p, "up", h, 4, 16, 3);
    path.shuffled = SRPixelShuffle(path.packed, 4, 2);
    h             = SRGanPrelu(p, "up_prelu", path.shuffled);
  }
  p.get("output/W", {3, 3, 4, 3}, nn::init::normal(.001));
  path.residual = SRConv(p, "output", h, 4, 3, 3);
  path.image    = bicubic + path.residual;
  return path;
}

/// head 3->4 leaky, 4x4/2 -> 8 leaky, 4x4/2 -> 8 leaky, global mean, raw score (no bias when relative).
inline Tensor AdversarialSRDiscriminate(nn::ParameterStore& p, const Tensor& image, bool relative = false) {
  Tensor       h      = nn::leaky_relu(SRConv(p, "head", image, 3, 4, 3), .2);
  h                   = nn::leaky_relu(SRConv(p, "down16", h, 4, 8, 4, 2), .2);
  h                   = nn::leaky_relu(SRConv(p, "down8", h, 8, 8, 4, 2), .2);
  const Tensor pooled = nn::mean(h, {1, 2});
  if (relative) return nn::matmul(pooled, p.get("score/W", {8, 1}, nn::init::uniform(std::sqrt(6.f / 9))));
  return nn::dense(p, "score", pooled, 1);
}

struct RelativisticSRLoss {
  Tensor discriminator, generator, real_relative, fake_relative;
};
inline RelativisticSRLoss RelativisticAverageSR(const Tensor& real, const Tensor& fake) {
  const Tensor rr = real - nn::mean(fake), rf = fake - nn::mean(real);
  return {.5 * (nn::mean(nn::softplus(-rr)) + nn::mean(nn::softplus(rf))),
          .5 * (nn::mean(nn::softplus(rr)) + nn::mean(nn::softplus(-rf))), rr, rf};
}

/// tflow ml::AdversarialSRModel. Capture(group) returns the gradients (and for
/// G the pre-update BN moments, "moment/<bn>/..."); Apply(group) runs the Adam
/// step and, for G, the BN moving-average update (momentum .9).
class AdversarialSRModel : public gan::GroupedGan {
 public:
  bool   esrgan;
  Tensor vgg_w, vgg_b;
  Arrays frozen;

  AdversarialSRModel(bool enhanced, const Arrays& features, bool training = true,
                     std::vector<std::uint64_t> seeds = {7001, 7013, 0})
      : GroupedGan(seeds), esrgan(enhanced) {
    for (auto name : {"block1_conv1_W", "block1_conv1_b"}) frozen.emplace(name, features.at(name).detach().clone());
    vgg_w = frozen.at("block1_conv1_W"), vgg_b = frozen.at("block1_conv1_b");
    check::Require(vgg_w.shape() == nn::Shape({3, 3, 3, 64}) && vgg_b.shape() == nn::Shape({64}),
                   "1792 pretrained frozen VGG first-layer values");
    {
      nn::NoGradGuard guard;
      const Tensor    lr = nn::zeros({2, 16, 16, 3}), hr = nn::zeros({2, 32, 32, 3});
      Forward(lr, hr, hr, false);
    }
    if (training) ResetOptimizers();
  }
  void ResetOptimizers() { MakeAdam(.0001, .9, .999, {0, 1}); }

  Tensor Extract(const Tensor& rgb) {
    nn::Conv2DOptions o;
    return nn::conv2d(VggImageNetInput(rgb), vgg_w, o) + vgg_b;
  }

  /// Every tflow Evaluate field (debug ones included).
  Arrays Forward(const Tensor& lr, const Tensor& target, const Tensor& bicubic, bool gan) {
    auto&      g = Group(0);
    auto&      d = Group(1);
    auto&      s = Group(2);
    const auto inference = AdversarialSRGenerate(g, s, lr, bicubic, esrgan, false);
    const auto training  = AdversarialSRGenerate(g, s, lr, bicubic, esrgan, true);
    const Tensor real_score = AdversarialSRDiscriminate(d, target, esrgan);
    const Tensor fake_score = AdversarialSRDiscriminate(d, training.image, esrgan);
    const Tensor fake_inference_score = AdversarialSRDiscriminate(d, inference.image, esrgan);
    const Tensor predicted = Extract(training.image), truth = Extract(target).detach();
    const Tensor prediction_feature = (esrgan ? predicted : nn::relu(predicted)) / 255.0;
    const Tensor target_feature     = (esrgan ? truth : nn::relu(truth)) / 255.0;
    const Tensor difference = training.image - target, feature_difference = prediction_feature - target_feature;
    const Tensor pixel_loss   = esrgan ? nn::mean(nn::abs(difference)) : nn::mean(nn::square(difference));
    const Tensor content_loss = esrgan ? nn::mean(nn::abs(feature_difference)) : nn::mean(nn::square(feature_difference));
    Tensor       d_loss, adversarial_loss, relative_real, relative_fake;
    if (esrgan) {
      const auto loss = RelativisticAverageSR(real_score, fake_score);
      d_loss = loss.discriminator, adversarial_loss = loss.generator;
      relative_real = loss.real_relative, relative_fake = loss.fake_relative;
    } else {
      d_loss           = nn::mean(nn::softplus(-real_score)) + nn::mean(nn::softplus(fake_score));
      adversarial_loss = nn::mean(nn::softplus(-fake_score));
      relative_real = real_score, relative_fake = fake_score;
    }
    Tensor gan_loss = content_loss + double(esrgan ? .005f : .001f) * adversarial_loss;
    if (esrgan) gan_loss = gan_loss + double(.01f) * pixel_loss;
    const double phase  = gan ? 1 : 0;
    const Tensor g_loss = (1.0 - phase) * pixel_loss + phase * gan_loss;
    Arrays out{{"prediction", inference.image},
               {"prediction_training", training.image},
               {"residual", inference.residual},
               {"real_score", real_score},
               {"fake_score", fake_score},
               {"fake_inference_score", fake_inference_score},
               {"d_loss", d_loss},
               {"g_loss", g_loss},
               {"gan_loss", gan_loss},
               {"pixel_loss", pixel_loss},
               {"content_loss", content_loss},
               {"adversarial_loss", adversarial_loss},
               {"relative_real", relative_real},
               {"relative_fake", relative_fake},
               {"prediction_feature", prediction_feature},
               {"target_feature", target_feature},
               {"packed", training.packed},
               {"shuffled", training.shuffled}};
    for (const auto& [k, v] : training.moments) out.emplace("moment/" + k, v);
    return out;
  }

  Arrays Evaluate(const Tensor& x, const Tensor& hr, bool gan, bool debug = false) {
    nn::NoGradGuard guard;
    Arrays          all = Forward(x, hr, sr::SRBicubic(x, 32, 32), gan), out;
    for (auto& [k, v] : all)
      if (debug || !(k.rfind("moment/", 0) == 0 || k == "prediction_feature" || k == "target_feature" || k == "packed" ||
                     k == "shuffled"))
        out.emplace(k, v);
    gan::RequireFinite(out, "finite adversarial SR fields");
    return out;
  }
  /// LR ONLY inference (no HR / phase input).
  Tensor Predict(const Tensor& x) {
    nn::NoGradGuard guard;
    return AdversarialSRGenerate(Group(0), Group(2), x, sr::SRBicubic(x, 32, 32), esrgan, false).image;
  }
  Arrays Capture(int group, const Tensor& x, const Tensor& hr, bool gan) {
    const auto fields = Forward(x, hr, sr::SRBicubic(x, 32, 32), gan);
    Arrays     out    = GradientOf(group, fields.at(group ? "d_loss" : "g_loss"));
    if (group == 0)
      for (const auto& [k, v] : fields)
        if (k.rfind("moment/", 0) == 0) out.emplace(k, v.detach());
    return out;
  }
  void Apply(int group, const Arrays& captured) {
    GroupedGan::Apply(group, captured);
    if (group != 0) return;
    nn::NoGradGuard guard;
    auto&           stats = Group(2);
    for (const auto& [k, v] : captured) {
      if (k.rfind("moment/", 0) != 0) continue;
      const auto key = k.substr(7), slash = key.substr(key.rfind('/') + 1), name = key.substr(0, key.rfind('/'));
      const auto var = name + (slash == "mean" ? "/moving_mean" : "/moving_var");
      stats.set(var, stats.at(var) * double(.9f) + v * double(1.f - .9f));
    }
  }
  void CheckFrozen() const {
    check::Require(gan::BitEqual(vgg_w, frozen.at("block1_conv1_W")) && gan::BitEqual(vgg_b, frozen.at("block1_conv1_b")),
                   "ALL 1792 pretrained feature values BIT EXACT");
  }
};

}  // namespace srgan
