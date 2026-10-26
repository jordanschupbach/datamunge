// Port of tflow's common/ml/encoder_pretraining.h + common/nn/encoder_pretraining.h: a frozen
// (pretrained or random) 784->32 ReLU encoder with fitting-only feature standardization and a
// supervised 32->10 softmax head. The encoder arrays are constants (never in the optimised
// store), so only the 330 head parameters receive gradients/updates. Array names match
// tflow's ("encoder/W", "head/linear/W", ...).
#pragma once

#include <map>

#include "common/autoencoder.hpp"

namespace ae {

/// relu(flatten(images) W + b) with a source autoencoder's encoder arrays.
inline nn::Tensor TransferFeatures(const nn::Tensor& images, const nn::NamedTensors& source) {
  nn::NoGradGuard guard;
  return nn::relu(nn::matmul(Flatten(images), Find(source, "encoder/W")) + Find(source, "encoder/b")).contiguous();
}

/// Fitting-only per-feature {mean, population std floored at 1e-6}, float32 [32] each.
inline std::pair<nn::Tensor, nn::Tensor> TransferScaling(const nn::Tensor& code) {
  const auto         c = code.contiguous().to_vector<float>();
  const auto         n = code.size(0);
  std::vector<float> center(32), scale(32);
  for (int j = 0; j < 32; ++j) {
    double mean = 0, var = 0;
    for (std::int64_t i = 0; i < n; ++i) mean += c[i * 32 + j];
    mean /= double(n);
    for (std::int64_t i = 0; i < n; ++i) {
      const double d = double(c[i * 32 + j]) - mean;
      var += d * d;
    }
    center[j] = float(mean);
    scale[j]  = float(std::max(1e-6, std::sqrt(var / double(n))));
  }
  return {nn::from_vector(center, {32}), nn::from_vector(scale, {32})};
}

struct EncoderClassifierOutput {
  nn::Tensor code, features, logits, probability, loss;
};

class EncoderClassifierModel {
 public:
  nn::Tensor                     encoder_w, encoder_b, center, scale;
  nn::ParameterStore             head{4703};
  std::unique_ptr<nn::Optimizer> optimizer;

  EncoderClassifierModel(const nn::NamedTensors& source, const nn::Tensor& center_, const nn::Tensor& scale_, bool training = true)
      : encoder_w(Find(source, "encoder/W").detach().clone()),
        encoder_b(Find(source, "encoder/b").detach().clone()),
        center(center_),
        scale(scale_) {
    check::Require(encoder_w.shape() == nn::Shape({784, 32}) && encoder_b.shape() == nn::Shape({32}), "784->32 source encoder");
    {
      nn::NoGradGuard guard;
      Forward(nn::zeros({1, 28, 28, 1}));
    }
    if (training) optimizer = MakeAdam(head);  // ONLY the supervised linear head is optimised
  }
  /// Frozen encoder (constants), fit-only standardization, linear head; loss with labels if given.
  EncoderClassifierOutput Forward(const nn::Tensor& images, const nn::Tensor& labels = {}) {
    EncoderClassifierOutput o;
    o.code        = nn::relu(nn::matmul(Flatten(images), encoder_w) + encoder_b);
    o.features    = (o.code - center) / scale;
    o.logits      = nn::dense(head, "linear", o.features, 10);
    o.probability = nn::softmax(o.logits, 1);
    if (labels.defined())
      o.loss = nn::mean(-nn::sum(nn::one_hot(labels, 10) * nn::log_softmax(o.logits, 1), {1}), {0});
    return o;
  }
  /// {code, features, logits, probability}: image-only inference.
  std::vector<nn::Tensor> Predict(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            o = Forward(images);
    return {o.code.contiguous(), o.features.contiguous(), o.logits.contiguous(), o.probability.contiguous()};
  }
  /// All 330 head cross-entropy derivatives.
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& labels) {
    return nn::grad(Forward(images, labels).loss, head.parameters());
  }
  void Update(const nn::Tensor& images, const nn::Tensor& labels) { optimizer->minimize(Forward(images, labels).loss); }
  /// encoder/W, encoder/b, head/<name>... (deep copies)
  nn::NamedTensors Arrays() const {
    nn::NamedTensors out{{"encoder/W", encoder_w.clone()}, {"encoder/b", encoder_b.clone()}};
    for (const auto& name : head.names()) out.emplace_back("head/" + name, head.at(name).detach().clone());
    return out;
  }
  /// Loads the head from an Arrays() file (tflow's initial_arrays.npz/best).
  void LoadHead(const std::filesystem::path& file) {
    const auto arrays = nn::load_npz(file.string());
    for (const auto& name : head.names()) head.set(name, Find(arrays, "head/" + name));
  }
  std::vector<std::string> EncoderSnapshot() const { return {Bytes(encoder_w), Bytes(encoder_b)}; }
};

/// {mean NLL, accuracy} of class probabilities against integer labels.
inline std::vector<double> TransferScore(const nn::Tensor& probability, const nn::Tensor& labels) {
  const auto p = probability.contiguous().to_vector<float>();
  const auto l = labels.contiguous().to_vector<int>();
  double     nll = 0, correct = 0;
  for (std::size_t b = 0; b < l.size(); ++b) {
    const int truth = l[b];
    int       best  = 0;
    for (int c = 0; c < 10; ++c) {
      const double v = p[b * 10 + c];
      check::Require(std::isfinite(v) && v >= 0 && v <= 1, "finiteclassprobabilities");
      if (v > p[b * 10 + best]) best = c;
    }
    nll -= std::log(std::max(1e-30, double(p[b * 10 + truth])));
    correct += best == truth;
  }
  return {nll / double(l.size()), correct / double(l.size())};
}

}  // namespace ae
