// Port of tflow's common/nn/gumbel_categorical_autoencoder.h and
// common/ml/gumbel_categorical_autoencoder.h: a four-variable 8-way categorical
// VAE trained with the Gumbel-softmax relaxation (explicit Gumbel noise banks,
// bit-identical to tflow's), with hard Gumbel-max samples for diagnostics.
#pragma once

#include "common/beta_variational_autoencoder.hpp"

namespace ae {

struct GumbelCategoricalSample {
  nn::Tensor log_probability, probability, relaxed, indices;  // indices int64
};

/// q = softmax(logits), y = softmax((logits + g) / temperature), hard = argmax(logits + g).
inline GumbelCategoricalSample GumbelCategorical(const nn::Tensor& logits, const nn::Tensor& noise,
                                                 float temperature) {
  check::Require(std::isfinite(temperature) && temperature > 0, "finite positive Gumbel temperature");
  GumbelCategoricalSample o;
  o.log_probability = nn::log_softmax(logits, -1);
  o.probability     = nn::exp(o.log_probability);
  auto scores       = logits + noise;
  o.relaxed         = nn::softmax(scores / double(temperature), -1);
  o.indices         = nn::argmax(scores, -1);
  return o;
}

/// KL(q || Uniform(classes)) over the last axis.
inline nn::Tensor CategoricalUniformKl(const nn::Tensor& probability, const nn::Tensor& log_probability, int classes) {
  return nn::sum(probability * (log_probability + double(float(std::log(classes)))), {-1});
}

inline nn::Tensor GumbelDecoder(nn::ParameterStore& p, const nn::Tensor& code) {
  return nn::dense(p, "output", nn::dense(p, "decoder", code, 32, nn::Activation::Relu), 784);
}

struct GumbelCategoricalAutoencoderOutput {
  nn::Tensor logits, log_probability, probability, relaxed, indices, hard, code, soft_prediction, hard_prediction, nll,
      hard_nll, kl, loss, hard_objective;
};

/// 784 -> 32 ReLU -> 4x8 logits; decoder 32 -> 32 ReLU -> 784 logits (soft relaxed and hard one-hot codes).
inline GumbelCategoricalAutoencoderOutput GumbelCategoricalAutoencoder(nn::ParameterStore& p, const nn::Tensor& image,
                                                                       const nn::Tensor& noise, float temperature) {
  GumbelCategoricalAutoencoderOutput o;
  const std::int64_t n           = image.size(0);
  auto               h           = nn::dense(p, "encoder", image, 32, nn::Activation::Relu);
  auto               flat_logits = nn::reshape(nn::dense(p, "categorical", h, 32), {n * 4, 8});
  auto               sample      = GumbelCategorical(flat_logits, nn::reshape(noise, {n * 4, 8}), temperature);
  o.logits          = nn::reshape(flat_logits, {n, 4, 8});
  o.log_probability = nn::reshape(sample.log_probability, {n, 4, 8});
  o.probability     = nn::reshape(sample.probability, {n, 4, 8});
  o.relaxed         = nn::reshape(sample.relaxed, {n, 4, 8});
  o.indices         = nn::cast(nn::reshape(sample.indices, {n, 4}), nn::DType::Int32);
  o.hard            = nn::one_hot(o.indices, 8);
  o.code            = nn::reshape(o.relaxed, {n, 32});
  auto soft_logits = GumbelDecoder(p, o.code), hard_logits = GumbelDecoder(p, nn::reshape(o.hard, {n, 32}));
  o.soft_prediction = nn::sigmoid(soft_logits);
  o.hard_prediction = nn::sigmoid(hard_logits);
  o.nll             = VaePixelNll(soft_logits, image);
  o.hard_nll        = VaePixelNll(hard_logits, image);
  o.kl = nn::sum(nn::reshape(CategoricalUniformKl(sample.probability, sample.log_probability, 8), {n, 4}), {1});
  o.loss           = nn::mean(o.nll + o.kl, {0});
  o.hard_objective = nn::mean(o.hard_nll + o.kl, {0});
  return o;
}

class GumbelCategoricalModel {
 public:
  nn::ParameterStore             parameters{2903};
  std::unique_ptr<nn::Optimizer> optimizer;
  float                          temperature;

  explicit GumbelCategoricalModel(float temperature_ = .7f, bool training = true) : temperature(temperature_) {
    {
      nn::NoGradGuard guard;
      GumbelCategoricalAutoencoder(parameters, nn::zeros({1, 784}), nn::zeros({1, 4, 8}), temperature);
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  static nn::Tensor Flatten(const nn::Tensor& images) { return nn::reshape(images, {images.size(0), 784}); }
  /// Named fields as tflow's Evaluate map (indices int32, everything else float32 and finite).
  nn::NamedTensors Evaluate(const nn::Tensor& images, const nn::Tensor& noise) {
    nn::NoGradGuard  guard;
    auto             o = GumbelCategoricalAutoencoder(parameters, Flatten(images), noise, temperature);
    nn::NamedTensors out{{"logits", o.logits},
                         {"log_probability", o.log_probability},
                         {"probability", o.probability},
                         {"relaxed", o.relaxed},
                         {"indices", o.indices},
                         {"hard", o.hard},
                         {"soft_prediction", o.soft_prediction},
                         {"hard_prediction", o.hard_prediction},
                         {"nll", o.nll},
                         {"hard_nll", o.hard_nll},
                         {"kl", o.kl},
                         {"loss", o.loss},
                         {"hard_objective", o.hard_objective}};
    for (auto& [name, t] : out) {
      t = t.contiguous();
      if (t.dtype() == nn::DType::Float32)
        for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "all categorical outputs finite");
    }
    return out;
  }
  /// Target-free decoding of four 8-way simplex codes [n, 4, 8].
  nn::Tensor DecodeSoft(const nn::Tensor& probabilities) {
    check::Require(probabilities.dtype() == nn::DType::Float32 && probabilities.dim() == 3 &&
                       probabilities.size(1) == 4 && probabilities.size(2) == 8,
                   "four8-way simplex codes");
    const auto v = probabilities.contiguous().to_vector<float>();
    for (std::size_t g = 0; g < v.size() / 8; ++g) {
      double sum = 0;
      for (int k = 0; k < 8; ++k) {
        const float value = v[g * 8 + k];
        check::Require(std::isfinite(value) && value >= 0 && value <= 1, "simplex values in0..1");
        sum += value;
      }
      check::Require(std::abs(sum - 1) < 2e-6, "each categorical code sums to1");
    }
    nn::NoGradGuard guard;
    return nn::sigmoid(GumbelDecoder(parameters, nn::reshape(probabilities, {probabilities.size(0), 32}))).contiguous();
  }
  /// Target-free decoding of hard category IDs [n, 4] (int32 in 0..7).
  nn::Tensor Decode(const nn::Tensor& ids) {
    check::Require(ids.dtype() == nn::DType::Int32 && ids.dim() == 2 && ids.size(1) == 4 && ids.size(0) > 0,
                   "four hard category IDs");
    for (int id : ids.contiguous().to_vector<int>()) check::Require(id >= 0 && id < 8, "category IDs in0..7");
    nn::NoGradGuard guard;
    return nn::sigmoid(GumbelDecoder(parameters, nn::reshape(nn::one_hot(ids, 8), {ids.size(0), 32}))).contiguous();
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& noise) {
    return nn::grad(GumbelCategoricalAutoencoder(parameters, Flatten(images), noise, temperature).loss,
                    parameters.parameters());
  }
  double Update(const nn::Tensor& images, const nn::Tensor& noise) {
    auto         loss  = GumbelCategoricalAutoencoder(parameters, Flatten(images), noise, temperature).loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value) && value >= 0, "finite relaxed objective");
    optimizer->minimize(loss);
    return value;
  }
};

struct GumbelNoiseBank {
  nn::Tensor uniform, gumbel, noise;  // float64, float64, float32 [samples, n, groups, classes]
};

/// tflow's GumbelNoise: u = (mt19937() + .5) / 2^32, g = -log(-log u).
inline GumbelNoiseBank GumbelNoise(std::int64_t n, std::int64_t samples, std::uint32_t seed, std::int64_t groups = 4,
                                   std::int64_t classes = 8) {
  check::Require(n > 0 && samples > 0 && groups > 0 && classes > 1, "positive categorical noise shape");
  const std::size_t   count = std::size_t(samples * n * groups * classes);
  std::vector<double> uniform(count), gumbel(count);
  std::vector<float>  noise(count);
  std::mt19937        random(seed);
  for (std::size_t i = 0; i < count; ++i) {
    const double u = (double(random()) + .5) / 4294967296., g = -std::log(-std::log(u));
    uniform[i] = u, gumbel[i] = g, noise[i] = float(g);
  }
  const nn::Shape shape{samples, n, groups, classes};
  return {nn::from_vector(uniform, shape), nn::from_vector(gumbel, shape), nn::from_vector(noise, shape)};
}
inline nn::Tensor GumbelNoiseAt(const GumbelNoiseBank& bank, std::int64_t k) {
  check::Require(k >= 0 && k < bank.noise.size(0), "available Gumbel draw");
  return nn::select(bank.noise, 0, k).contiguous();
}

/// {soft NLL, hard NLL, categorical KL, relaxed objective} per image (NLLs averaged over draws, KL from draw 0).
inline std::vector<double> GumbelScore(GumbelCategoricalModel& model, const nn::Tensor& images,
                                       const GumbelNoiseBank& noise) {
  double             nll = 0, hard = 0, kl = 0;
  const std::int64_t n = images.size(0), samples = noise.noise.size(0);
  for (std::int64_t k = 0; k < samples; ++k) {
    auto       f  = model.Evaluate(images, GumbelNoiseAt(noise, k));
    const auto a  = Find(f, "nll").to_vector<float>(), h = Find(f, "hard_nll").to_vector<float>(),
               kv = Find(f, "kl").to_vector<float>();
    for (std::int64_t b = 0; b < n; ++b) {
      nll += a[b];
      hard += h[b];
      if (k == 0) kl += kv[b];
    }
  }
  nll /= double(n * samples);
  hard /= double(n * samples);
  kl /= double(n);
  return {nll, hard, kl, nll + kl};
}

}  // namespace ae
