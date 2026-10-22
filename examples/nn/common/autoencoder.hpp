// Shared support for the autoencoder ports: a port of the parts of tflow's
// common/ml/{autoencoder,convolutional_autoencoder,variational_autoencoder,
// saliency}.h and common/nn/{convolutional_autoencoder,variational_autoencoder,
// smoothgrad}.h that convolutional_autoencoder and variational_autoencoder use.
//
// Data: the balanced 1000 fit / 200 validation / 200 test MNIST pools of
// $DATAMUNGE_DATA_DIR/saliency/saliency_data.npz (tflow's `just
// autoencoder-assets`). Checkpoints are .npz files of the parameter store
// instead of TF SaveV2 bundles. Parameter initialisation uses datamunge's
// generator, so trajectories match tflow only statistically; batch orders and
// Gaussian noise banks are bit-identical to tflow's.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace ae {

namespace nn = datamunge::nn;

// ---- data ----------------------------------------------------------------------------

struct SaliencyData {
  nn::Tensor images, labels, source;  // float [n, 28, 28, 1] in [0, 1]; int32 [n]; int32 [n]
};

inline SaliencyData Rows(const SaliencyData& d, const std::vector<std::int64_t>& rows) {
  return {nn::take_rows(d.images, rows), nn::take_rows(d.labels, rows), nn::take_rows(d.source, rows)};
}

inline nn::Tensor Find(const nn::NamedTensors& arrays, const std::string& name) {
  for (const auto& [key, value] : arrays)
    if (key == name) return value;
  throw std::runtime_error("missing array " + name);
}

/// The fit, validation and test pools with tflow's integrity checks.
inline std::vector<SaliencyData> LoadSaliency(const std::filesystem::path& root) {
  const auto file = root / "saliency_data.npz";
  check::Require(std::filesystem::exists(file),
                 file.string() + " missing; run `just autoencoder-assets` in tflow_examples or pass data=");
  const auto              arrays = nn::load_npz(file.string());
  std::vector<SaliencyData> result;
  std::set<int>           ids;
  std::set<std::string>   bytes;
  for (const std::string split : {"fit", "validation", "test"}) {
    const auto x = Find(arrays, split + "_images").contiguous();  // uint8 is read as int32
    const int  n = split == "fit" ? 1000 : 200;
    check::Require(x.shape() == nn::Shape({n, 28, 28, 1}), "source28x28 grayscale digit images");
    SaliencyData d{nn::div(nn::cast(x, nn::DType::Float32), 255.0), Find(arrays, split + "_labels").contiguous(),
                   Find(arrays, split + "_source_ids").contiguous()};
    check::Require(d.labels.dtype() == nn::DType::Int32 && d.source.dtype() == nn::DType::Int32 &&
                       d.labels.shape() == nn::Shape({n}) && d.source.shape() == nn::Shape({n}),
                   "aligned source/label vectors");
    const auto pixels = x.to_vector<int>();
    const auto labels = d.labels.to_vector<int>(), source = d.source.to_vector<int>();
    int        counts[10] = {};
    for (int b = 0; b < n; ++b) {
      const int id = source[b], label = labels[b];
      check::Require(label >= 0 && label < 10 && ids.insert(id).second, "source IDs disjoint,labels valid");
      check::Require(split == "test" ? id >= 60000 && id < 70000 : id >= 0 && id < 60000,
                     "original test/train boundary");
      ++counts[label];
      std::string row(784, '\0');
      for (int i = 0; i < 784; ++i) {
        check::Require(pixels[b * 784 + i] >= 0 && pixels[b * 784 + i] < 256, "uint8 pixels");
        row[i] = static_cast<char>(pixels[b * 784 + i]);
      }
      check::Require(bytes.insert(row).second, "no duplicated input bytes across all pools");
    }
    for (int c : counts) check::Require(c == n / 10, "balanced digit pools");
    result.push_back(d);
  }
  return result;
}

/// Copies tflow's provenance files next to the outputs when they exist.
inline void CopyManifests(const std::filesystem::path& data, const std::filesystem::path& root) {
  const auto opts = std::filesystem::copy_options::overwrite_existing;
  if (std::filesystem::exists(data / "manifest.json"))
    std::filesystem::copy_file(data / "manifest.json", root / "data_manifest.json", opts);
  const auto source = paths::Dataset("autoencoder") / "source_manifest.json";
  if (std::filesystem::exists(source)) std::filesystem::copy_file(source, root / "source_manifest.json", opts);
}

// ---- tensor helpers --------------------------------------------------------------------

/// Raw bytes, for exact-equality checks (tflow's tensor_data() comparisons).
inline std::string Bytes(const nn::Tensor& t) {
  const auto values = t.contiguous().to_vector<float>();
  return std::string(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
}

inline double MaxAbsDiff(const nn::Tensor& a, const nn::Tensor& b, std::int64_t count) {
  const auto x = a.contiguous().to_vector<float>(), y = b.contiguous().to_vector<float>();
  double     out = 0;
  for (std::int64_t i = 0; i < count; ++i) out = std::max(out, std::abs(double(x[i]) - y[i]));
  return out;
}

/// Mean squared error in double precision; the prediction must be finite in [0, 1].
inline double Mse(const nn::Tensor& prediction, const nn::Tensor& original) {
  check::Require(prediction.numel() == original.numel(), "aligned reconstruction pixels");
  const auto p = prediction.contiguous().to_vector<float>(), o = original.contiguous().to_vector<float>();
  double     sum = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const double x = p[i], y = o[i];
    check::Require(std::isfinite(x) && x >= 0 && x <= 1, "finite bounded sigmoid reconstruction");
    sum += (x - y) * (x - y);
  }
  return sum / double(p.size());
}

/// Per-pixel mean [784] of the fit images.
inline nn::Tensor PixelMean(const nn::Tensor& fit) {
  const auto  x = fit.contiguous().to_vector<float>();
  const auto  n = fit.size(0);
  std::vector<float> out(784);
  for (int i = 0; i < 784; ++i) {
    double sum = 0;
    for (std::int64_t b = 0; b < n; ++b) sum += x[b * 784 + i];
    out[i] = float(sum / double(n));
  }
  return nn::from_vector(out, {784});
}

/// `mean` [784] repeated n times as [n, 784].
inline nn::Tensor Constant(const nn::Tensor& mean, std::int64_t n) {
  return nn::tile(nn::reshape(mean, {1, 784}), {n, 1}).contiguous();
}

/// Values of every parameter (deep copies), in creation order.
inline nn::NamedTensors Arrays(const nn::ParameterStore& store) {
  nn::NamedTensors out;
  for (const auto& name : store.names()) out.emplace_back(name, store.at(name).detach().clone());
  return out;
}

inline std::vector<std::string> Snapshot(const nn::ParameterStore& store) {
  std::vector<std::string> out;
  for (const auto& name : store.names()) out.push_back(Bytes(store.at(name)));
  return out;
}

/// Checks every gradient is finite and every tensor gets a nonzero one; returns them named.
inline nn::NamedTensors CheckGradients(const nn::ParameterStore& store, const std::vector<nn::Tensor>& gradients,
                                       const std::string& finite, const std::string& nonzero) {
  nn::NamedTensors out;
  check::Require(gradients.size() == store.size(), "one gradient per parameter");
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    check::Require(gradients[i].defined(), nonzero);
    double norm = 0;
    for (float v : gradients[i].contiguous().to_vector<float>()) {
      check::Require(std::isfinite(v), finite);
      norm += double(v) * v;
    }
    check::Require(norm > 0, nonzero);
    out.emplace_back(store.names()[i], gradients[i]);
  }
  return out;
}

// ---- figures ---------------------------------------------------------------------------

/// One row of 28x28 panels (each [784] or [28, 28, 1] values in [0, 1]) scaled 4x,
/// 4 px apart on gray 220 background, as tflow's digit<k>.png.
inline void PanelRow(const std::filesystem::path& path, const std::vector<nn::Tensor>& panels) {
  std::vector<nn::Tensor> tiles;
  for (const auto& p : panels) tiles.push_back(nn::reshape(p, {28, 28, 1}));
  nn::save_image_grid(nn::stack(tiles, 0), path.string(), std::int64_t(tiles.size()), 0.0, 1.0, 4, 220.0 / 255.0,
                      4);
}

/// The b-th 784-pixel row of [n, 784] / [n, 28, 28, 1] probabilities.
inline nn::Tensor Row(const nn::Tensor& x, std::int64_t b) {
  return nn::reshape(nn::slice_axis(x, 0, b, b + 1), {784});
}

// ---- convolutional autoencoder -------------------------------------------------------

struct ConvAeOutput {
  nn::Tensor conv1, pool1, conv2, code, decoder1, reconstruction;
};

inline nn::Tensor ConvAeDecode(nn::ParameterStore& p, const nn::Tensor& code, nn::Tensor* decoder1 = nullptr) {
  nn::ConvLayerOptions relu;
  relu.init       = nn::init::he_uniform();
  relu.activation = nn::Activation::Relu;
  auto d1 = nn::conv2d(p, "decoder1", nn::upsample_nearest(code, 2), 8, 3, relu);
  if (decoder1) *decoder1 = d1;
  nn::ConvLayerOptions output;
  output.activation = nn::Activation::Sigmoid;
  return nn::conv2d(p, "decoder2", nn::upsample_nearest(d1, 2), 1, 3, output);
}

/// Conv3x3 1->8 ReLU, AvgPool2, Conv3x3 8->4 ReLU, AvgPool2 (7x7x4 code), then
/// nearest-repeat2, Conv3x3 4->8 ReLU, nearest-repeat2, Conv3x3 8->1 sigmoid.
inline ConvAeOutput ConvolutionalAutoencoder(nn::ParameterStore& p, const nn::Tensor& image) {
  ConvAeOutput         out;
  nn::ConvLayerOptions relu;
  relu.init       = nn::init::he_uniform();
  relu.activation = nn::Activation::Relu;
  out.conv1          = nn::conv2d(p, "encoder1", image, 8, 3, relu);
  out.pool1          = nn::avg_pool2d(out.conv1, 2, 2);
  out.conv2          = nn::conv2d(p, "encoder2", out.pool1, 4, 3, relu);
  out.code           = nn::avg_pool2d(out.conv2, 2, 2);
  out.reconstruction = ConvAeDecode(p, out.code, &out.decoder1);
  return out;
}

class ConvolutionalAutoencoderModel {
 public:
  nn::ParameterStore             parameters{1801};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit ConvolutionalAutoencoderModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      ConvolutionalAutoencoder(parameters, nn::zeros({1, 28, 28, 1}));
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer   = std::make_unique<nn::Optimizer>(parameters, .003, config);
    }
  }
  static nn::Tensor Loss(const nn::Tensor& reconstruction, const nn::Tensor& image) {
    return nn::mean(nn::square(reconstruction - image));
  }
  /// {code, reconstruction, loss}
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            net = ConvolutionalAutoencoder(parameters, images);
    return {net.code, net.reconstruction, Loss(net.reconstruction, images)};
  }
  std::vector<nn::Tensor> Activations(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            n = ConvolutionalAutoencoder(parameters, images);
    return {n.conv1, n.pool1, n.conv2, n.code, n.decoder1, n.reconstruction};
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return ConvAeDecode(parameters, codes);
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images) {
    auto net = ConvolutionalAutoencoder(parameters, images);
    return nn::grad(Loss(net.reconstruction, images), parameters.parameters());
  }
  /// One Adam step; returns the pre-update MSE.
  double Update(const nn::Tensor& images) {
    auto         loss  = Loss(ConvolutionalAutoencoder(parameters, images).reconstruction, images);
    const double value = loss.item<float>();
    check::Require(std::isfinite(value) && value >= 0, "finite pre-update MSE");
    optimizer->minimize(loss);
    return value;
  }
};

// ---- variational autoencoder ---------------------------------------------------------

inline nn::Tensor VaeReparameterize(const nn::Tensor& mu, const nn::Tensor& log_variance, const nn::Tensor& epsilon) {
  return mu + nn::exp(0.5 * log_variance) * epsilon;
}
/// KL(N(mu, exp(lv)) || N(0, I)) per row.
inline nn::Tensor VaeKl(const nn::Tensor& mu, const nn::Tensor& log_variance) {
  return 0.5 * nn::sum(nn::square(mu) + nn::exp(log_variance) - 1.0 - log_variance, {1});
}
/// Bernoulli NLL summed over pixels, stable for large |logits|.
inline nn::Tensor VaePixelNll(const nn::Tensor& logits, const nn::Tensor& target) {
  return nn::sum(nn::softplus(logits) - target * logits, {1});
}

struct VaeOutput {
  nn::Tensor mu, log_variance, z, logits, probability, reconstruction_nll, kl, loss;
};

inline nn::Tensor VaeDecodeLogits(nn::ParameterStore& p, const nn::Tensor& z) {
  return nn::dense(p, "output", nn::dense(p, "decoder", z, 64, nn::Activation::Relu), 784);
}

/// 784 -> 64 ReLU -> (mu, log variance) [2] each; z = mu + exp(lv / 2) eps;
/// 2 -> 64 ReLU -> 784 logits. Loss: mean over the batch of summed pixel NLL + KL.
inline VaeOutput VariationalAutoencoder(nn::ParameterStore& p, const nn::Tensor& image, const nn::Tensor& epsilon) {
  VaeOutput out;
  auto      h          = nn::dense(p, "encoder", image, 64, nn::Activation::Relu);
  out.mu                 = nn::dense(p, "mu", h, 2);
  out.log_variance       = nn::dense(p, "log_variance", h, 2);
  out.z                  = VaeReparameterize(out.mu, out.log_variance, epsilon);
  out.logits             = VaeDecodeLogits(p, out.z);
  out.probability        = nn::sigmoid(out.logits);
  out.reconstruction_nll = VaePixelNll(out.logits, image);
  out.kl                 = VaeKl(out.mu, out.log_variance);
  out.loss               = nn::mean(out.reconstruction_nll + out.kl, {0});
  return out;
}

class VariationalAutoencoderModel {
 public:
  nn::ParameterStore             parameters{1951};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit VariationalAutoencoderModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      VariationalAutoencoder(parameters, nn::zeros({1, 784}), nn::zeros({1, 2}));
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer   = std::make_unique<nn::Optimizer>(parameters, .003, config);
    }
  }
  static nn::Tensor Flatten(const nn::Tensor& images) { return nn::reshape(images, {images.size(0), 784}); }
  /// {mu, log_variance, z, probability, reconstruction_nll, kl, loss}, all checked finite.
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images, const nn::Tensor& noise) {
    nn::NoGradGuard         guard;
    auto                    n = VariationalAutoencoder(parameters, Flatten(images), noise);
    std::vector<nn::Tensor> out{n.mu, n.log_variance, n.z, n.probability, n.reconstruction_nll, n.kl, n.loss};
    for (auto& t : out) {
      t = t.contiguous();
      for (float v : t.to_vector<float>()) check::Require(std::isfinite(v), "ALL posterior/output/objective values finite");
    }
    return out;
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::sigmoid(VaeDecodeLogits(parameters, codes));
  }
  std::vector<nn::Tensor> Gradient(const nn::Tensor& images, const nn::Tensor& noise) {
    return nn::grad(VariationalAutoencoder(parameters, Flatten(images), noise).loss, parameters.parameters());
  }
  /// One Adam step; returns the pre-update negative ELBO.
  double Update(const nn::Tensor& images, const nn::Tensor& noise) {
    auto         loss  = VariationalAutoencoder(parameters, Flatten(images), noise).loss;
    const double value = loss.item<float>();
    check::Require(std::isfinite(value) && value >= 0, "finite negativeELBO");
    optimizer->minimize(loss);
    return value;
  }
};

/// tflow's nn::GaussianNoise: Box-Muller pairs from std::mt19937(seed), as
/// float64 [samples, shape...].
inline nn::Tensor GaussianNoise(const nn::Shape& shape, std::int64_t samples, std::uint32_t seed) {
  nn::Shape full{samples};
  full.insert(full.end(), shape.begin(), shape.end());
  std::vector<double> bank(static_cast<std::size_t>(nn::shape_numel(full)));
  std::mt19937        rng(seed);
  constexpr double    tau = 6.283185307179586476925286766559;
  for (std::size_t i = 0; i < bank.size(); i += 2) {
    const double u = (double(rng()) + .5) / 4294967296., v = (double(rng()) + .5) / 4294967296.;
    const double radius = std::sqrt(-2 * std::log(u));
    bank[i]             = radius * std::cos(tau * v);
    if (i + 1 < bank.size()) bank[i + 1] = radius * std::sin(tau * v);
  }
  return nn::from_vector(bank, full);
}

struct VaeNoiseBank {
  nn::Tensor standard, epsilon;  // float64 / float32 [samples, n, 2]
};
inline VaeNoiseBank VaeNoise(std::int64_t n, std::int64_t samples, std::uint32_t seed) {
  auto standard = GaussianNoise({n, 2}, samples, seed);
  return {standard, nn::cast(standard, nn::DType::Float32).contiguous()};
}
inline nn::Tensor VaeNoiseAt(const VaeNoiseBank& bank, std::int64_t k) {
  return nn::select(bank.epsilon, 0, k).contiguous();
}

/// Monte Carlo {reconstruction NLL, KL, negative ELBO} per image over every sample of `bank`.
inline std::vector<double> VaeScore(VariationalAutoencoderModel& model, const nn::Tensor& images,
                                    const VaeNoiseBank& bank) {
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
