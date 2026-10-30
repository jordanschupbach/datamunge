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
#include <memory>
#include <ostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace ae {

namespace nn = datamunge::nn;

// ---- data ----------------------------------------------------------------------------

using SaliencyData=nn::models::SaliencyData;

inline SaliencyData Rows(const SaliencyData& d, const std::vector<std::int64_t>& rows) {
  return {nn::take_rows(d.images, rows), nn::take_rows(d.labels, rows), nn::take_rows(d.source, rows)};
}

inline nn::Tensor Find(const nn::NamedTensors& arrays, const std::string& name) {
  for (const auto& [key, value] : arrays)
    if (key == name) return value;
  throw std::runtime_error("missing array " + name);
}

/// The fit, validation and test pools with tflow's integrity checks.
inline std::vector<SaliencyData> LoadSaliency(const std::filesystem::path&root){dataset_assets::EnsureSaliency(root.string());return nn::models::LoadSaliency(root);}

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

// ---- dense undercomplete autoencoder (tflow's common/nn/autoencoder.h + ml/autoencoder.h) ----

/// [n, 28, 28, 1] (or any n x 784 values) as [n, 784] original pixels.
inline nn::Tensor Flatten(const nn::Tensor& images) { return nn::reshape(images, {images.size(0), 784}); }

struct DenseAeOutput {
  nn::Tensor code, reconstruction;
};

/// "encoder" features -> latent ReLU, "decoder" latent -> features sigmoid (Glorot uniform, zero biases).
inline DenseAeOutput UndercompleteAutoencoder(nn::ParameterStore& p, const nn::Tensor& input,
                                              std::int64_t features = 784, std::int64_t latent = 32) {
  check::Require(features > 1 && latent > 0 && latent < features, "strictly undercomplete positive bottleneck");
  auto code = nn::dense(p, "encoder", input, latent, nn::Activation::Relu);
  return {code, nn::dense(p, "decoder", code, features, nn::Activation::Sigmoid)};
}

/// tflow's ReconstructionMse: mean over every element.
inline nn::Tensor ReconstructionMse(const nn::Tensor& prediction, const nn::Tensor& original) {
  return nn::mean(nn::square(prediction - original));
}

/// Adam(.003), the optimiser of every short autoencoder example.
inline std::unique_ptr<nn::Optimizer> MakeAdam(nn::ParameterStore& store, double learning_rate = .003) {
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Adam;
  return std::make_unique<nn::Optimizer>(store, learning_rate, config);
}

/// tflow's CorruptForDenoising: clamp(float(x + sigma z), 0, 1) with z one float64 bank [1, shape...].
inline nn::Tensor CorruptForDenoising(const nn::Tensor& clean, const nn::Tensor& standard_noise, float sigma) {
  nn::Shape shape{1};
  shape.insert(shape.end(), clean.shape().begin(), clean.shape().end());
  check::Require(clean.dtype() == nn::DType::Float32 && clean.dim() >= 2 &&
                     standard_noise.dtype() == nn::DType::Float64 && standard_noise.shape() == shape &&
                     std::isfinite(sigma) && sigma >= 0,
                 "aligned single Gaussian bank and nonnegative sigma");
  const auto          x = clean.contiguous().to_vector<float>();
  const auto          z = standard_noise.contiguous().to_vector<double>();
  std::vector<float>  out(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    check::Require(std::isfinite(x[i]) && x[i] >= 0 && x[i] <= 1 && std::isfinite(z[i]),
                   "finite unit-range clean pixels/noise");
    out[i] = std::clamp(float(double(x[i]) + double(sigma) * z[i]), 0.f, 1.f);
  }
  return nn::from_vector(out, clean.shape());
}

/// tflow's data.npz record: the three pools (+ any extra arrays).
inline void SaveData(const std::filesystem::path& file, const std::vector<SaliencyData>& pools,
                     const nn::NamedTensors& extra = {}) {
  nn::NamedTensors arrays{{"fit_images", pools[0].images},         {"validation_images", pools[1].images},
                          {"test_images", pools[2].images},        {"fit_source_ids", pools[0].source},
                          {"validation_source_ids", pools[1].source}, {"test_source_ids", pools[2].source},
                          {"test_labels", pools[2].labels}};
  arrays.insert(arrays.end(), extra.begin(), extra.end());
  nn::save_npz(file.string(), arrays);
}

/// Index of the first record of `digit` in a label vector (gallery rows).
inline int FirstOfDigit(const nn::Tensor& labels, int digit) {
  const auto l = labels.contiguous().to_vector<int>();
  for (std::size_t b = 0; b < l.size(); ++b)
    if (l[b] == digit) return int(b);
  throw std::runtime_error("no record of digit " + std::to_string(digit));
}

/// Writes digit<k>.png (one panel row per digit: the b-th row of every tensor) and its html sections.
inline void DigitGallery(const std::filesystem::path& root, const SaliencyData& test,
                         const std::vector<nn::Tensor>& panels, std::ostream& html) {
  const auto source = test.source.contiguous().to_vector<int>();
  for (int digit = 0; digit < 10; ++digit) {
    const int               b = FirstOfDigit(test.labels, digit);
    std::vector<nn::Tensor> row;
    for (const auto& t : panels) row.push_back(Row(t, b));
    const std::string file = "digit" + std::to_string(digit) + ".png";
    PanelRow(root / file, row);
    html << "<section><h2>Digit" << digit << ",source" << source[b] << "</h2><img src='" << file << "'></section>";
  }
}

/// Byte-identical parameter stores (matched-initialisation checks).
inline bool SameArrays(const nn::ParameterStore& a, const nn::ParameterStore& b) {
  if (a.names() != b.names()) return false;
  for (const auto& name : a.names())
    if (Bytes(a.at(name)) != Bytes(b.at(name))) return false;
  return true;
}

// ---- 2.3 applications: frozen checkpoints and case figures ------------------------------

/// tflow's ml::UndercompleteModel (784->32 ReLU->784 sigmoid, seed 1471) for the frozen
/// applications; the training port keeps its own copy in undercomplete_autoencoder.cpp.
class UndercompleteModel {
 public:
  nn::ParameterStore             parameters{1471};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit UndercompleteModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      UndercompleteAutoencoder(parameters, nn::zeros({1, 784}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  /// {code, reconstruction, loss}
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            x = Flatten(images);
    auto            n = UndercompleteAutoencoder(parameters, x);
    return {n.code, n.reconstruction, ReconstructionMse(n.reconstruction, x)};
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::dense(parameters, "decoder", codes, 784, nn::Activation::Sigmoid);
  }
};

/// Restores a frozen checkpoint directory written by a training port (best.npz) or by
/// tflow (model_arrays.npz = its restored best checkpoint); returns the file used.
inline std::filesystem::path LoadFrozen(nn::ParameterStore& store, const std::filesystem::path& directory,
                                        const std::string& what) {
  for (const char* name : {"best.npz", "model_arrays.npz"}) {
    const auto file = directory / name;
    if (std::filesystem::exists(file)) {
      store.load(file.string());
      return file;
    }
  }
  check::Require(false, what + ": no best.npz/model_arrays.npz in " + directory.string() + " (run the training port first)");
  return {};
}

/// One case strip as tflow's case_<k>.png: the b-th 784-pixel row of every tensor, 3x nearest,
/// 4 px apart on gray 220 (84 x (88 n - 4) RGB).
inline void CaseStrip3(const std::filesystem::path& path, const std::vector<nn::Tensor>& panels, std::int64_t b) {
  std::vector<nn::Tensor> tiles;
  for (const auto& p : panels) tiles.push_back(nn::reshape(Row(p, b), {28, 28, 1}));
  nn::save_image_grid(nn::tile(nn::stack(tiles, 0), {1, 1, 1, 3}), path.string(), std::int64_t(tiles.size()), 0.0, 1.0,
                      4, 220.0 / 255.0, 3);
}

/// tflow's latent-application Mosaic: [n, 784] rows, `columns` per row, 3x nearest, 4 px gray 220 gaps.
inline void Mosaic3(const std::filesystem::path& path, const nn::Tensor& values, std::int64_t columns) {
  const auto n = values.size(0);
  nn::save_image_grid(nn::tile(nn::reshape(values, {n, 28, 28, 1}), {1, 1, 1, 3}), path.string(), columns, 0.0, 1.0, 4,
                      220.0 / 255.0, 3);
}

}  // namespace ae
