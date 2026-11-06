/// @file mobile_classifier.cpp
/// @brief Native complete MobileNet fitting, validation selection and NPZ inference restoration.
#include <algorithm>
#include <cmath>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/io.hpp>
#include <datamunge/nn/losses.hpp>
#include <datamunge/nn/mobile_classifier.hpp>
#include <datamunge/nn/mobilenet_models.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/training.hpp>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>

namespace datamunge::nn {
/// @brief Private strict input and checkpoint validation helpers.
namespace {
/// @brief Require finite CPU float32 NHWC RGB pixels in the documented range.
void pixels(const Tensor& x, int side) {
  if (!x.defined() || x.dtype() != DType::Float32 || x.dim() != 4 || x.shape()[0] < 1 ||
      x.shape()[1] != side || x.shape()[2] != side || x.shape()[3] != 3)
    throw std::invalid_argument("MobileNet requires nonempty float32 NHWC RGB images");
  std::vector<float> data;
  try {
    data = x.to_vector<float>();
  } catch (const std::logic_error&) {
    throw std::invalid_argument("MobileNet inputs must be CPU tensors");
  }
  for (float p : data)
    if (!std::isfinite(p) || p < 0 || p > 1)
      throw std::invalid_argument("MobileNet pixels must be finite in [0,1]");
}
/// @brief Require CPU int64 class labels aligned to an image batch.
std::vector<std::int64_t> labels_for(const Tensor& y, std::int64_t count, int classes) {
  if (!y.defined() || y.dtype() != DType::Int64 || y.shape() != Shape{count})
    throw std::invalid_argument("MobileNet label shape/dtype mismatch");
  std::vector<std::int64_t> labels;
  try {
    labels = y.to_vector<std::int64_t>();
  } catch (const std::logic_error&) {
    throw std::invalid_argument("MobileNet labels must be CPU tensors");
  }
  for (auto value : labels)
    if (value < 0 || value >= classes)
      throw std::invalid_argument("MobileNet label out of range");
  return labels;
}
/// @brief Copy parameters/moments into namespaced inference archive entries.
void append_store(NamedTensors& values, const ParameterStore& p, const std::string& prefix) {
  for (const auto& name : p.names())
    values.emplace_back(prefix + name, p.at(name).detach().clone());
}
/// @brief Strictly restore one namespace into an already initialized architecture.
void restore_store(ParameterStore& store, const NamedTensors& data, const std::string& prefix) {
  std::set<std::string> seen;
  for (const auto& [name, value] : data)
    if (name.rfind(prefix, 0) == 0) {
      auto key = name.substr(prefix.size());
      if (!store.contains(key) || !seen.insert(key).second || value.dtype() != DType::Float32 ||
          value.shape() != store.at(key).shape())
        throw std::runtime_error("MobileNet checkpoint tensor name/shape/dtype mismatch");
      for (float x : value.to_vector<float>())
        if (!std::isfinite(x))
          throw std::runtime_error("nonfinite MobileNet checkpoint");
      store.set(key, value);
    }
  if (seen.size() != store.names().size())
    throw std::runtime_error("incomplete MobileNet checkpoint");
}
} // namespace
/// @brief Complete native model state, with architecture separate from trainable/moving tensors.
struct MobileNetClassifier::Impl {
  int            classes;    ///< Output width.
  int            side;       ///< Square RGB input side.
  double         width;      ///< MobileNet width multiplier.
  ParameterStore parameters; ///< Learned convolution, classifier and BN affine leaves.
  ParameterStore moments;    ///< Moving BN mean and variance tensors.

  /// @brief Initialize all layers once before constructing an optimizer/checkpoint validator.
  Impl(int count, int size, double multiplier, std::uint64_t seed)
      : classes(count), side(size), width(multiplier), parameters(seed), moments(seed) {
    if (classes < 2 || classes > 1000 || side < 16 || side > 256 || side % 16 ||
        !std::isfinite(width) || width < .03125 || width > 1)
      throw std::invalid_argument("invalid MobileNet classifier architecture");
    NoGradGuard no_grad;
    forward(zeros({1, side, side, 3}), false);
  }
  /// @brief Run all thirteen depthwise/pointwise blocks with explicit BN training mode.
  Tensor forward(const Tensor& x, bool training) {
    models::MobileNetV1Config config;
    config.alpha   = width;
    config.classes = classes;
    return models::MobileNetV1(parameters, moments, x * 2 - 1, config, training).logits;
  }
};
MobileNetClassifier::MobileNetClassifier(int classes, int side, double width, std::uint64_t seed)
    : impl_(std::make_unique<Impl>(classes, side, width, seed)) {}
MobileNetClassifier::~MobileNetClassifier()                                               = default;
MobileNetClassifier::MobileNetClassifier(MobileNetClassifier&& other) noexcept            = default;
MobileNetClassifier& MobileNetClassifier::operator=(MobileNetClassifier&& other) noexcept = default;
Tensor               MobileNetClassifier::predict(const Tensor& images) {
  pixels(images, impl_->side);
  NoGradGuard no_grad;
  return softmax(impl_->forward(images, false), -1).detach().clone();
}
std::int64_t MobileNetClassifier::parameter_count() const {
  return impl_->parameters.count();
}
MobileClassifierTraining MobileNetClassifier::fit(
    const Tensor&           images,
    const Tensor&           labels,
    const Tensor&           validation,
    const Tensor&           validation_labels,
    MobileClassifierOptions options) {
  auto& p = *impl_;
  pixels(images, p.side);
  pixels(validation, p.side);
  labels_for(labels, images.shape()[0], p.classes);
  auto truth = labels_for(validation_labels, validation.shape()[0], p.classes);
  if (!options.epochs || !options.batch_size || !std::isfinite(options.learning_rate) ||
      options.learning_rate <= 0 || !std::isfinite(options.brightness_jitter) ||
      options.brightness_jitter < 0 || options.brightness_jitter > .5 ||
      !std::isfinite(options.grayscale_probability) || options.grayscale_probability < 0 ||
      options.grayscale_probability > 1)
    throw std::invalid_argument("invalid MobileNet training options");
  Optimizer       optimizer(p.parameters, options.learning_rate, {OptimizerKind::Adam});
  BatchOrder      order(std::size_t(images.shape()[0]), options.batch_size, options.seed);
  std::mt19937_64 random(options.seed);
  std::uniform_real_distribution<float> gain(
      float(1 - options.brightness_jitter), float(1 + options.brightness_jitter));
  MobileClassifierTraining result;
  double                   best         = std::numeric_limits<double>::infinity();
  auto                     best_weights = p.parameters.clone(), best_moments = p.moments.clone();
  for (std::size_t epoch = 1; epoch <= options.epochs; ++epoch) {
    order.begin_epoch();
    double      total = 0;
    std::size_t count = 0;
    for (auto ids = order.next(); !ids.empty(); ids = order.next()) {
      auto       batch  = take_rows(images, ids);
      auto       values = batch.to_vector<float>();
      const auto stride = std::size_t(p.side * p.side * 3);
      for (std::size_t row = 0; row < ids.size(); ++row) {
        float g = gain(random);
        if (options.grayscale_probability > 0 &&
            std::generate_canonical<double, 53>(random) < options.grayscale_probability)
          for (std::size_t i = row * stride; i < (row + 1) * stride; i += 3) {
            float luminance = .299f * values[i] + .587f * values[i + 1] + .114f * values[i + 2];
            values[i] = values[i + 1] = values[i + 2] = luminance;
          }
        for (std::size_t i = row * stride; i < (row + 1) * stride; ++i)
          values[i] = std::clamp(values[i] * g, 0.f, 1.f);
      }
      batch         = from_vector(values, batch.shape());
      auto   logits = p.forward(batch, true);
      auto   loss   = sparse_softmax_cross_entropy(logits, take_rows(labels, ids));
      double value  = loss.item();
      if (!std::isfinite(value))
        throw std::runtime_error("nonfinite MobileNet training loss");
      optimizer.minimize(loss);
      total += value * ids.size();
      count += ids.size();
    }
    NoGradGuard no_grad;
    auto        logits = p.forward(validation, false);
    double      loss   = sparse_softmax_cross_entropy(logits, validation_labels).item();
    if (!std::isfinite(loss))
      throw std::runtime_error("nonfinite MobileNet validation loss");
    auto        predictions = argmax(logits, -1).to_vector<std::int64_t>();
    std::size_t correct     = 0;
    for (std::size_t i = 0; i < truth.size(); ++i)
      correct += predictions[i] == truth[i];
    result.history.push_back({epoch, total / count, loss, double(correct) / truth.size()});
    if (loss < best) {
      best                  = loss;
      result.selected_epoch = epoch;
      best_weights          = p.parameters.clone();
      best_moments          = p.moments.clone();
    }
    if (options.progress)
      options.progress(result.history.back());
  }
  p.parameters.copy_from(best_weights);
  p.moments.copy_from(best_moments);
  return result;
}
void MobileNetClassifier::save(const std::string& path) const {
  const auto&  p = *impl_;
  NamedTensors values;
  values.emplace_back(
      "model/config",
      from_vector(std::vector<double>{1, double(p.classes), double(p.side), p.width}));
  append_store(values, p.parameters, "weights/");
  append_store(values, p.moments, "moments/");
  save_npz(path, values);
}
MobileNetClassifier MobileNetClassifier::load(const std::string& path) {
  auto                  data   = load_npz(path);
  const Tensor*         config = nullptr;
  std::set<std::string> names;
  for (const auto& [name, tensor] : data) {
    if (!names.insert(name).second)
      throw std::runtime_error("duplicate MobileNet checkpoint tensor");
    if (name == "model/config")
      config = &tensor;
    else if (name.rfind("weights/", 0) != 0 && name.rfind("moments/", 0) != 0)
      throw std::runtime_error("unknown MobileNet checkpoint entry");
  }
  if (!config || config->dtype() != DType::Float64 || config->shape() != Shape{4})
    throw std::runtime_error("invalid MobileNet checkpoint configuration");
  auto c = config->to_vector<double>();
  if (c[0] != 1 || !std::isfinite(c[1]) || !std::isfinite(c[2]) || c[1] < 2 || c[1] > 1000 ||
      c[2] < 16 || c[2] > 256 || std::floor(c[1]) != c[1] || std::floor(c[2]) != c[2])
    throw std::runtime_error("invalid MobileNet checkpoint architecture");
  try {
    MobileNetClassifier model{int(c[1]), int(c[2]), c[3]};
    restore_store(model.impl_->parameters, data, "weights/");
    restore_store(model.impl_->moments, data, "moments/");
    return model;
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error(std::string("invalid MobileNet checkpoint: ") + e.what());
  }
}
} // namespace datamunge::nn
