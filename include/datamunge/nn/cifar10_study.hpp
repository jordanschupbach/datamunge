#pragma once

/// @file cifar10_study.hpp
/// @brief Native data preparation and inference for the reproducible CIFAR-10 dataset study.

#include <algorithm>
#include <datamunge/datasets/download.hpp>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <numeric>

namespace datamunge::nn::study {

/// @brief Class names in the original CIFAR-10 binary label order.
inline const std::vector<std::string> cifar10_classes{
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"};

/// @brief Owned training/validation tensors, original row IDs and fitting-only channel statistics.
///
/// Images are Float32 NHWC in [0,1], labels Int64. Channel mean/SD have shape [3].
/// The official test file is never opened by prepare_cifar10().
/// @par Example
/// @code
/// const auto data = dm::nn::study::prepare_cifar10();
/// const auto count = data.fit.labels.numel(); // 45000
/// @endcode
/// @see prepare_cifar10()
struct Cifar10Study {
  ImageDataset              fit;            ///< Fitting rows, 4500 per class by default.
  ImageDataset              validation;     ///< Selection rows, 500 per class by default.
  Tensor                    mean;           ///< Fitting-only population RGB means in [0,1].
  Tensor                    sd;             ///< Fitting-only population RGB standard deviations.
  std::vector<std::int64_t> fit_ids;        ///< Original concatenated training-file row IDs.
  std::vector<std::int64_t> validation_ids; ///< Disjoint original validation row IDs.
  std::string               directory;      ///< Native shared cache holding the binary files.
};

/// @brief Create a seeded, exactly stratified split of all five official training files.
///
/// Fetches through the canonical cache; computes population channel moments only
/// on fitting images, with Float64 reductions. Does not load official test data.
/// The split uses std::mt19937_64/std::shuffle; persisted row IDs are authoritative
/// when reproducing across different standard-library shuffle implementations.
/// @param seed Split shuffle seed; default 20261030.
/// @param validation_per_class Held-out images per class, in [1,4999]; default 500.
/// @return Owned fitting/validation images, labels, IDs and channel moments.
/// @throws std::invalid_argument For an invalid holdout size or unexpected class counts.
/// @throws std::runtime_error For native acquisition, decoding or file failures.
/// @par Example
/// @code
/// const auto data = dm::nn::study::prepare_cifar10();
/// const auto rows = data.validation.labels.numel(); // 5000
/// @endcode
/// @par References
/// - Krizhevsky (2009), Learning Multiple Layers of Features from Tiny Images.
///   https://www.cs.toronto.edu/~kriz/cifar.html
/// @see standardize_cifar10()
inline Cifar10Study prepare_cifar10(
    std::uint64_t seed                 = 20261030,
    std::int64_t  validation_per_class = 500) {
  if (validation_per_class < 1 || validation_per_class >= 5000)
    throw std::invalid_argument("validation_per_class must be in [1,4999]");
  NoGradGuard  guard;
  Cifar10Study out;
  out.directory  = datamunge::datasets::fetch("cifar10");
  const auto raw = load_cifar10(out.directory, "train");
  if (raw.labels.numel() != 50000)
    throw std::invalid_argument("CIFAR-10 study requires all 50000 training images");
  const auto      labels = raw.labels.to_vector<std::int64_t>();
  std::mt19937_64 rng(seed);
  for (int cls = 0; cls < 10; ++cls) {
    std::vector<std::int64_t> ids;
    for (std::int64_t i = 0; i < 50000; ++i)
      if (labels[i] == cls)
        ids.push_back(i);
    if (ids.size() != 5000)
      throw std::invalid_argument("CIFAR-10 class count differs from 5000");
    std::shuffle(ids.begin(), ids.end(), rng);
    out.validation_ids.insert(
        out.validation_ids.end(), ids.begin(), ids.begin() + validation_per_class);
    out.fit_ids.insert(out.fit_ids.end(), ids.begin() + validation_per_class, ids.end());
  }
  out.fit        = {take_rows(raw.images, out.fit_ids), take_rows(raw.labels, out.fit_ids)};
  out.validation = {
      take_rows(raw.images, out.validation_ids), take_rows(raw.labels, out.validation_ids)};
  Tensor sum1 = zeros({3}, DType::Float64), sum2 = zeros({3}, DType::Float64);
  for (std::int64_t start = 0; start < out.fit.images.size(0); start += 500) {
    const auto x =
        slice_axis(out.fit.images, 0, start, std::min(start + 500, out.fit.images.size(0)))
            .to(DType::Float64);
    sum1 = sum1 + sum(x, {0, 1, 2});
    sum2 = sum2 + sum(square(x), {0, 1, 2});
  }
  const double pixels = out.fit.images.size(0) * 1024.;
  const auto   mean64 = sum1 / pixels;
  out.mean            = mean64.to(DType::Float32);
  out.sd              = sqrt(sum2 / pixels - square(mean64)).to(DType::Float32);
  return out;
}

/// @brief Reload a persisted study split and normalization without reshuffling or refitting.
///
/// Checks a complete, disjoint 45000/5000 partition of row IDs, exactly 4500/500
/// examples per class and finite Float32 RGB means/positive SDs. Loads only the
/// official training files. The caller verifies source-file fingerprints before use.
/// @param path NPZ with mean, sd, fit_ids and validation_ids written by the study trainer.
/// @return Owned study tensors using precisely the archived IDs and moments.
/// @throws std::invalid_argument For missing entries, invalid moments, IDs or class supports.
/// @throws std::runtime_error For NPZ, dataset acquisition or decoding failures.
/// @par Example
/// @code
/// const auto data = dm::nn::study::restore_cifar10_study("artifacts/preprocessing.npz");
/// @endcode
/// @see prepare_cifar10()
inline Cifar10Study restore_cifar10_study(const std::string& path) {
  NoGradGuard  guard;
  Cifar10Study out;
  for (const auto& [name, value] : load_npz(path)) {
    if (name == "mean")
      out.mean = value;
    if (name == "sd")
      out.sd = value;
    if (name == "fit_ids") {
      if (value.dtype() != DType::Int64 || value.shape() != Shape{45000})
        throw std::invalid_argument("invalid persisted fitting IDs");
      out.fit_ids = value.to_vector<std::int64_t>();
    }
    if (name == "validation_ids") {
      if (value.dtype() != DType::Int64 || value.shape() != Shape{5000})
        throw std::invalid_argument("invalid persisted validation IDs");
      out.validation_ids = value.to_vector<std::int64_t>();
    }
  }
  if (!out.mean.defined() || !out.sd.defined() || out.mean.shape() != Shape{3} ||
      out.sd.shape() != Shape{3} || out.mean.dtype() != DType::Float32 ||
      out.sd.dtype() != DType::Float32 || !all(is_finite(out.mean)).item<bool>() ||
      !all(is_finite(out.sd)).item<bool>() || min(out.sd).item() <= 0 ||
      out.fit_ids.size() != 45000 || out.validation_ids.size() != 5000)
    throw std::invalid_argument("invalid persisted CIFAR preprocessing");
  std::vector<bool> seen(50000, false);
  for (const auto* ids : {&out.fit_ids, &out.validation_ids})
    for (const auto id : *ids) {
      if (id < 0 || id >= 50000 || seen[id])
        throw std::invalid_argument("invalid/overlapping persisted CIFAR row ID");
      seen[id] = true;
    }
  out.directory  = datamunge::datasets::fetch("cifar10");
  const auto raw = load_cifar10(out.directory, "train");
  if (raw.labels.numel() != 50000)
    throw std::invalid_argument("all CIFAR training files required");
  out.fit        = {take_rows(raw.images, out.fit_ids), take_rows(raw.labels, out.fit_ids)};
  out.validation = {
      take_rows(raw.images, out.validation_ids), take_rows(raw.labels, out.validation_ids)};
  for (const auto* dataset : {&out.fit, &out.validation}) {
    std::vector<int> counts(10, 0);
    for (const auto label : dataset->labels.to_vector<std::int64_t>()) {
      if (label < 0 || label >= 10)
        throw std::invalid_argument("invalid CIFAR label");
      ++counts[label];
    }
    for (const auto count : counts)
      if (count != dataset->labels.numel() / 10)
        throw std::invalid_argument("unbalanced persisted CIFAR split");
  }
  return out;
}

/// @brief Apply fixed RGB standardization without fitting or changing moments.
/// @param images Float32 [N,H,W,3] images in [0,1].
/// @param mean Broadcast-compatible RGB means.
/// @param sd Broadcast-compatible positive RGB population standard deviations.
/// @return Differentiable (images-mean)/sd, preserving NHWC shape.
/// @throws std::invalid_argument For incompatible tensor shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::study::standardize_cifar10(n::zeros({1,32,32,3}), n::zeros({3}),
/// n::ones({3}));
/// @endcode
/// @see prepare_cifar10()
inline Tensor standardize_cifar10(const Tensor& images, const Tensor& mean, const Tensor& sd) {
  return (images - mean) / sd;
}

/// @brief Independently zero-pad, randomly crop and horizontally flip every training image.
///
/// Padding is four raw black pixels on each side; crops are uniformly selected
/// from the 9x9 legal offsets. Statistics are applied afterwards by the caller.
/// @param images Float32 NHWC [N,32,32,3] in [0,1].
/// @param generator Mutable stream controlling each image's crop and flip.
/// @return Owned Float32 batch with the same shape and raw value range.
/// @throws std::invalid_argument For incompatible image geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::Generator generator(151);
/// const auto x = n::study::augment_cifar10(n::zeros({2,32,32,3}), generator);
/// @endcode
/// @par References
/// - He et al. (2016), Deep Residual Learning for Image Recognition.
///   https://arxiv.org/abs/1512.03385
/// @see standardize_cifar10()
inline Tensor augment_cifar10(const Tensor& images, Generator& generator) {
  if (images.dim() != 4 || images.size(0) < 1 || images.shape() != Shape{images.size(0), 32, 32, 3})
    throw std::invalid_argument("augmentation requires nonempty NHWC CIFAR images");
  const auto          padded = pad(images, {{0, 0}, {4, 4}, {4, 4}, {0, 0}});
  std::vector<Tensor> result;
  result.reserve(images.size(0));
  for (std::int64_t i = 0; i < images.size(0); ++i) {
    const auto image = slice_axis(padded, 0, i, i + 1);
    result.push_back(
        augment::random_flip_left_right(augment::random_crop(image, 32, 32, generator), generator));
  }
  return concat(result, 0);
}

/// @brief Compact four-stage ResNet-18 configuration used in the dataset study.
/// @return Ten-class 3x3/no-pool-stem model with widths 16/32/64/128 and BN retention .9.
/// @par Example
/// @code
/// const auto config = dm::nn::study::cifar10_resnet_config();
/// const auto width_divisor = config.width_divisor; // 4
/// @endcode
/// @par References
/// - He et al. (2016), Deep Residual Learning for Image Recognition.
///   https://arxiv.org/abs/1512.03385
/// @see models::ResNet()
inline models::ResNetConfig cifar10_resnet_config() {
  models::ResNetConfig config;
  config.classes       = 10;
  config.small_stem    = true;
  config.width_divisor = 4;
  config.momentum      = .9;
  return config;
}

/// @brief Evaluate the study's compact ResNet in batches with frozen BN moments.
///
/// Missing entries are created on first use, following models::ResNet(). Later
/// calls reuse the learned leaves and moving moments without updating either.
/// @param parameters Mutable learned store, initialized or populated on first use.
/// @param state Mutable BN moment store, initialized or populated on first use.
/// @param images Raw Float32 NHWC RGB images in [0,1].
/// @param mean Fixed fitting-only RGB means.
/// @param sd Fixed fitting-only RGB population standard deviations.
/// @param batch Positive inference batch size, default 128.
/// @return Detached Float32 [N,10] raw class logits in input row order.
/// @throws std::invalid_argument For empty data, nonpositive batch or incompatible model tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto z =
/// n::study::cifar10_logits(p,state,n::zeros({1,32,32,3}),n::zeros({3}),n::ones({3}));
/// @endcode
/// @see cifar10_resnet_config()
inline Tensor cifar10_logits(
    ParameterStore& parameters,
    ParameterStore& state,
    const Tensor&   images,
    const Tensor&   mean,
    const Tensor&   sd,
    std::int64_t    batch = 128) {
  if (batch < 1 || images.dim() != 4 || images.size(0) < 1)
    throw std::invalid_argument("nonempty images and positive inference batch required");
  NoGradGuard         guard;
  std::vector<Tensor> logits;
  for (std::int64_t start = 0; start < images.size(0); start += batch) {
    const auto x = slice_axis(images, 0, start, std::min(start + batch, images.size(0)));
    logits.push_back(
        models::ResNet(
            parameters, state, standardize_cifar10(x, mean, sd), cifar10_resnet_config(), false)
            .logits);
  }
  return concat(logits, 0);
}
} // namespace datamunge::nn::study
