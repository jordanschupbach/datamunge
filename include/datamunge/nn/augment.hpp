#pragma once

/// @file augment.hpp
/// @brief Data augmentation for float NHWC image batches.
///
/// Every function takes images [N, H, W, C] (or one image [H, W, C], returned
/// with the same rank), returns a new tensor and records no autograd history:
/// augmentation is input preprocessing. Integer inputs are converted to
/// float32. Random helpers take an explicit `Generator`, so a run is
/// reproducible from its seed, and draw independent parameters per image
/// unless stated otherwise.
///
/// The deterministic helpers mirror tflow's `data::augment` (batch-shared
/// parameters) and the TF image ops of the same names. Colour ops expect RGB
/// in [0, 1] where that matters (saturation, hue, `color_jitter`'s clamp).

#include <cstdint>
#include <vector>

#include "datamunge/nn/random.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn::augment {

/// A pixel rectangle. It may extend past the image; only the overlap counts.
struct Rect {
  std::int64_t top{0};
  std::int64_t left{0};
  std::int64_t height{0};
  std::int64_t width{0};
};

/// Result of mixup / CutMix. `weight` is the share of each original image
/// (and its label) kept; the partner image contributes 1 - weight.
struct MixedBatch {
  Tensor images;
  Tensor labels;
  double weight{1.0};
};

// ---- deterministic transforms ------------------------------------------------------

/// Mirrors the W axis (TF flip_left_right).
Tensor flip_left_right(const Tensor& images);
/// Mirrors the H axis (TF flip_up_down).
Tensor flip_up_down(const Tensor& images);
/// images[:, top:top+height, left:left+width, :]; throws if the window leaves the image.
Tensor crop(const Tensor& images, std::int64_t top, std::int64_t left, std::int64_t height, std::int64_t width);
/// Sets the pixels inside `rect` (clipped to the image) to `fill`, for every image.
Tensor cutout(const Tensor& images, const Rect& rect, double fill = 0.0);
/// images * weight + images[permutation] * (1 - weight), and the same for the
/// float [N, K] `labels` (one-hot or soft; convert class indices with one_hot).
MixedBatch mixup(const Tensor& images, const Tensor& labels, const std::vector<std::int64_t>& permutation,
                 double weight);
/// Pastes `rect` from images[permutation] into every image. The label weight
/// is the fraction of pixels kept, measured after clipping (tflow Cutmix).
MixedBatch cutmix(const Tensor& images, const Tensor& labels, const std::vector<std::int64_t>& permutation,
                  const Rect& rect);

/// images + delta (TF adjust_brightness; no clamping).
Tensor adjust_brightness(const Tensor& images, double delta);
/// (x - mean) * factor + mean with the mean per image and channel over H and
/// W (TF adjust_contrast / AdjustContrastv2).
Tensor adjust_contrast(const Tensor& images, double factor);
/// Scales HSV saturation by `factor`, clamped to [0, 1] (TF adjust_saturation; C must be 3).
Tensor adjust_saturation(const Tensor& images, double factor);
/// Rotates hue by `delta` turns, delta in [-1, 1] (TF adjust_hue; C must be 3).
Tensor adjust_hue(const Tensor& images, double delta);
/// Brightness, contrast, saturation and hue in that order, then clamps to
/// [0, 1] (tflow ColorJitter). Identity at (0, 1, 1, 0) apart from the clamp.
Tensor color_jitter(const Tensor& images, double brightness, double contrast, double saturation, double hue);
/// (x - mean) / max(stddev, 1 / sqrt(H*W*C)) per image (TF per_image_standardization).
Tensor per_image_standardization(const Tensor& images);

// ---- random transforms -------------------------------------------------------------

/// Flips each image left-right with probability `p`.
Tensor random_flip_left_right(const Tensor& images, Generator& gen, double p = 0.5);
/// Flips each image up-down with probability `p`.
Tensor random_flip_up_down(const Tensor& images, Generator& gen, double p = 0.5);
/// tflow `ImageBatch` training augmentation: pads by `pad` pixels of `fill`
/// on every side and crops back to H x W at a uniform offset, then (with
/// `flip`) mirrors left-right with probability 1/2. Per image it draws the
/// row shift, the column shift and then the coin, in that order. CIFAR
/// training uses pad = 4, flip = true. tflow pads with 0 in uint8 units,
/// i.e. black; apply this to [0, 1] images (fill 0) before rescaling.
Tensor random_pad_crop(const Tensor& images, std::int64_t pad, Generator& gen, bool flip = false, double fill = 0.0);
/// A uniformly placed height x width window of each image.
Tensor random_crop(const Tensor& images, std::int64_t height, std::int64_t width, Generator& gen);
/// Cutout (DeVries and Taylor 2017): a size x size square centred at a
/// uniform pixel, clipped at the borders, set to `fill` in each image.
Tensor random_cutout(const Tensor& images, std::int64_t size, Generator& gen, double fill = 0.0);
/// Adds a uniform delta in [-max_delta, max_delta] to each image.
Tensor random_brightness(const Tensor& images, double max_delta, Generator& gen);
/// adjust_contrast with a factor uniform in [lower, upper) per image.
Tensor random_contrast(const Tensor& images, double lower, double upper, Generator& gen);
/// adjust_saturation with a factor uniform in [lower, upper) per image.
Tensor random_saturation(const Tensor& images, double lower, double upper, Generator& gen);
/// adjust_hue with a delta uniform in [-max_delta, max_delta] per image.
Tensor random_hue(const Tensor& images, double max_delta, Generator& gen);

/// Beta(alpha, alpha) draw used as a mixing weight (tflow BetaMixWeight).
double beta_mix_weight(Generator& gen, double alpha);
/// mixup (Zhang et al. 2018) with a random batch permutation and a
/// Beta(alpha, alpha) weight shared by the batch.
MixedBatch random_mixup(const Tensor& images, const Tensor& labels, double alpha, Generator& gen);
/// CutMix (Yun et al. 2019): weight lambda ~ Beta(alpha, alpha), a box of
/// H*sqrt(1-lambda) x W*sqrt(1-lambda) centred at a uniform pixel, a random
/// partner permutation; labels use the clipped area. Shared by the batch.
MixedBatch random_cutmix(const Tensor& images, const Tensor& labels, double alpha, Generator& gen);

}  // namespace datamunge::nn::augment
