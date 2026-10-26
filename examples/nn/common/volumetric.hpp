// 3D U-Net (Cicek et al. 2016, three cropped skips) and residual PReLU V-Net
// (Milletari et al. 2016, four skips), volumetric losses, synthetic
// ellipsoid volumes and short checkpointed training with slice previews.
// A port of tflow's common/nn/volumetric.h and common/ml/volumetric.h.
//   https://arxiv.org/abs/1606.06650
//   https://arxiv.org/abs/1606.04797
//
// TF -> datamunge: models are functions of nn::ParameterStore (plus a
// separate store for batch-norm moving statistics); Conv3DLayer is the
// nn::conv3d layer, the learned 2x2x2 up-convolution (Conv3DBackpropInputV2)
// is the nn::conv3d_transpose layer. TF needed custom Conv3D gradient
// registrations; datamunge's 3-D convolutions are differentiable natively.
// A TF graph runs BN moving-average updates only with the training step, so
// here `update_stats` is passed explicitly and loss-only evaluations skip it.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/image_io.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace volumetric {

namespace nn = datamunge::nn;
using nn::Padding;
using nn::Tensor;

// ---- layers and models (tflow common/nn/volumetric.h) ----------------------------------

/// Learned 2x2x2 stride-2 up-convolution "<name>/W" [2, 2, 2, out, in] (He uniform), optional bias.
inline Tensor VolumeTranspose(nn::ParameterStore& p, const std::string& name, const Tensor& x, std::int64_t out,
                              bool bias = false) {
  nn::ConvLayerOptions o;
  o.stride = 2, o.padding = Padding::Valid, o.bias = bias, o.init = nn::init::he_uniform();
  return nn::conv3d_transpose(p, name, x, out, 2, o);
}

/// Floor-centred crop of `source` [N, D, H, W, C] to the spatial size of `reference`.
inline Tensor VolumeCrop(const Tensor& source, const Tensor& reference) {
  const auto& a = source.shape();
  const auto& b = reference.shape();
  return nn::slice(source, {0, (a[1] - b[1]) / 2, (a[2] - b[2]) / 2, (a[3] - b[3]) / 2, 0}, {a[0], b[1], b[2], b[3], a[4]});
}

/// Output side of the 3D U-Net for an input side (VALID: 4 voxels lost per double convolution).
inline int Unet3DOutputSide(int side, bool valid = true) {
  for (int i = 0; i < 3; ++i) {
    if (valid) side -= 4;
    if (side < 2) throw std::invalid_argument("3D U-Net input too small");
    side /= 2;
  }
  if (valid) side -= 4;
  for (int i = 0; i < 3; ++i) {
    side *= 2;
    if (valid) side -= 4;
    if (side < 1) throw std::invalid_argument("3D U-Net input too small");
  }
  return side;
}

struct Unet3DConfig {
  int  input_channels = 3, classes = 3, base = 32;
  bool valid = true, batch_norm = true, batch_stats_at_test = true;
};

struct VolumeOutputs {
  Tensor              logits;
  std::vector<Tensor> encoder, decoder;
};

/// 3D U-Net. `training` selects batch statistics (as does batch_stats_at_test);
/// `update_stats` additionally updates the moving averages in `state`.
inline VolumeOutputs Unet3D(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& image,
                            const Unet3DConfig& cfg = {}, bool training = false, bool update_stats = false) {
  if (cfg.input_channels < 1 || cfg.classes < 2 || cfg.base < 1) throw std::invalid_argument("invalid 3D U-Net widths");
  VolumeOutputs        out;
  nn::ConvLayerOptions conv;
  conv.padding = cfg.valid ? Padding::Valid : Padding::Same;
  conv.bias    = !cfg.batch_norm;
  conv.init    = nn::init::he_uniform();
  const auto layer = [&](const std::string& name, const Tensor& x, std::int64_t channels) {
    Tensor h = nn::conv3d(p, name, x, channels, 3, conv);
    if (cfg.batch_norm) {
      nn::BatchNormOptions bn;
      bn.momentum     = .9;
      bn.eps          = 1e-5;
      bn.update_stats = training && update_stats;
      h = nn::batch_norm(p, state, name + "/bn", h, training || cfg.batch_stats_at_test, bn);
    }
    return nn::relu(h);
  };
  Tensor       h        = image;
  std::int64_t channels = cfg.input_channels;
  for (int i = 0; i < 4; ++i) {
    const std::int64_t first = std::int64_t{cfg.base} << i, second = 2 * first;
    h        = layer("encoder" + std::to_string(i) + "/first", h, first);
    h        = layer("encoder" + std::to_string(i) + "/second", h, second);
    channels = second;
    out.encoder.push_back(h);
    if (i < 3) h = nn::max_pool3d(h, 2, 2, Padding::Valid);
  }
  for (int i = 2; i >= 0; --i) {
    h                       = VolumeTranspose(p, "decoder" + std::to_string(i) + "/up", h, channels);
    const std::int64_t skip = std::int64_t{cfg.base} << (i + 1);
    h                       = nn::concat({VolumeCrop(out.encoder[static_cast<std::size_t>(i)], h), h}, 4);
    h                       = layer("decoder" + std::to_string(i) + "/first", h, skip);
    h                       = layer("decoder" + std::to_string(i) + "/second", h, skip);
    channels                = skip;
    out.decoder.push_back(h);
  }
  out.logits = nn::conv3d(p, "score", h, cfg.classes, 1);
  return out;
}

/// Per-channel PReLU with slopes "<name>/alpha" (initially 0.25); slope 1 at x = 0.
inline Tensor VolumePRelu(nn::ParameterStore& p, const std::string& name, const Tensor& x) {
  const Tensor alpha = p.get(name + "/alpha", {x.shape().back()}, nn::init::constant(.25));
  return nn::where(nn::greater_equal(x, 0.0), x, x * alpha);
}

/// `layers` 5x5x5 convolutions with PReLU, the input added before the last activation.
inline Tensor VolumeResidual(nn::ParameterStore& p, const std::string& name, const Tensor& input, std::int64_t channels,
                             int layers) {
  if (channels < 1 || layers < 1) throw std::invalid_argument("positive volume residual dimensions required");
  nn::ConvLayerOptions conv;
  conv.init = nn::init::he_uniform();
  Tensor h  = input;
  for (int i = 0; i < layers; ++i) {
    h = nn::conv3d(p, name + "/conv" + std::to_string(i), h, channels, 5, conv);
    if (i == layers - 1) h = h + input;
    h = VolumePRelu(p, name + "/act" + std::to_string(i), h);
  }
  return h;
}

struct VNetConfig {
  int base = 16, classes = 2;
};

inline VolumeOutputs VNet(nn::ParameterStore& p, const Tensor& image, const VNetConfig& cfg = {}) {
  if (cfg.base < 1 || cfg.classes < 2) throw std::invalid_argument("invalid V-Net widths");
  VolumeOutputs        out;
  nn::ConvLayerOptions conv;
  conv.init = nn::init::he_uniform();
  Tensor h  = nn::conv3d(p, "input", image, cfg.base, 5, conv);
  h         = VolumePRelu(p, "input/act", h + nn::tile(image, {1, 1, 1, 1, cfg.base}));
  out.encoder.push_back(h);
  const int    counts[] = {2, 3, 3, 3};
  std::int64_t channels = cfg.base;
  for (int i = 0; i < 4; ++i) {
    nn::ConvLayerOptions down = conv;
    down.stride = 2, down.padding = Padding::Valid;
    const std::int64_t width = 2 * channels;
    h        = nn::conv3d(p, "down" + std::to_string(i), h, width, 2, down);
    h        = VolumePRelu(p, "down" + std::to_string(i) + "/act", h);
    h        = VolumeResidual(p, "encoder" + std::to_string(i + 1), h, width, counts[i]);
    channels = width;
    out.encoder.push_back(h);
  }
  const int decoder_layers[] = {3, 3, 2, 1};
  for (int i = 3; i >= 0; --i) {
    const std::int64_t skip = std::int64_t{cfg.base} << i;
    h        = VolumeTranspose(p, "up" + std::to_string(i), h, skip, true);
    h        = VolumePRelu(p, "up" + std::to_string(i) + "/act", h);
    h        = nn::concat({h, out.encoder[static_cast<std::size_t>(i)]}, 4);
    channels = 2 * skip;
    h        = VolumeResidual(p, "decoder" + std::to_string(i), h, channels, decoder_layers[3 - i]);
    out.decoder.push_back(h);
  }
  out.logits = nn::conv3d(p, "score", h, cfg.classes, 1);
  return out;
}

// ---- losses (tflow common/ml/volumetric.h) ------------------------------------------------

/// Class-weighted mean softmax CE over voxels whose label is in [0, classes);
/// other labels (255) are unannotated and get weight 0.
inline Tensor SparseVolumeCE(const Tensor& logits, const Tensor& labels, int classes, const Tensor& class_weights) {
  const Tensor truth  = nn::reshape(labels, {-1});
  const Tensor valid  = nn::cast(nn::logical_and(nn::greater_equal(truth, 0.0), nn::less(truth, static_cast<double>(classes))),
                                 nn::DType::Float32);
  const Tensor target = nn::one_hot(nn::clip(truth, 0, classes - 1), classes);
  const Tensor weights = nn::sum(target * class_weights, {1}) * valid;
  const Tensor ce      = -nn::sum(nn::log_softmax(nn::reshape(logits, {-1, classes})) * target, {1});
  return nn::sum(ce * weights) / nn::maximum(nn::sum(weights), nn::scalar(1e-6));
}

/// V-Net's squared-denominator Dice loss of foreground probabilities [N, ...]
/// (labels 0/1; others ignored), averaged over volumes that have labels.
inline Tensor SquaredVolumeDice(const Tensor& foreground, const Tensor& labels) {
  const Tensor valid =
      nn::cast(nn::logical_and(nn::greater_equal(labels, 0.0), nn::less(labels, 2.0)), nn::DType::Float32);
  const Tensor prediction  = foreground * valid;
  const Tensor truth       = nn::cast(nn::equal(labels, 1.0), nn::DType::Float32);
  const Tensor numerator   = nn::sum(prediction * truth, {1, 2, 3}) * 2.0 + 1e-6;
  const Tensor denominator = nn::sum(nn::square(prediction) + nn::square(truth), {1, 2, 3}) + 1e-6;
  const Tensor has         = nn::cast(nn::greater(nn::sum(valid, {1, 2, 3}), 0.0), nn::DType::Float32);
  return nn::sum((1.0 - numerator / denominator) * has) / nn::maximum(nn::sum(has), nn::scalar(1.0));
}

// ---- data -----------------------------------------------------------------------------

struct VolumeData {
  int                count = 0, side = 16;
  std::vector<float> pixels;
  std::vector<int>   labels;
};

/// Noisy oblique ellipsoids (foreground .8, background -.8, noise .12, a weak x ramp).
inline VolumeData EllipsoidVolumes(int count, int side, int seed) {
  if (count < 1 || side < 16 || side % 16) throw std::invalid_argument("positive volume count and side divisible by16 required");
  VolumeData data;
  data.count       = count;
  data.side        = side;
  const int voxels = side * side * side;
  data.pixels.resize(std::size_t(count) * voxels);
  data.labels.resize(std::size_t(count) * voxels);
  std::mt19937                          random(static_cast<std::mt19937::result_type>(seed));
  std::uniform_real_distribution<float> center(.38f * side, .62f * side), radius(.16f * side, .28f * side), unit(0, 1);
  std::normal_distribution<float>       noise(0, .12f);
  for (int n = 0; n < count; ++n) {
    float c[3], r[3];
    for (int k = 0; k < 3; ++k) {
      c[k] = center(random);
      r[k] = radius(random);
    }
    const float tilt = (unit(random) - .5f) * .4f;
    for (int z = 0; z < side; ++z)
      for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
          const float dz = (z + .5f - c[0]) / r[0], dy = (y + .5f - c[1]) / r[1], dx = (x + .5f - c[2]) / r[2] - tilt * dz;
          const int    foreground = dz * dz + dy * dy + dx * dx <= 1;
          const size_t i          = size_t(n) * voxels + (z * side + y) * side + x;
          data.labels[i]          = foreground;
          data.pixels[i]          = (foreground ? .8f : -.8f) + noise(random) + .05f * (float(x) / side - .5f);
        }
  }
  return data;
}

/// Images [n, s, s, s, 1] and int32 labels [n, s, s, s]; `sparse` keeps only axial slices z % 4 == 2 (others 255).
inline std::pair<Tensor, Tensor> VolumeBatch(const VolumeData& data, const std::vector<int>& rows, bool sparse = false) {
  const int          side = data.side, voxels = side * side * side;
  const auto         n = static_cast<std::int64_t>(rows.size());
  std::vector<float> images(rows.size() * voxels);
  std::vector<int>   labels(rows.size() * voxels);
  for (size_t i = 0; i < rows.size(); ++i) {
    if (rows[i] < 0 || rows[i] >= data.count) throw std::out_of_range("volume batch row");
    std::copy_n(data.pixels.data() + size_t(rows[i]) * voxels, voxels, images.data() + i * voxels);
    for (int j = 0; j < voxels; ++j)
      labels[i * voxels + j] = sparse && (j / (side * side)) % 4 != 2 ? 255 : data.labels[size_t(rows[i]) * voxels + j];
  }
  return {nn::from_vector(images, {n, side, side, side, 1}), nn::from_vector(labels, {n, side, side, side})};
}

struct VolumeMetrics {
  double       dice = 0, iou = 0, accuracy = 0;
  std::int64_t tp = 0, fp = 0, fn = 0, tn = 0;
};

using LogitsFn = std::function<Tensor(const Tensor& images)>;

/// Dense held-out scoring, one volume at a time: mean per-volume foreground Dice and IoU, voxel accuracy.
inline VolumeMetrics ScoreVolumes(const LogitsFn& logits_of, const VolumeData& data) {
  nn::NoGradGuard guard;
  VolumeMetrics   total;
  const int       voxels = data.side * data.side * data.side;
  for (int row = 0; row < data.count; ++row) {
    const auto   batch  = VolumeBatch(data, {row});
    const Tensor logits = logits_of(batch.first);
    const auto   predicted = nn::argmax(logits, 4).to_vector<std::int64_t>();
    const auto   probabilities = nn::softmax(logits, 4).to_vector<float>();
    const auto   truth = batch.second.to_vector<int>();
    std::int64_t tp = 0, fp = 0, fn = 0, tn = 0;
    for (int i = 0; i < voxels; ++i) {
      const auto p = predicted[static_cast<std::size_t>(i)];
      const int  t = truth[static_cast<std::size_t>(i)];
      check::Require(p == 0 || p == 1, "binary volume predictions");
      double sum = 0;
      for (int c = 0; c < 2; ++c) {
        const double value = probabilities[static_cast<std::size_t>(2 * i + c)];
        check::Require(std::isfinite(value) && value >= 0 && value <= 1, "finite volume probabilities");
        sum += value;
      }
      check::Near(sum, 1, 1e-6, "voxelwise probability normalization");
      tp += p && t;
      fp += p && !t;
      fn += !p && t;
      tn += !p && !t;
    }
    total.dice += (2 * tp + fp + fn) ? double(2 * tp) / double(2 * tp + fp + fn) : 1;
    total.iou += (tp + fp + fn) ? double(tp) / double(tp + fp + fn) : 1;
    total.tp += tp, total.fp += fp, total.fn += fn, total.tn += tn;
  }
  total.dice /= data.count;
  total.iou /= data.count;
  total.accuracy = double(total.tp + total.tn) / (std::int64_t(data.count) * voxels);
  return total;
}

/// Axial / coronal / sagittal centre slices of the first two volumes: input, ground truth, prediction.
inline void VolumePreview(const LogitsFn& logits_of, const VolumeData& data, const std::filesystem::path& filename) {
  nn::NoGradGuard guard;
  const int side = data.side, count = std::min(2, data.count), scale = 4;
  const int rows = count * 3 * side * scale, cols = 3 * side * scale;
  std::vector<float> rgb(static_cast<std::size_t>(rows) * cols * 3, 0.f);
  for (int n = 0; n < count; ++n) {
    const auto batch = VolumeBatch(data, {n});
    const auto image = batch.first.to_vector<float>();
    const auto truth = batch.second.to_vector<int>();
    const auto pred  = nn::argmax(logits_of(batch.first), 4).to_vector<std::int64_t>();
    for (int plane = 0; plane < 3; ++plane)
      for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
          const int z = plane == 0 ? side / 2 : y, yy = plane == 1 ? side / 2 : plane == 0 ? y : x,
                    xx = plane == 2 ? side / 2 : x, index = (z * side + yy) * side + xx;
          const float intensity = std::floor(std::clamp((image[static_cast<std::size_t>(index)] + 1.2f) / 2.4f * 255, 0.f, 255.f));
          const int   t = truth[static_cast<std::size_t>(index)];
          const auto  p = pred[static_cast<std::size_t>(index)];
          for (int a = 0; a < scale; ++a)
            for (int b = 0; b < scale; ++b)
              for (int c = 0; c < 3; ++c) {
                const int   row = ((n * 3 + plane) * side + y) * scale + a, col = x * scale + b;
                const auto at = [&](int column) { return (static_cast<std::size_t>(row) * cols + column) * 3 + c; };
                rgb[at(col)]                     = intensity;
                rgb[at(side * scale + col)]      = t ? (c == 1 ? 220.f : 60.f) : 20.f;
                rgb[at(2 * side * scale + col)]  = p ? (c == 1 ? 220.f : 60.f) : 20.f;
              }
        }
  }
  nn::save_png(nn::from_vector(rgb, {rows, cols, 3}), filename.string(), 0, 255);
}

// ---- training ------------------------------------------------------------------------------

}  // namespace volumetric
