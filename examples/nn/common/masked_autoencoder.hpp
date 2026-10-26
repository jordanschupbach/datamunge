// Port of tflow's common/ml/masked_autoencoder.h + common/nn/masked_autoencoder.h:
// a tiny asymmetric masked autoencoder (He et al. MAE) on 16 non-overlapping 7x7
// MNIST patches: a visible-only width-32 transformer encoder (4 of 16 tokens), a
// width-16 mask-token decoder, fixed 2-D sin/cos positions and a hidden-only loss.
// Parameter names match tflow's ("patch/W", "encoder/attention/Wq", "mask_token", ...),
// so its initial_arrays.npz loads directly. Training steps (forward, MaeLoss,
// nn::grad / optimizer) are written out in the example.
#pragma once

#include "common/autoencoder.hpp"

namespace ae {

/// Fixed 2-D sin/cos positions [16, width]: first half encodes the patch row, second the column.
inline nn::Tensor MaePosition(int width) {
  std::vector<float> out(static_cast<std::size_t>(16 * width));
  for (int patch = 0; patch < 16; ++patch)
    for (int axis = 0; axis < 2; ++axis)
      for (int k = 0; k < width / 4; ++k) {
        const double a = (axis ? patch % 4 : patch / 4) * std::pow(10000., -2. * k / (width / 2));
        out[patch * width + axis * width / 2 + k]             = float(std::sin(a));
        out[patch * width + axis * width / 2 + width / 4 + k] = float(std::cos(a));
      }
  return nn::from_vector(out, {16, width});
}

struct MaeBlockOutput {
  nn::Tensor output, attention;
};

/// Pre-norm transformer block: x + MHA(LN x) (2 heads, no biases), then + FF(LN .) with
/// ratio 2 and tanh-GELU.
inline MaeBlockOutput MaeBlock(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, int width) {
  auto                 normalized = nn::layer_norm(p, name + "/ln1", x);
  nn::MultiHeadOptions options;
  options.need_weights = true;
  auto attended        = nn::multi_head_attention(p, name + "/attention", normalized, normalized, 2, options);
  auto residual        = x + attended.output;
  auto ff = nn::dense(p, name + "/ff1", nn::layer_norm(p, name + "/ln2", residual), 2 * width, nn::Activation::GeluTanh);
  return {residual + nn::dense(p, name + "/ff2", ff, width), attended.weights};
}

struct MaskedAutoencoderOutput {
  nn::Tensor patches, visible, mask, code, prediction, encoder_attention, decoder_attention;
};

/// [B, 28, 28, 1] -> [B, 16, 49] row-major 7x7 patches.
inline nn::Tensor MaePatches(const nn::Tensor& image) {
  const auto B = image.size(0);
  return nn::reshape(nn::transpose(nn::reshape(image, {B, 4, 7, 4, 7}), {0, 1, 3, 2, 4}), {B, 16, 49});
}

/// Encoder half. selector [B, 4, 16] is a one-hot gather/permutation: hidden pixels never enter the encoder.
inline void MaeEncode(nn::ParameterStore& p, const nn::Tensor& image, const nn::Tensor& selector,
                      MaskedAutoencoderOutput& out) {
  out.patches       = MaePatches(image);
  out.visible       = nn::matmul(selector, out.patches);
  out.mask          = 1.0 - nn::sum(selector, {1});
  auto embedded     = nn::dense(p, "patch", out.visible, 32);
  auto position     = nn::matmul(selector, nn::expand_dims(MaePosition(32), 0));
  auto encoder      = MaeBlock(p, "encoder", embedded + position, 32);
  out.code          = nn::layer_norm(p, "encoder/norm", encoder.output);
  out.encoder_attention = encoder.attention;
}

/// Decoder half: from the code and patch locations only (no image/target).
inline void MaeDecode(nn::ParameterStore& p, const nn::Tensor& code, const nn::Tensor& selector,
                      MaskedAutoencoderOutput& out) {
  auto mask      = 1.0 - nn::sum(selector, {1});
  auto projected = nn::dense(p, "decoder_projection", code, 16);
  auto restored  = nn::matmul(selector, projected, true);
  auto token     = p.get("mask_token", {16}, nn::init::zeros());
  auto decoder_input = restored + nn::expand_dims(mask, 2) * token + MaePosition(16);
  auto decoder       = MaeBlock(p, "decoder", decoder_input, 16);
  out.decoder_attention = decoder.attention;
  out.prediction = nn::dense(p, "prediction", nn::layer_norm(p, "decoder/norm", decoder.output), 49,
                             nn::Activation::Sigmoid);
}

inline MaskedAutoencoderOutput MaskedAutoencoder(nn::ParameterStore& p, const nn::Tensor& image,
                                                 const nn::Tensor& selector) {
  MaskedAutoencoderOutput out;
  MaeEncode(p, image, selector, out);
  MaeDecode(p, out.code, selector, out);
  return out;
}

/// Squared error over hidden patches only, divided by 49 x the number of hidden patches.
inline nn::Tensor MaeLoss(const nn::Tensor& prediction, const nn::Tensor& patches, const nn::Tensor& mask) {
  auto error = nn::square(prediction - patches);
  return nn::sum(error * nn::expand_dims(mask, 2)) / (49.0 * nn::sum(mask));
}

class MaskedAutoencoderModel {
 public:
  nn::ParameterStore             parameters{1877};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit MaskedAutoencoderModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      std::vector<float> one(64, 0.f);
      for (int k = 0; k < 4; ++k) one[k * 16 + k] = 1;
      MaskedAutoencoder(parameters, nn::zeros({1, 28, 28, 1}), nn::from_vector(one, {1, 4, 16}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  nn::Tensor Loss(const nn::Tensor& images, const nn::Tensor& selection) {
    auto n = MaskedAutoencoder(parameters, images, selection);
    return MaeLoss(n.prediction, n.patches, n.mask);
  }
  /// {code [B, 4, 32], prediction [B, 16, 49], hidden-only MSE}
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images, const nn::Tensor& selection) {
    nn::NoGradGuard guard;
    auto            n = MaskedAutoencoder(parameters, images, selection);
    return {n.code, n.prediction, MaeLoss(n.prediction, n.patches, n.mask)};
  }
  nn::Tensor Decode(const nn::Tensor& codes, const nn::Tensor& selection) {
    nn::NoGradGuard         guard;
    MaskedAutoencoderOutput out;
    MaeDecode(parameters, codes, selection, out);
    return out.prediction;
  }
  /// {patches, visible, mask, encoder attention, decoder attention}
  std::vector<nn::Tensor> Probe(const nn::Tensor& images, const nn::Tensor& selection) {
    nn::NoGradGuard guard;
    auto            n = MaskedAutoencoder(parameters, images, selection);
    return {n.patches, n.visible, n.mask, n.encoder_attention, n.decoder_attention};
  }
};

inline std::uint32_t MaeMix(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}

/// Reproducible per-example 4-of-16 visible patch selection [n, 4, 16] (hash of source id, epoch, seed).
inline nn::Tensor MaeSelection(const nn::Tensor& source, int epoch = 0, std::uint32_t seed = 1907) {
  const auto         ids = source.contiguous().to_vector<int>();
  std::vector<float> out(ids.size() * 64, 0.f);
  for (std::size_t b = 0; b < ids.size(); ++b) {
    std::vector<std::pair<std::uint32_t, int>> keys;
    for (int i = 0; i < 16; ++i)
      keys.emplace_back(MaeMix(seed ^ (std::uint32_t(ids[b]) * 0x9e3779b9u) ^ (std::uint32_t(epoch) * 0x85ebca6bu) ^
                               (std::uint32_t(i) * 0xc2b2ae35u)),
                        i);
    std::sort(keys.begin(), keys.end());
    for (int k = 0; k < 4; ++k) out[b * 64 + k * 16 + keys[k].second] = 1;
  }
  return nn::from_vector(out, {std::int64_t(ids.size()), 4, 16});
}

/// Flat pixel index of element `within` of 7x7 patch `patch`.
inline int MaePixel(int patch, int within) { return ((patch / 4) * 7 + within / 7) * 28 + (patch % 4) * 7 + within % 7; }

/// [n, 16, 49] patches -> [n, 784] image.
inline nn::Tensor MaeImage(const nn::Tensor& patches) {
  const auto         n = patches.size(0);
  const auto         x = patches.contiguous().to_vector<float>();
  std::vector<float> out(static_cast<std::size_t>(n * 784));
  for (std::int64_t b = 0; b < n; ++b)
    for (int p = 0; p < 16; ++p)
      for (int i = 0; i < 49; ++i) out[b * 784 + MaePixel(p, i)] = x[(b * 16 + p) * 49 + i];
  return nn::from_vector(out, {n, 784});
}

/// Whether patch `patch` of example b is hidden (selection values as a flat [n, 4, 16] vector).
inline bool MaeHidden(const std::vector<float>& selection, std::int64_t b, int patch) {
  for (int k = 0; k < 4; ++k)
    if (selection[b * 64 + k * 16 + patch] != 0) return false;
  return true;
}

/// Independent native hidden-only MSE.
inline double MaeMse(const nn::Tensor& prediction, const nn::Tensor& original, const nn::Tensor& selection) {
  const auto  pixels = MaeImage(prediction).to_vector<float>();
  const auto  o = original.contiguous().to_vector<float>(), s = selection.contiguous().to_vector<float>();
  double      sum   = 0;
  std::size_t count = 0;
  for (std::int64_t b = 0; b < prediction.size(0); ++b)
    for (int p = 0; p < 16; ++p)
      if (MaeHidden(s, b, p))
        for (int i = 0; i < 49; ++i) {
          const auto   index = b * 784 + MaePixel(p, i);
          const double y     = pixels[index];
          check::Require(std::isfinite(y) && y >= 0 && y <= 1, "bounded finite patch prediction");
          const double e = y - o[index];
          sum += e * e;
          ++count;
        }
  check::Require(count == std::size_t(prediction.size(0)) * 12 * 49, "exact75percent hidden loss denominator");
  return sum / double(count);
}

}  // namespace ae
