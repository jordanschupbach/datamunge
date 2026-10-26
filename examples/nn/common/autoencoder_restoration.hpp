// Port of tflow's common/ml/autoencoder_restoration.h: target-free denoising and
// visible-only masked inpainting using frozen three-epoch checkpoints.
#pragma once

#include "common/denoising_autoencoder.hpp"
#include "common/masked_autoencoder.hpp"

namespace ae {

/// {code, patch prediction} from the visible image + mask ONLY; no clean hidden targets/loss computed.
inline std::vector<nn::Tensor> RestoreVisible(MaskedAutoencoderModel& model, const nn::Tensor& observed,
                                              const nn::Tensor& selector) {
  nn::NoGradGuard         guard;
  MaskedAutoencoderOutput out;
  MaeEncode(model.parameters, observed, selector, out);
  MaeDecode(model.parameters, out.code, selector, out);
  return {out.code.contiguous(), out.prediction.contiguous()};
}

/// `original` with every hidden patch pixel replaced by `hidden` (same shape).
inline nn::Tensor VisibleImage(const nn::Tensor& original, const nn::Tensor& selection, float hidden = 0) {
  const auto         x = original.contiguous().to_vector<float>(), s = selection.contiguous().to_vector<float>();
  std::vector<float> out(x.size());
  for (std::int64_t b = 0; b < original.size(0); ++b)
    for (int p = 0; p < 16; ++p)
      for (int i = 0; i < 49; ++i) {
        const auto index = b * 784 + MaePixel(p, i);
        out[index]       = MaeHidden(s, b, p) ? hidden : x[index];
      }
  return nn::from_vector(out, original.shape());
}

/// [n, 784]: hidden patch pixels from `hidden_prediction`, visible pixels copied from `observed`.
inline nn::Tensor InpaintComposite(const nn::Tensor& observed, const nn::Tensor& hidden_prediction,
                                   const nn::Tensor& selection) {
  const auto o = observed.contiguous().to_vector<float>(), h = hidden_prediction.contiguous().to_vector<float>(),
             s = selection.contiguous().to_vector<float>();
  const auto         n = observed.size(0);
  std::vector<float> out(static_cast<std::size_t>(n * 784));
  for (std::int64_t b = 0; b < n; ++b)
    for (int p = 0; p < 16; ++p)
      for (int i = 0; i < 49; ++i) {
        const auto index = b * 784 + MaePixel(p, i);
        out[index]       = MaeHidden(s, b, p) ? h[index] : o[index];
      }
  return nn::from_vector(out, {n, 784});
}

/// MSE over hidden patch pixels only (denominator 588 per case).
inline double HiddenRestorationMse(const nn::Tensor& prediction, const nn::Tensor& clean, const nn::Tensor& selection) {
  const auto  p = prediction.contiguous().to_vector<float>(), c = clean.contiguous().to_vector<float>(),
             s = selection.contiguous().to_vector<float>();
  double      sum   = 0;
  std::int64_t count = 0;
  for (std::int64_t b = 0; b < clean.size(0); ++b)
    for (int q = 0; q < 16; ++q)
      if (MaeHidden(s, b, q))
        for (int i = 0; i < 49; ++i) {
          const auto   index = b * 784 + MaePixel(q, i);
          const double delta = double(p[index]) - double(c[index]);
          sum += delta * delta;
          ++count;
        }
  check::Require(count == clean.size(0) * 588, "hidden-only denominator75percent");
  return sum / double(count);
}

}  // namespace ae
