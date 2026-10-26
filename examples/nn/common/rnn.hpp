// Helpers shared by the explicit-unroll recurrent cells (tflow's common/nn/rnn.h
// and the masking idiom its 4.1 cells repeat). The cells themselves are
// datamunge::nn::rnn_step / gru_step / lstm_step or are written out in the
// per-cell headers when an example needs intermediate gate values.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <string>
#include <vector>

#include "common/recurrent_model.hpp"

namespace rnn_cells {

/// Float [B, 1] mask: 1 where lengths > t (tflow: ExpandDims(Cast(Greater(lengths, t)), 1)).
inline Tensor StepMask(const Tensor& lengths, int t) {
  return nn::expand_dims(nn::cast(nn::greater(lengths, double(t)), nn::DType::Float32), 1);
}

/// Bool [B, 1] mask for SelectV2-style blends.
inline Tensor StepValid(const Tensor& lengths, int t) { return nn::expand_dims(nn::greater(lengths, double(t)), 1); }

/// mask * next + (1 - mask) * previous, exactly tflow's arithmetic masking.
inline Tensor Blend(const Tensor& mask, const Tensor& next, const Tensor& previous) {
  return nn::add(nn::mul(mask, next), nn::mul(nn::sub(1.0, mask), previous));
}

/// Dense head "<name>/W" ~ N(0, 0.1^2), "<name>/b" = 0 (tflow pre-adds head/W
/// with AddNormal(.1) before calling Dense, so the normal init wins).
inline Tensor Head(nn::ParameterStore& p, const std::string& name, const Tensor& x, std::int64_t in, std::int64_t out) {
  const Tensor W = p.get(name + "/W", {in, out}, nn::init::normal(.1));
  const Tensor b = p.get(name + "/b", {out}, nn::init::zeros());
  return nn::bias_add(nn::matmul(x, W), b);
}

/// Zeros where the [B, 1] bool mask is false (tflow: SelectV2(valid, x, ZerosLike(x))).
inline Tensor MaskedDiagnostics(const Tensor& valid, const Tensor& x) { return nn::where(valid, x, nn::zeros_like(x)); }

}  // namespace rnn_cells
