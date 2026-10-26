// Explicit NHWC image-to-column extraction and its additive adjoint col2im:
// a port of tflow's common/nn/im2col.h onto float32 nn::Tensor.
// Each row is one output location; columns order kernel row, kernel column,
// input channel. SAME puts an odd extra pad at the bottom/right, as TensorFlow
// does. Kernels perform cross-correlation (no spatial reversal). Rectangular
// kernels, separate spatial strides and dilation are supported.
// Im2Col materializes overlapping patches; it trades memory for GEMM reuse.
// (nn::conv2d does the same internally; this explicit version is for teaching.)
#pragma once

#include <algorithm>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace im2col {

namespace nn = datamunge::nn;

struct PatchGeometry {
  std::int64_t n, h, w, c, kh, kw, sh, sw, dh, dw, oh, ow, top, left;
  PatchGeometry(const nn::Shape& input, int kernel_h, int kernel_w, int stride_h = 1, int stride_w = 1,
                int dilation_h = 1, int dilation_w = 1, const std::string& padding = "VALID") {
    std::int64_t elements = 1;
    for (auto d : input) elements *= d;
    if (input.size() != 4 || elements < 1 || kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 || stride_w <= 0 ||
        dilation_h <= 0 || dilation_w <= 0 || (padding != "VALID" && padding != "SAME"))
      throw std::invalid_argument("invalid patch geometry");
    n  = input[0];
    h  = input[1];
    w  = input[2];
    c  = input[3];
    kh = kernel_h;
    kw = kernel_w;
    sh = stride_h;
    sw = stride_w;
    dh = dilation_h;
    dw = dilation_w;
    const std::int64_t effective_h = (kh - 1) * dh + 1, effective_w = (kw - 1) * dw + 1;
    oh = padding == "SAME" ? (h + sh - 1) / sh : (h >= effective_h ? (h - effective_h) / sh + 1 : 0);
    ow = padding == "SAME" ? (w + sw - 1) / sw : (w >= effective_w ? (w - effective_w) / sw + 1 : 0);
    if (!oh || !ow) throw std::invalid_argument("patch geometry has no output positions");
    top  = padding == "SAME" ? std::max(std::int64_t{0}, (oh - 1) * sh + effective_h - h) / 2 : 0;
    left = padding == "SAME" ? std::max(std::int64_t{0}, (ow - 1) * sw + effective_w - w) / 2 : 0;
  }
  std::int64_t rows() const { return n * oh * ow; }
  std::int64_t columns() const { return kh * kw * c; }
  nn::Shape    InputShape() const { return {n, h, w, c}; }
};

/// [N,H,W,C] float32 -> [rows, columns] patch matrix.
inline nn::Tensor Im2Col(const nn::Tensor& input, const PatchGeometry& g) {
  if (input.dtype() != nn::DType::Float32 || input.shape() != g.InputShape())
    throw std::invalid_argument("Im2Col requires matching float32 NHWC input");
  const auto         x = input.contiguous().to_vector<float>();
  std::vector<float> out(static_cast<std::size_t>(g.rows() * g.columns()), 0.f);
  for (std::int64_t n = 0; n < g.n; ++n)
    for (std::int64_t y = 0; y < g.oh; ++y)
      for (std::int64_t xo = 0; xo < g.ow; ++xo)
        for (std::int64_t ky = 0; ky < g.kh; ++ky)
          for (std::int64_t kx = 0; kx < g.kw; ++kx) {
            const std::int64_t iy = y * g.sh + ky * g.dh - g.top, ix = xo * g.sw + kx * g.dw - g.left;
            if (iy < 0 || iy >= g.h || ix < 0 || ix >= g.w) continue;
            for (std::int64_t c = 0; c < g.c; ++c)
              out[static_cast<std::size_t>(((n * g.oh + y) * g.ow + xo) * g.columns() + (ky * g.kw + kx) * g.c + c)] =
                  x[static_cast<std::size_t>(((n * g.h + iy) * g.w + ix) * g.c + c)];
          }
  return nn::from_vector(out, {g.rows(), g.columns()});
}

/// Adjoint of Im2Col: scatter-adds each patch entry back to its pixel.
inline nn::Tensor Col2Im(const nn::Tensor& columns, const PatchGeometry& g) {
  if (columns.dtype() != nn::DType::Float32 || columns.shape() != nn::Shape({g.rows(), g.columns()}))
    throw std::invalid_argument("Col2Im requires matching float32 patch matrix");
  const auto         in = columns.contiguous().to_vector<float>();
  std::vector<float> out(static_cast<std::size_t>(g.n * g.h * g.w * g.c), 0.f);
  for (std::int64_t n = 0; n < g.n; ++n)
    for (std::int64_t y = 0; y < g.oh; ++y)
      for (std::int64_t xo = 0; xo < g.ow; ++xo)
        for (std::int64_t ky = 0; ky < g.kh; ++ky)
          for (std::int64_t kx = 0; kx < g.kw; ++kx) {
            const std::int64_t iy = y * g.sh + ky * g.dh - g.top, ix = xo * g.sw + kx * g.dw - g.left;
            if (iy < 0 || iy >= g.h || ix < 0 || ix >= g.w) continue;
            for (std::int64_t c = 0; c < g.c; ++c)
              out[static_cast<std::size_t>(((n * g.h + iy) * g.w + ix) * g.c + c)] +=
                  in[static_cast<std::size_t>(((n * g.oh + y) * g.ow + xo) * g.columns() + (ky * g.kw + kx) * g.c + c)];
          }
  return nn::from_vector(out, g.InputShape());
}

}  // namespace im2col
