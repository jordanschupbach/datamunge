// CMD: just run-nn conv_layers
// Convolution layers: 1D, 2D, 3D, depthwise, separable, transposed and dilated.
//
// Port of tflow's conv_layers. The layers in datamunge/nn/layers.hpp
// (nn::conv1d, conv2d, depthwise_conv2d, separable_conv2d, conv2d_transpose)
// wrap the ops in datamunge/nn/conv.hpp. This program checks the output-size
// rules for VALID / SAME padding, strides and dilations against the ops,
// verifies the 1-D, 2-D, 3-D and depthwise convolutions against hand-written
// cross-correlation loops, compares the parameter counts of full and separable
// convolutions, shows that the transposed convolution is the gradient of the
// convolution with respect to its input (and the checkerboard pattern of
// overlaps it produces), and that a dilated kernel equals a larger kernel
// with holes.
//
// TF -> datamunge: graphs, placeholders and sessions disappear (layers run
// eagerly), nn::Params is nn::ParameterStore, AddSymbolicGradients is
// nn::grad, Conv2DBackpropInput is nn::conv2d_backprop_input. The library has
// no 3-D convolution yet (planned for P3), so Conv3D is built locally below
// as a sum of 2-D convolutions over the depth offsets of the kernel. Inputs
// come from the same std::mt19937_64 streams as TF's; the weights come from
// datamunge's own generator, so the printed 5x5 kernel differs from TF's
// while every |diff| check holds for any weights.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Padding;
using nn::Shape;
using nn::Tensor;

namespace {

Tensor Random(const Shape& shape, std::mt19937_64& rng) {
  std::int64_t n = 1;
  for (auto d : shape) n *= d;
  std::normal_distribution<float> normal(0.f, 1.f);
  std::vector<float>              v(static_cast<std::size_t>(n));
  for (auto& e : v) e = normal(rng);
  return nn::from_vector(v, shape);
}

std::string ShapeStr(const Tensor& t) {
  std::string s = "[";
  for (std::size_t i = 0; i < t.shape().size(); ++i) s += (i ? "," : "") + std::to_string(t.shape()[i]);
  return s + "]";
}

const char* PadName(Padding p) { return p == Padding::Valid ? "VALID" : "SAME"; }

// Padding before the data on one axis, under TensorFlow's SAME rule.
std::int64_t PadBefore(std::int64_t in, std::int64_t k, std::int64_t stride, std::int64_t dilation, Padding padding) {
  if (padding == Padding::Valid) return 0;
  const std::int64_t out   = nn::conv_output_size(in, k, stride, dilation, padding);
  const std::int64_t total = std::max<std::int64_t>(0, (out - 1) * stride + dilation * (k - 1) + 1 - in);
  return total / 2;
}

// Reference 2-D cross-correlation, NHWC, W [k, k, in, out]; with `depthwise`
// W is [k, k, in, mult] and there are in*mult outputs.
Tensor Conv2DRef(const Tensor& x, const Tensor& W, std::int64_t stride, std::int64_t dilation, Padding padding,
                 bool depthwise = false) {
  const std::int64_t N = x.size(0), H = x.size(1), Wd = x.size(2), Cin = x.size(3);
  const std::int64_t k = W.size(0), mult = W.size(3), Cout = depthwise ? Cin * mult : W.size(3);
  const std::int64_t Ho = nn::conv_output_size(H, k, stride, dilation, padding);
  const std::int64_t Wo = nn::conv_output_size(Wd, k, stride, dilation, padding);
  const std::int64_t ph = PadBefore(H, k, stride, dilation, padding), pw = PadBefore(Wd, k, stride, dilation, padding);
  const auto         X = x.to_vector<float>(), F = W.to_vector<float>();
  const std::int64_t Fc = W.size(2);
  std::vector<float> Y(static_cast<std::size_t>(N * Ho * Wo * Cout));
  for (std::int64_t n = 0; n < N; ++n)
    for (std::int64_t i = 0; i < Ho; ++i)
      for (std::int64_t j = 0; j < Wo; ++j)
        for (std::int64_t o = 0; o < Cout; ++o) {
          double acc = 0;
          for (std::int64_t a = 0; a < k; ++a)
            for (std::int64_t b = 0; b < k; ++b) {
              const std::int64_t r = i * stride + a * dilation - ph, c = j * stride + b * dilation - pw;
              if (r < 0 || r >= H || c < 0 || c >= Wd) continue;  // zero padding
              const float* xp = &X[((n * H + r) * Wd + c) * Cin];
              const float* fp = &F[(a * k + b) * Fc * mult];
              if (depthwise) {
                const std::int64_t ci = o / mult, m = o % mult;
                acc += xp[ci] * fp[ci * mult + m];
              } else {
                for (std::int64_t ci = 0; ci < Cin; ++ci) acc += xp[ci] * fp[ci * Cout + o];
              }
            }
          Y[((n * Ho + i) * Wo + j) * Cout + o] = static_cast<float>(acc);
        }
  return nn::from_vector(Y, {N, Ho, Wo, Cout});
}

float MaxDiff(const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape()) return INFINITY;
  const auto u = a.to_vector<float>(), v = b.to_vector<float>();
  float      d = 0;
  for (std::size_t i = 0; i < u.size(); ++i) d = std::max(d, std::abs(u[i] - v[i]));
  return d;
}

// Local 3-D convolution (not in the library yet): x [N, D, H, W, C], W [kd, kh, kw, C, O],
// VALID along depth, `padding` on H and W. y[:, d] = sum_a conv2d(x[:, d + a], W[a]).
Tensor Conv3DLocal(const Tensor& x, const Tensor& W, Padding padding) {
  const std::int64_t N = x.size(0), D = x.size(1), H = x.size(2), Wd = x.size(3), C = x.size(4);
  const std::int64_t kd = W.size(0), Do = D - kd + 1;
  std::vector<Tensor> slices;
  for (std::int64_t d = 0; d < Do; ++d) {
    Tensor acc;
    for (std::int64_t a = 0; a < kd; ++a) {
      const Tensor xs = nn::slice_axis(x, 1, d + a, d + a + 1).reshape({N, H, Wd, C});
      const Tensor ws = nn::slice_axis(W, 0, a, a + 1).reshape({W.size(1), W.size(2), C, W.size(4)});
      const Tensor y  = nn::conv2d(xs, ws, nn::conv_options(1, padding));
      acc             = acc.defined() ? acc + y : y;
    }
    slices.push_back(acc);
  }
  return nn::stack(slices, 1);
}

// ---------------------------------------------------------------------------
// 1. Output sizes
// ---------------------------------------------------------------------------

void OutputSizes() {
  std::printf("== Output length for input length 7, kernel 3: formula vs the op (Conv1D)\n");
  std::printf("  %-8s %7s %9s %9s %7s\n", "padding", "stride", "dilation", "formula", "op");
  for (Padding padding : {Padding::Valid, Padding::Same})
    for (int stride : {1, 2, 3})
      for (int dilation : {1, 2}) {
        if (stride > 1 && dilation > 1) continue;  // TF: unsupported combination on CPU
        nn::ParameterStore    p(1);
        nn::ConvLayerOptions  o;
        o.stride = stride, o.dilation = dilation, o.padding = padding, o.bias = false;
        const Tensor       y       = nn::conv1d(p, "conv", nn::zeros({1, 7, 1}), 1, 3, o);
        const std::int64_t formula = nn::conv_output_size(7, 3, stride, dilation, padding);
        std::printf("  %-8s %7d %9d %9lld %7lld\n", PadName(padding), stride, dilation, static_cast<long long>(formula),
                    static_cast<long long>(y.size(1)));
        check::Require(y.size(1) == formula, "conv1d output length vs formula");
      }
  // TF's formula column, for the 8 rows above.
  const std::int64_t expected[8] = {5, 3, 3, 2, 7, 7, 4, 3};
  int                row         = 0;
  for (Padding padding : {Padding::Valid, Padding::Same})
    for (int stride : {1, 2, 3})
      for (int dilation : {1, 2}) {
        if (stride > 1 && dilation > 1) continue;
        check::Require(nn::conv_output_size(7, 3, stride, dilation, padding) == expected[row++], "output size vs TF");
      }
}

// ---------------------------------------------------------------------------
// 2. 2-D convolution against the reference loops
// ---------------------------------------------------------------------------

void Conv2DChecks() {
  std::printf("\n== Conv2D on x [2, 6, 5, 3] with a 3x3 kernel to 4 channels (no bias), vs hand-written loops\n");
  std::mt19937_64 rng(1);
  const Tensor    xv = Random({2, 6, 5, 3}, rng);
  std::printf("  %-8s %7s %9s %14s %12s\n", "padding", "stride", "dilation", "output shape", "max |diff|");
  for (Padding padding : {Padding::Valid, Padding::Same})
    for (int stride : {1, 2})
      for (int dilation : {1, 2}) {
        if (stride > 1 && dilation > 1) continue;
        nn::ParameterStore   p(2);
        nn::ConvLayerOptions o;
        o.stride = stride, o.dilation = dilation, o.padding = padding, o.bias = false;
        const Tensor y = nn::conv2d(p, "conv", xv, 4, 3, o);
        const float  d = MaxDiff(y, Conv2DRef(xv, p.at("conv/W"), stride, dilation, padding));
        std::printf("  %-8s %7d %9d %14s %12.1e\n", PadName(padding), stride, dilation, ShapeStr(y).c_str(), d);
        check::Require(d < 1e-5f, "conv2d vs reference loops");
      }
  nn::ParameterStore p(2);
  nn::conv2d(p, "conv", xv, 4, 3);
  std::printf("  parameters of a 3x3 conv 3 -> 4 with bias: 3*3*3*4 + 4 = %lld\n", static_cast<long long>(p.count()));
  check::Require(p.count() == 112, "conv2d parameter count");
}

// ---------------------------------------------------------------------------
// 3. 1-D and 3-D convolutions
// ---------------------------------------------------------------------------

void Conv1DAnd3D() {
  std::printf("\n== Conv1D on x [2, 9, 2] (kernel 3, 3 channels) and Conv3D on x [1, 4, 4, 4, 2] (kernel 2, 3 channels), vs loops\n");
  std::mt19937_64      rng(3);
  nn::ParameterStore   p(3);
  nn::ConvLayerOptions o;
  o.bias = false, o.padding = Padding::Valid;
  const Tensor x1v = Random({2, 9, 2}, rng), x3v = Random({1, 4, 4, 4, 2}, rng);
  const Tensor y1  = nn::conv1d(p, "c1", x1v, 3, 3, o);
  const Tensor W3  = p.get("c3/W", {2, 2, 2, 2, 3}, nn::init::glorot_uniform());
  const Tensor y3  = Conv3DLocal(x3v, W3, Padding::Valid);
  // 1-D reference (W1 is [1, k, C, O]).
  const auto X1 = x1v.to_vector<float>(), W1 = p.at("c1/W").to_vector<float>(), Y1 = y1.to_vector<float>();
  float      d1 = 0;
  for (int n = 0; n < 2; ++n)
    for (int t = 0; t < 7; ++t)
      for (int o2 = 0; o2 < 3; ++o2) {
        double acc = 0;
        for (int a = 0; a < 3; ++a)
          for (int ci = 0; ci < 2; ++ci) acc += X1[(n * 9 + t + a) * 2 + ci] * W1[(a * 2 + ci) * 3 + o2];
        d1 = std::max<float>(d1, std::abs(acc - Y1[(n * 7 + t) * 3 + o2]));
      }
  std::printf("  Conv1D: output shape %s, max |diff| = %.1e\n", ShapeStr(y1).c_str(), d1);
  // 3-D reference.
  const auto X3 = x3v.to_vector<float>(), F3 = W3.to_vector<float>(), Y3 = y3.to_vector<float>();
  float      d3 = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int l = 0; l < 3; ++l)
        for (int o2 = 0; o2 < 3; ++o2) {
          double acc = 0;
          for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
              for (int c = 0; c < 2; ++c)
                for (int ci = 0; ci < 2; ++ci)
                  acc += X3[(((i + a) * 4 + j + b) * 4 + l + c) * 2 + ci] * F3[(((a * 2 + b) * 2 + c) * 2 + ci) * 3 + o2];
          d3 = std::max<float>(d3, std::abs(acc - Y3[((i * 3 + j) * 3 + l) * 3 + o2]));
        }
  std::printf("  Conv3D (local, sum of 2-D convs over depth): output shape %s, max |diff| = %.1e, parameters 2*2*2*2*3 = %lld\n",
              ShapeStr(y3).c_str(), d3, static_cast<long long>(W3.numel()));
  check::Require(ShapeStr(y1) == "[2,7,3]" && ShapeStr(y3) == "[1,3,3,3,3]", "conv1d/conv3d shapes");
  check::Require(d1 < 1e-5f && d3 < 1e-5f, "conv1d/conv3d vs reference loops");
}

// ---------------------------------------------------------------------------
// 4. Depthwise and separable convolutions
// ---------------------------------------------------------------------------

void DepthwiseSeparable() {
  std::printf("\n== Depthwise (multiplier 2) and separable convolutions on x [1, 6, 6, 3]\n");
  std::mt19937_64      rng(4);
  nn::ParameterStore   p(4);
  nn::ConvLayerOptions o;
  o.bias         = false;
  const Tensor xv  = Random({1, 6, 6, 3}, rng);
  const Tensor dw  = nn::depthwise_conv2d(p, "dw", xv, 2, 3, o);
  const Tensor sep = nn::separable_conv2d(p, "sep", xv, 8, 3, o);
  const float  ddw = MaxDiff(dw, Conv2DRef(xv, p.at("dw/W"), 1, 1, Padding::Same, true));
  std::printf("  depthwise: output shape %s (3 channels x multiplier 2), max |diff| vs loops = %.1e\n", ShapeStr(dw).c_str(), ddw);
  // Separable = depthwise then 1x1: compose the two references.
  const Tensor h    = Conv2DRef(xv, p.at("sep/depthwise/W"), 1, 1, Padding::Same, true);
  const float  dsep = MaxDiff(sep, Conv2DRef(h, p.at("sep/pointwise/W"), 1, 1, Padding::Valid));
  std::printf("  separable: output shape %s, max |diff| vs depthwise-then-pointwise loops = %.1e\n", ShapeStr(sep).c_str(), dsep);
  check::Require(ShapeStr(dw) == "[1,6,6,6]" && ShapeStr(sep) == "[1,6,6,8]", "depthwise/separable shapes");
  check::Require(ddw < 1e-5f && dsep < 1e-5f, "depthwise/separable vs reference loops");
  // Parameter counts, from the layers themselves.
  const int          cin = 64, cout = 128, k = 3;
  nn::ParameterStore full(0), sepp(0);
  const Tensor       z = nn::zeros({1, 4, 4, cin});
  nn::conv2d(full, "f", z, cout, k, o);
  nn::separable_conv2d(sepp, "s", z, cout, k, o);
  std::printf("  parameters for %dx%d, %d -> %d channels: full %lld, separable %lld (= %d depthwise + %d pointwise), ratio %.1fx\n", k, k,
              cin, cout, static_cast<long long>(full.count()), static_cast<long long>(sepp.count()), k * k * cin, cin * cout,
              static_cast<double>(full.count()) / static_cast<double>(sepp.count()));
  check::Require(full.count() == 73728 && sepp.count() == 8768, "full/separable parameter counts");
}

// ---------------------------------------------------------------------------
// 5. Transposed convolution
// ---------------------------------------------------------------------------

void Transposed() {
  std::printf("\n== Transposed convolution\n");
  std::mt19937_64 rng(5);
  // (a) It is the gradient of the convolution with respect to its input.
  {
    nn::ParameterStore   p(5);
    nn::ConvLayerOptions o;
    o.bias = false, o.stride = 2, o.padding = Padding::Same;
    Tensor       x = Random({1, 6, 6, 2}, rng);
    const Tensor r = Random({1, 3, 3, 3}, rng);
    x.requires_grad_();
    const Tensor y    = nn::conv2d(p, "conv", x, 3, 3, o);  // x [1, 6, 6, 2] -> y [1, 3, 3, 3]
    const Tensor loss = (y * r).sum();                      // dL/dy = r
    const Tensor gx   = nn::grad(loss, {x})[0];
    // The transposed conv with the *same* kernel applied to r.
    const Tensor W      = p.at("conv/W").detach();
    const Tensor xt     = nn::conv2d_backprop_input({1, 6, 6, 2}, W, r, nn::conv_options(2));
    const Tensor xt2    = nn::conv2d_transpose(r, W, {1, 6, 6, 2}, nn::conv_options(2));
    const float  d_bpi  = MaxDiff(gx, xt);
    const float  d_tconv = MaxDiff(gx, xt2);
    std::printf("  (a) stride-2 SAME conv [1,6,6,2] -> [1,3,3,3]: max |dL/dx - transposed_conv(dL/dy)| = %.1e"
                " (conv2d_backprop_input), %.1e (conv2d_transpose)\n", d_bpi, d_tconv);
    check::Require(d_bpi < 1e-6f && d_tconv < 1e-6f, "transposed conv as the input gradient");
  }
  // (b) Shapes as a learnable upsampler, and (c) the checkerboard of overlaps.
  {
    nn::ParameterStore   p(6);
    nn::ConvLayerOptions same, valid;
    same.stride = 2, same.padding = Padding::Same, same.bias = false;
    valid.stride = 2, valid.padding = Padding::Valid, valid.bias = false;
    const Tensor xv       = nn::ones({1, 4, 4, 1});
    nn::conv2d_transpose(p, "same", xv, 1, 3, same);  // creates same/W
    // Ones kernel, ones input: the output counts how many kernel copies overlap each pixel.
    p.set("same/W", nn::ones({3, 3, 1, 1}));
    const Tensor up_same  = nn::conv2d_transpose(p, "same", xv, 1, 3, same);
    const Tensor up_valid = nn::conv2d_transpose(p, "valid", xv, 1, 3, valid);
    std::printf("  (b) stride 2, kernel 3: input [1,4,4,1] -> SAME %s, VALID %s\n", ShapeStr(up_same).c_str(), ShapeStr(up_valid).c_str());
    check::Require(ShapeStr(up_same) == "[1,8,8,1]" && ShapeStr(up_valid) == "[1,9,9,1]", "transposed conv shapes");
    std::printf("  (c) ones input through a ones 3x3 kernel, stride 2, SAME: overlap counts per output pixel\n");
    const auto v = up_same.to_vector<float>();
    for (int i = 0; i < 8; ++i) {
      std::printf("      ");
      for (int j = 0; j < 8; ++j) {
        std::printf(" %2.0f", v[i * 8 + j]);
        // TF's pattern: rows/cols 0, 1 and odd ones see 1 copy, even ones >= 2 see 2.
        const int ci = (i >= 2 && i % 2 == 0) ? 2 : 1, cj = (j >= 2 && j % 2 == 0) ? 2 : 1;
        check::Require(v[i * 8 + j] == ci * cj, "checkerboard overlap counts");
      }
      std::printf("\n");
    }
    std::printf("      (uneven overlap when the kernel size is not a multiple of the stride: the checkerboard artifact)\n");
  }
}

// ---------------------------------------------------------------------------
// 6. Dilated convolution
// ---------------------------------------------------------------------------

void Dilated() {
  std::printf("\n== Dilated convolution: a 3x3 kernel with dilation 2 equals a 5x5 kernel with holes\n");
  std::mt19937_64      rng(7);
  nn::ParameterStore   p(7);
  nn::ConvLayerOptions o;
  o.bias = false, o.dilation = 2, o.padding = Padding::Same;
  const Tensor xv = Random({1, 8, 8, 1}, rng);
  const Tensor y  = nn::conv2d(p, "d", xv, 1, 3, o);
  // The equivalent 5x5 kernel: W5[2a, 2b] = W3[a, b], zeros in between.
  const auto         w3 = p.at("d/W").to_vector<float>();
  std::vector<float> w5(25, 0.f);
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) w5[2 * a * 5 + 2 * b] = w3[a * 3 + b];
  const Tensor y5 = nn::conv2d(xv, nn::from_vector(w5, {5, 5, 1, 1}), nn::conv_options(1, Padding::Same));  // ordinary 5x5
  std::printf("  the 5x5 kernel with holes:\n");
  for (int i = 0; i < 5; ++i) {
    std::printf("      ");
    for (int j = 0; j < 5; ++j) std::printf(" %6.3f", w5[i * 5 + j]);
    std::printf("\n");
  }
  const float d = MaxDiff(y, y5);
  std::printf("  max |dilated 3x3 conv - 5x5 conv with holes| = %.1e\n", d);
  check::Require(d < 1e-5f, "dilated conv vs 5x5 kernel with holes");
  std::printf("  receptive field of a stack of 3x3 convolutions, stride 1: dilations 1,1,1 -> %d; 1,2,4 -> %d; 1,2,4,8 -> %d\n",
              1 + 2 * (1 + 1 + 1), 1 + 2 * (1 + 2 + 4), 1 + 2 * (1 + 2 + 4 + 8));
}

}  // namespace

int main() {
  try {
    OutputSizes();
    Conv2DChecks();
    Conv1DAnd3D();
    DepthwiseSeparable();
    Transposed();
    Dilated();
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
