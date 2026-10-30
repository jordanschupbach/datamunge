// CMD: just run-nn convolution_im2col
// Convolution from scratch through im2col, GEMM and col2im.
//
// Port of tflow's convolution_im2col. Explicitly pack NHWC patches into rows
// and reshape HWIO kernels into a matrix. Multiply them with nn::matmul and
// compare every output with nn::conv2d. Backpropagate through the two matrix
// factors and scatter-add patch gradients with col2im; compare input/filter
// derivatives with nn's convolution backprop (conv2d_backprop_input /
// conv2d_backprop_filter) and selected input derivatives with finite
// differences. Cover VALID/SAME, rectangular kernels, multiple batches/channels,
// spatial strides and dilation. Also check the im2col/col2im adjoint and
// overlap counts.
// TensorFlow's CPU backprop kernels reject dilation>1, so the TF version uses
// the exactly equivalent larger kernel with inserted zeros for those gradient
// references. nn's backprop supports dilation natively; this port keeps the
// zero-expanded kernel reference as well, so both are checked.
// Inputs come from the same std::mt19937(149) stream as TF's (bit-identical).

#include <algorithm>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <sstream>

#include "common/checks.hpp"
#include "common/im2col.hpp"

namespace nn = datamunge::nn;
using im2col::PatchGeometry;
using nn::Shape;
using nn::Tensor;

namespace {

Tensor Random(const Shape& shape, std::mt19937& rng) {
  std::int64_t n = 1;
  for (auto d : shape) n *= d;
  std::vector<float>              v(static_cast<std::size_t>(n));
  std::normal_distribution<float> distribution(0, .3);
  for (auto& e : v) e = distribution(rng);
  return nn::from_vector(v, shape);
}

double Difference(const Tensor& x, const Tensor& y) {
  check::Require(x.shape() == y.shape(), "convolution comparison shape");
  const auto a = x.contiguous().to_vector<float>(), b = y.contiguous().to_vector<float>();
  double     result = 0;
  for (std::size_t i = 0; i < a.size(); ++i) result = std::max(result, std::abs(double(a[i]) - b[i]));
  return result;
}

std::string ShapeStr(const Shape& s) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < s.size(); ++i) out << (i ? "," : "") << s[i];
  return out.str() + ']';
}

void Compare(const Shape& input_shape, int kh, int kw, int outputs, int sh, int sw, int dh, int dw,
             const std::string& padding, std::mt19937& rng) {
  const PatchGeometry g(input_shape, kh, kw, sh, sw, dh, dw, padding);
  const Shape         filter_shape{kh, kw, g.c, outputs}, output_shape{g.n, g.oh, g.ow, outputs};
  const auto input = Random(input_shape, rng), filter = Random(filter_shape, rng), upstream = Random(output_shape, rng);
  // Zero-expanded kernel: the dilated filter written as an ordinary larger one.
  const Shape        expanded_shape{(kh - 1) * dh + 1, (kw - 1) * dw + 1, g.c, outputs};
  const auto         f = filter.to_vector<float>();
  std::vector<float> expanded(static_cast<std::size_t>(expanded_shape[0] * expanded_shape[1] * g.c * outputs), 0.f);
  for (int a = 0; a < kh; ++a)
    for (int b = 0; b < kw; ++b)
      for (std::int64_t c = 0; c < g.c; ++c)
        for (int o = 0; o < outputs; ++o)
          expanded[static_cast<std::size_t>(((a * dh * expanded_shape[1] + b * dw) * g.c + c) * outputs + o)] =
              f[static_cast<std::size_t>(((a * kw + b) * g.c + c) * outputs + o)];
  const Tensor expanded_filter = nn::from_vector(expanded, expanded_shape);

  const auto   patches = im2col::Im2Col(input, g);
  const Tensor kernel  = nn::reshape(filter, {g.columns(), outputs});
  auto         manual_of = [&](const Tensor& columns) {
    return nn::reshape(nn::matmul(columns, kernel), output_shape);
  };
  const Tensor manual = manual_of(patches);
  nn::Conv2DOptions options;
  options.strides   = {sh, sw};
  options.dilations = {dh, dw};
  options.padding   = padding == "SAME" ? nn::Padding::Same : nn::Padding::Valid;
  const Tensor native = nn::conv2d(input, filter, options);
  const Tensor dy       = nn::reshape(upstream, {g.rows(), outputs});
  const Tensor dcolumns = nn::matmul(dy, kernel, false, true);
  const Tensor dfilter  = nn::reshape(nn::matmul(patches, dy, true, false), filter_shape);
  // Native dilated backprop, and the zero-expanded undilated kernel as TF uses it.
  const Tensor native_dx = nn::conv2d_backprop_input(input_shape, filter, upstream, options);
  const Tensor native_dw = nn::conv2d_backprop_filter(input, filter_shape, upstream, options);
  nn::Conv2DOptions plain = options;
  plain.dilations         = {1, 1};
  const Tensor expanded_dx = nn::conv2d_backprop_input(input_shape, expanded_filter, upstream, plain);
  const auto   expanded_dw = nn::conv2d_backprop_filter(input, expanded_shape, upstream, plain).to_vector<float>();
  std::vector<float> gathered(f.size());
  for (int a = 0; a < kh; ++a)
    for (int b = 0; b < kw; ++b)
      for (std::int64_t c = 0; c < g.c; ++c)
        for (int o = 0; o < outputs; ++o)
          gathered[static_cast<std::size_t>(((a * kw + b) * g.c + c) * outputs + o)] =
              expanded_dw[static_cast<std::size_t>(((a * dh * expanded_shape[1] + b * dw) * g.c + c) * outputs + o)];
  const Tensor reference_dw = nn::from_vector(gathered, filter_shape);

  const auto   dx             = im2col::Col2Im(dcolumns, g);
  const double forward_error  = Difference(manual, native), input_error = Difference(dx, native_dx),
               filter_error   = Difference(dfilter, native_dw);
  const double expanded_error = std::max(Difference(dx, expanded_dx), Difference(dfilter, reference_dw));
  check::Require(forward_error < 2e-6 && input_error < 2e-6 && filter_error < 2e-6,
                 "GEMM convolution and gradients match native kernels");
  check::Require(expanded_error < 2e-6, "zero-expanded kernel gradients match dilated GEMM gradients");
  double     lhs = 0, rhs = 0;
  const auto p = patches.to_vector<float>(), dc = dcolumns.to_vector<float>(), xi = input.to_vector<float>(),
             dxv = dx.to_vector<float>(), up = upstream.to_vector<float>();
  for (std::size_t i = 0; i < p.size(); ++i) lhs += double(p[i]) * dc[i];
  for (std::size_t i = 0; i < xi.size(); ++i) rhs += double(xi[i]) * dxv[i];
  check::Near(lhs, rhs, 2e-6, "im2col/col2im adjoint identity");
  double finite_error = 0;
  for (std::int64_t index : {std::int64_t{0}, input.numel() / 2, input.numel() - 1}) {
    const float epsilon = .002f;
    double      values[2]{};
    for (int side = 0; side < 2; ++side) {
      auto changed = xi;
      changed[static_cast<std::size_t>(index)] += side ? epsilon : -epsilon;
      const auto out = manual_of(im2col::Im2Col(nn::from_vector(changed, input_shape), g)).to_vector<float>();
      for (std::size_t i = 0; i < up.size(); ++i) values[side] += double(out[i]) * up[i];
    }
    const double numeric = (values[1] - values[0]) / (2 * epsilon);
    finite_error         = std::max(finite_error, std::abs(numeric - dxv[static_cast<std::size_t>(index)]));
  }
  check::Require(finite_error < 3e-5, "convolution input finite-difference derivative");
  std::cout << "  " << padding << " " << ShapeStr(input_shape) << " kernel=" << kh << 'x' << kw << " stride=" << sh
            << ',' << sw << " dilation=" << dh << ',' << dw << ": output=" << ShapeStr(output_shape)
            << "; max errors(y,dx,dw,finite)=" << forward_error << ',' << input_error << ',' << filter_error << ','
            << finite_error << '\n';
}

}  // namespace

int main() {
  try {
    std::mt19937 rng(149);
    std::cout << "== Convolution through im2col + GEMM\n";
    Compare({2, 5, 6, 2}, 3, 2, 3, 1, 1, 1, 1, "VALID", rng);
    Compare({2, 5, 6, 2}, 3, 2, 3, 2, 2, 1, 1, "SAME", rng);
    Compare({1, 6, 7, 3}, 2, 3, 2, 2, 1, 1, 1, "SAME", rng);
    Compare({1, 7, 8, 2}, 3, 2, 3, 1, 1, 2, 2, "VALID", rng);
    Compare({1, 7, 8, 2}, 3, 2, 3, 1, 1, 2, 2, "SAME", rng);
    Compare({2, 4, 5, 3}, 1, 1, 4, 1, 1, 1, 1, "VALID", rng);
    const Tensor        ones = nn::ones({1, 5, 5, 1});
    const PatchGeometry g(ones.shape(), 3, 3);
    const auto          coverage = im2col::Col2Im(im2col::Im2Col(ones, g), g);
    check::Near(coverage.at({0, 0, 0, 0}), 1, 0, "corner covered once");
    check::Near(coverage.at({0, 2, 2, 0}), 9, 0, "central pixel accumulates nine patches");
    bool rejected = false;
    try {
      PatchGeometry invalid(ones.shape(), 0, 3);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check::Require(rejected, "invalid patch kernel rejected");
    std::cout << "  overlap accumulation: corner=1, center=9; invalid geometry rejected\n"
                 "All im2col convolution checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
