// CMD: just run-nn pooling
// Pooling: max, average and global.
//
// Port of tflow's pooling. Pooling summarises each window of a feature map by
// one number, with no parameters; datamunge/nn/conv.hpp provides max_pool2d,
// avg_pool2d, global_{avg,max}_pool2d and their 3-D versions max_pool3d /
// avg_pool3d. This program prints the pooled maps
// of a small image by hand next to the ops, shows how SAME padding changes the
// average at the border, follows the gradient (max pooling routes it to the
// arg-max, average pooling spreads it uniformly), checks global pooling
// against reductions, and measures the small translation tolerance pooling
// buys.
//
// TF -> datamunge: no graph or session; AddSymbolicGradients is nn::grad;
// MaxPool3D / AvgPool3D are nn::max_pool3d / nn::avg_pool3d. Inputs come from
// the same std::mt19937_64 streams as TF's, so every printed number matches TF's.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/conv.hpp>
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

void PrintMap(const char* label, const Tensor& t, const char* fmt = " %5.1f") {
  const auto v = t.to_vector<float>();
  const auto H = t.size(1), W = t.size(2), C = t.size(3);
  std::printf("  %s [%lld x %lld]\n", label, static_cast<long long>(H), static_cast<long long>(W));
  for (std::int64_t i = 0; i < H; ++i) {
    std::printf("     ");
    for (std::int64_t j = 0; j < W; ++j) std::printf(fmt, v[(i * W + j) * C]);
    std::printf("\n");
  }
}

Tensor Image(std::int64_t h, std::int64_t w, const std::vector<float>& values) { return nn::from_vector(values, {1, h, w, 1}); }

std::string ShapeStr(const Tensor& t) {
  std::string s = "[";
  for (std::size_t i = 0; i < t.shape().size(); ++i) s += (i ? "," : "") + std::to_string(t.shape()[i]);
  return s + "]";
}

void RequireMap(const Tensor& t, const std::vector<float>& expected, const std::string& what) {
  const auto v = t.to_vector<float>();
  check::Require(v.size() == expected.size(), what + ": size");
  for (std::size_t i = 0; i < v.size(); ++i) check::Near(v[i], expected[i], 1e-6, what);
}

// ---------------------------------------------------------------------------
// 1. Max and average pooling by hand, VALID and SAME
// ---------------------------------------------------------------------------

void ByHand() {
  std::printf("== 2x2 windows, stride 2, on a 5x5 image\n");
  const Tensor xv        = Image(5, 5, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
  const Tensor max_valid = nn::max_pool2d(xv, 2, 2, Padding::Valid), avg_valid = nn::avg_pool2d(xv, 2, 2, Padding::Valid);
  const Tensor max_same  = nn::max_pool2d(xv, 2, 2, Padding::Same), avg_same = nn::avg_pool2d(xv, 2, 2, Padding::Same);
  PrintMap("input", xv);
  PrintMap("max pool VALID (windows that fit entirely: rows/cols 0-3)", max_valid);
  PrintMap("avg pool VALID", avg_valid);
  PrintMap("max pool SAME (a third window covers only row/col 4)", max_same);
  PrintMap("avg pool SAME (the border window averages its 1 or 2 real entries, padding is not counted)", avg_same, " %6.2f");
  // Hand computation of the SAME average at the bottom-right corner.
  std::printf("  check: avg SAME at (2, 2) = 25 / 1 = %.2f, at (2, 1) = (23 + 24) / 2 = %.2f, at (0, 0) = (1 + 2 + 6 + 7) / 4 = %.2f\n",
              avg_same.at({0, 2, 2, 0}), avg_same.at({0, 2, 1, 0}), avg_same.at({0, 0, 0, 0}));
  RequireMap(max_valid, {7, 9, 17, 19}, "max pool VALID");
  RequireMap(avg_valid, {4, 6, 14, 16}, "avg pool VALID");
  RequireMap(max_same, {7, 9, 10, 17, 19, 20, 22, 24, 25}, "max pool SAME");
  RequireMap(avg_same, {4, 6, 7.5, 14, 16, 17.5, 21.5, 23.5, 25}, "avg pool SAME");
}

// ---------------------------------------------------------------------------
// 2. Gradients: routing vs spreading
// ---------------------------------------------------------------------------

void Gradients() {
  std::printf("\n== dL/dx for L = sum(pool(x)), 2x2 stride 2 VALID, on a 4x4 image with one large entry per window\n");
  Tensor xv = Image(4, 4, {9, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 7, 1, 6, 1});
  xv.requires_grad_();
  const Tensor gm = nn::grad(nn::max_pool2d(xv, 2, 2).sum(), {xv})[0];
  const Tensor ga = nn::grad(nn::avg_pool2d(xv, 2, 2).sum(), {xv})[0];
  PrintMap("input", xv, " %4.0f");
  PrintMap("dL/dx through max pooling: 1 at each window's arg-max, 0 elsewhere", gm, " %5.2f");
  PrintMap("dL/dx through average pooling: 1/4 everywhere", ga, " %5.2f");
  RequireMap(gm, {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0}, "max pool gradient");
  RequireMap(ga, std::vector<float>(16, 0.25f), "avg pool gradient");
}

// ---------------------------------------------------------------------------
// 3. Global pooling and 3-D pooling
// ---------------------------------------------------------------------------

void GlobalAnd3D() {
  std::printf("\n== Global pooling [2, 7, 5, 3] -> [2, 3], and 3-D pooling [1, 8, 8, 8, 2] with 2x2x2 windows\n");
  std::mt19937_64                 rng(1);
  std::normal_distribution<float> normal(0.f, 1.f);
  std::vector<float>              xs(2 * 7 * 5 * 3), vs(8 * 8 * 8 * 2);
  for (auto& e : xs) e = normal(rng);
  for (auto& e : vs) e = normal(rng);
  const Tensor xv = nn::from_vector(xs, {2, 7, 5, 3}), vv = nn::from_vector(vs, {1, 8, 8, 8, 2});
  const Tensor gavg = nn::global_avg_pool2d(xv), gmax = nn::global_max_pool2d(xv);
  const Tensor m3 = nn::max_pool3d(vv, 2, 2), a3 = nn::avg_pool3d(vv, 2, 2);
  float err_avg = 0, err_max = 0;
  for (int n = 0; n < 2; ++n)
    for (int c = 0; c < 3; ++c) {
      double s = 0, m = -1e9;
      for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 5; ++j) {
          const float v = xs[((n * 7 + i) * 5 + j) * 3 + c];
          s += v, m = std::max<double>(m, v);
        }
      err_avg = std::max<float>(err_avg, std::abs(s / 35 - gavg.at({n, c})));
      err_max = std::max<float>(err_max, std::abs(m - gmax.at({n, c})));
    }
  std::printf("  global average: shape %s, max |diff| vs mean over H, W = %.1e; global max: max |diff| = %.1e\n", ShapeStr(gavg).c_str(),
              err_avg, err_max);
  // 3-D: check one cell of each.
  double m = -1e9, s = 0;
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int c = 0; c < 2; ++c) {
        const float val = vs[(((2 + a) * 8 + 4 + b) * 8 + 6 + c) * 2 + 1];
        m = std::max<double>(m, val), s += val;
      }
  const double op_m = m3.at({0, 1, 2, 3, 1}), op_a = a3.at({0, 1, 2, 3, 1});
  std::printf("  3-D max pool: shape %s, cell (1,2,3,ch 1): op %.4f, hand %.4f; avg pool: op %.4f, hand %.4f\n",
              ShapeStr(m3).c_str(), op_m, m, op_a, s / 8);
  check::Require(ShapeStr(gavg) == "[2,3]" && ShapeStr(m3) == "[1,4,4,4,2]" && ShapeStr(a3) == "[1,4,4,4,2]", "global/3-D pool shapes");
  check::Require(err_avg < 1e-6f && err_max == 0.f, "global pooling vs reductions");
  check::Near(op_m, m, 1e-6, "3-D max pool cell");
  check::Near(op_a, s / 8, 1e-6, "3-D avg pool cell");
}

// ---------------------------------------------------------------------------
// 4. Translation tolerance
// ---------------------------------------------------------------------------

void Translation() {
  std::printf("\n== Translation tolerance: a random 16x16 image vs the same image shifted right by 1 pixel\n");
  std::mt19937_64                       rng(2);
  std::uniform_real_distribution<float> U(0.f, 1.f);
  std::vector<float>                    av(256), bv(256);
  for (auto& e : av) e = U(rng);
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) bv[i * 16 + j] = j == 0 ? 0.f : av[i * 16 + j - 1];
  const Tensor a = Image(16, 16, av), b = Image(16, 16, bv);
  auto         pools = [](const Tensor& x) {
    const Tensor p2 = nn::max_pool2d(x, 2, 2);
    return std::vector<Tensor>{p2, nn::max_pool2d(p2, 2, 2), nn::global_max_pool2d(x)};
  };
  const auto out = pools(a), out_shift = pools(b);
  auto agreement = [](const Tensor& u, const Tensor& w, float tol) {
    const auto x = u.to_vector<float>(), y = w.to_vector<float>();
    int        same = 0;
    for (std::size_t i = 0; i < x.size(); ++i) same += std::abs(x[i] - y[i]) <= tol;
    return 100.0 * same / static_cast<double>(x.size());
  };
  const double raw = agreement(a, b, 1e-6f), one = agreement(out[0], out_shift[0], 1e-6f), two = agreement(out[1], out_shift[1], 1e-6f);
  const bool   global_same = std::abs(out[2].item<float>() - out_shift[2].item<float>()) < 1e-6f;
  std::printf("  entries unchanged by the shift (tolerance 1e-6): raw %.0f%%, after one 2x2 max pool %.0f%%, after two %.0f%%, global max %s\n",
              raw, one, two, global_same ? "unchanged" : "changed");
  // TF (same image): raw 0%, 28%, 31%, unchanged.
  check::Require(std::round(raw) == 0 && std::round(one) == 28 && std::round(two) == 31 && global_same, "translation tolerance vs TF");
}

}  // namespace

int main() {
  try {
    ByHand();
    Gradients();
    GlobalAnd3D();
    Translation();
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
