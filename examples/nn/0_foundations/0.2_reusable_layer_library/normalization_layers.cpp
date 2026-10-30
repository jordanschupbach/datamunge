// CMD: just run-nn normalization_layers
// Layer normalization, group normalization, instance normalization and RMSNorm.
//
// Port of tflow's normalization_layers. The normalisers in
// datamunge/nn/layers.hpp differ only in the axes their mean and variance are
// taken over: layer norm over the feature axis of each example, group norm
// over the spatial axes and a group of channels of each example (instance
// norm being one channel per group, layer norm for images being a single
// group), and RMSNorm dropping the centering altogether. This program checks
// each against a hand computation, shows that they are independent of the
// other examples in the batch (unlike batch norm), verifies the group-norm
// limits G = 1 and G = C, and measures the shift and scale invariances that
// make them work.
//
// TF -> datamunge: no graph or session; the feature / channel count is read
// from x instead of being passed. The inputs come from the same
// std::mt19937_64 streams as TF, so the printed values match TF's.

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
using nn::Tensor;

namespace {

constexpr double kEps = 1e-5;

std::vector<float> Random(std::size_t n, std::mt19937_64& rng, float mean = 0.f, float std = 1.f) {
  std::vector<float>              v(n);
  std::normal_distribution<float> normal(mean, std);
  for (float& x : v) x = normal(rng);
  return v;
}

// Standardise the entries of `t` grouped by `group(i)` over all members of
// each group (no centering if !center). Returns the result.
template <typename Group>
std::vector<double> StandardiseBy(const std::vector<float>& t, int ngroups, Group group, double eps, bool center = true) {
  std::vector<double> sum(ngroups, 0), sq(ngroups, 0), cnt(ngroups, 0);
  for (std::size_t i = 0; i < t.size(); ++i) {
    const int g = group(i);
    sum[g] += t[i], sq[g] += double(t[i]) * t[i], cnt[g] += 1;
  }
  std::vector<double> y(t.size());
  for (std::size_t i = 0; i < t.size(); ++i) {
    const int    g    = group(i);
    const double mean = center ? sum[g] / cnt[g] : 0, var = sq[g] / cnt[g] - mean * mean;
    y[i]              = (t[i] - mean) / std::sqrt(var + eps);
  }
  return y;
}

float MaxErr(const Tensor& got, const std::vector<double>& want) {
  const std::vector<float> g = got.to_vector<float>();
  float                    e = 0;
  for (std::size_t i = 0; i < g.size(); ++i) e = std::max<float>(e, std::abs(g[i] - want[i]));
  return e;
}

float MaxDiff(const Tensor& a, const Tensor& b) {
  const std::vector<float> u = a.to_vector<float>(), v = b.to_vector<float>();
  float                    d = 0;
  for (std::size_t i = 0; i < u.size(); ++i) d = std::max(d, std::abs(u[i] - v[i]));
  return d;
}

// ---------------------------------------------------------------------------
// 1. Layer norm and RMSNorm on vectors and sequences, batch independence
// ---------------------------------------------------------------------------

void LayerAndRms() {
  std::printf("== Layer norm and RMSNorm over the last axis (features = 6)\n");
  nn::ParameterStore p;
  auto               ln  = [&](const Tensor& x) { return nn::layer_norm(p, "ln", x, kEps); };
  auto               rms = [&](const Tensor& x) { return nn::rms_norm(p, "rms", x, kEps); };
  std::mt19937_64    rng(1);
  const std::vector<float> xs = Random(18, rng, 2.f, 3.f);
  const Tensor             xv = nn::from_vector(xs, {3, 6});
  const Tensor             y_ln = ln(xv), y_rms = rms(xv);
  auto                     row  = [](std::size_t i) { return static_cast<int>(i / 6); };
  const float              e_ln = MaxErr(y_ln, StandardiseBy(xs, 3, row, kEps));
  const float              e_rms = MaxErr(y_rms, StandardiseBy(xs, 3, row, kEps, false));
  std::printf("  [3, 6] input: max |LN - hand| = %.1e, max |RMSNorm - hand| = %.1e\n", e_ln, e_rms);
  std::printf("  row 0:      x = ");
  for (int j = 0; j < 6; ++j) std::printf(" %7.3f", xs[j]);
  std::printf("\n              LN(x) = ");
  for (int j = 0; j < 6; ++j) std::printf(" %7.3f", y_ln.at({0, j}));
  std::printf("\n          RMSNorm(x) = ");
  for (int j = 0; j < 6; ++j) std::printf(" %7.3f", y_rms.at({0, j}));
  std::printf("\n");
  // TF row 0: LN 0.224 -0.552 -0.244 1.846 0.190 -1.464; RMSNorm 0.907 0.405 0.604 1.957 0.885 -0.186.
  check::Require(e_ln < 1e-5f && e_rms < 1e-5f, "LN / RMSNorm disagree with the hand computation");
  // Sequence input [2, 4, 6]: each position is normalised on its own.
  const std::vector<float> ss = Random(48, rng, -1.f, 2.f);
  const Tensor             ys = ln(nn::from_vector(ss, {2, 4, 6}));
  const float              e_seq = MaxErr(ys, StandardiseBy(ss, 8, row, kEps));
  std::printf("  [2, 4, 6] input: output shape [%lld,%lld,%lld], max |LN - per-position hand| = %.1e\n",
              static_cast<long long>(ys.size(0)), static_cast<long long>(ys.size(1)), static_cast<long long>(ys.size(2)),
              e_seq);
  check::Require(ys.dim() == 3 && ys.size(0) == 2 && ys.size(1) == 4 && ys.size(2) == 6 && e_seq < 1e-5f,
                 "LN on a sequence");
  // Batch independence: change row 1, row 0's output is unchanged.
  std::vector<float> xw = xs;
  for (int j = 0; j < 6; ++j) xw[6 + j] = 100.f * (j - 2);
  const Tensor yw = ln(nn::from_vector(xw, {3, 6}));
  float        d  = 0;
  for (int j = 0; j < 6; ++j) d = std::max<float>(d, std::abs(y_ln.at({0, j}) - yw.at({0, j})));
  std::printf("  replacing row 1 by other values changes row 0's LN output by %.1e (batch norm would change it)\n", d);
  check::Require(d == 0, "LN must not depend on other rows");
  // Rows with zero mean: RMSNorm equals layer norm.
  std::vector<float> z = xs;
  for (int i = 0; i < 3; ++i) {
    float m = 0;
    for (int j = 0; j < 6; ++j) m += z[i * 6 + j] / 6;
    for (int j = 0; j < 6; ++j) z[i * 6 + j] -= m;
  }
  const Tensor zt = nn::from_vector(z, {3, 6});
  d               = MaxDiff(ln(zt), rms(zt));
  std::printf("  on zero-mean rows RMSNorm and LN agree to %.1e\n", d);
  check::Require(d < 1e-5f, "RMSNorm == LN on zero-mean rows");
}

// ---------------------------------------------------------------------------
// 2. Group norm on images: G = 1 (layer norm), G = 2, G = C (instance norm)
// ---------------------------------------------------------------------------

void GroupNorms() {
  std::printf(
      "\n== Group norm on x [2, 4, 4, 8] (N, H, W, C): statistics per (example, group of C/G channels) over H x W x "
      "C/G\n");
  nn::ParameterStore p;
  std::mt19937_64    rng(2);
  std::vector<float> xs = Random(2 * 4 * 4 * 8, rng);
  // Give each channel its own offset and scale so the groupings differ visibly.
  for (std::size_t i = 0; i < xs.size(); ++i) {
    const int c = static_cast<int>(i % 8);
    xs[i]       = xs[i] * (1 + c) + 3 * c;
  }
  const Tensor            xv     = nn::from_vector(xs, {2, 4, 4, 8});
  const std::vector<int>  groups = {1, 2, 4, 8};
  std::vector<Tensor>     out;
  for (int g : groups) out.push_back(nn::group_norm(p, "gn" + std::to_string(g), xv, g, kEps));
  const Tensor inst = nn::instance_norm(p, "in", xv, kEps);
  std::printf("  %4s %12s %16s\n", "G", "stats/example", "max |GN - hand|");
  for (size_t k = 0; k < groups.size(); ++k) {
    const int   G = groups[k], per = 8 / G;
    auto        group = [=](std::size_t i) { return static_cast<int>((i / 128) * G + (i % 8) / per); };  // 128 = H*W*C
    const float e     = MaxErr(out[k], StandardiseBy(xs, 2 * G, group, kEps));
    std::printf("  %4d %12d %16.1e\n", G, G, e);
    check::Require(e < 1e-5f, "group norm vs hand computation");
  }
  const float d = MaxDiff(out[3], inst);
  std::printf("  InstanceNorm == GroupNorm(G = C): max difference %.1e\n", d);
  check::Require(d == 0, "instance norm must equal group norm with G = C");
  // Output statistics per (example, channel) for G = 1 and G = 8.
  const std::vector<float> g1 = out[0].to_vector<float>(), g8 = out[3].to_vector<float>();
  std::printf("  G = 1: per-channel means of example 0 after normalisation (channel scales survive): ");
  for (int c = 0; c < 8; ++c) {
    double s = 0;
    for (int hw = 0; hw < 16; ++hw) s += g1[hw * 8 + c];
    std::printf(" %6.2f", s / 16);
  }
  std::printf("\n  G = 8: per-channel means of example 0 after normalisation (each channel standardised): ");
  for (int c = 0; c < 8; ++c) {
    double s = 0;
    for (int hw = 0; hw < 16; ++hw) s += g8[hw * 8 + c];
    std::printf(" %6.2f", s / 16);
    check::Near(s / 16, 0, 1e-5, "G = 8 channel mean");
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// 3. Invariances
// ---------------------------------------------------------------------------

void Invariances() {
  std::printf("\n== Invariances on a [4, 16] input (gamma = 1, beta = 0): max |f(T x) - f(x)|\n");
  nn::ParameterStore p;
  auto               ln  = [&](const Tensor& x) { return nn::layer_norm(p, "ln", x, kEps); };
  auto               rms = [&](const Tensor& x) { return nn::rms_norm(p, "rms", x, kEps); };
  std::mt19937_64    rng(3);
  const std::vector<float> xs = Random(64, rng, 0.5f, 2.f);
  const Tensor             xv = nn::from_vector(xs, {4, 16});
  const Tensor             b_ln = ln(xv), b_rms = rms(xv);
  struct T {
    const char* label;
    float       a, c;
    bool        ln_inv, rms_inv;
  };  // x -> a x + c (per row the same)
  const T transforms[] = {{"x -> x + 5 (shift)", 1, 5, true, false},
                          {"x -> 3 x (scale)", 3, 0, true, true},
                          {"x -> -2 x (negative scale)", -2, 0, false, false},
                          {"x -> 3 x + 5", 3, 5, true, false}};
  std::printf("  %-30s %12s %12s\n", "transform", "LayerNorm", "RMSNorm");
  for (const T& tr : transforms) {
    std::vector<float> xt = xs;
    for (float& v : xt) v = tr.a * v + tr.c;
    const Tensor t     = nn::from_vector(xt, {4, 16});
    const float  d_ln  = MaxDiff(ln(t), b_ln);
    const float  d_rms = MaxDiff(rms(t), b_rms);
    std::printf("  %-30s %12.1e %12.1e\n", tr.label, d_ln, d_rms);
    check::Require((d_ln < 1e-4f) == tr.ln_inv, std::string("LN invariance: ") + tr.label);
    check::Require((d_rms < 1e-4f) == tr.rms_inv, std::string("RMSNorm invariance: ") + tr.label);
  }
  std::printf("  (LN is invariant to any affine change of a row with a > 0; RMSNorm only to positive scaling;\n"
              "   a negative scale flips the sign of both outputs)\n");
  // Output moments of LN are fixed by construction: mean 0 and variance ~ 1 per row.
  double s = 0, ss = 0;
  for (int j = 0; j < 16; ++j) s += b_ln.at({0, j}), ss += b_ln.at({0, j}) * b_ln.at({0, j});
  const double m = s / 16, var = ss / 16 - m * m;
  std::printf("  LN output row 0: mean %.1e, variance %.4f\n", m, var);
  check::Require(std::abs(m) < 1e-6 && std::abs(var - 1) < 1e-4, "LN output moments");
}

}  // namespace

int main() {
  try {
    LayerAndRms();
    GroupNorms();
    Invariances();
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
