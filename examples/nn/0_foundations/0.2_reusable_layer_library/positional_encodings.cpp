// CMD: just run-nn positional_encodings
// Positional encodings: sinusoidal, learned, rotary (RoPE) and ALiBi.
//
// Port of tflow's positional_encodings. Attention has no notion of order, so
// position is added to the inputs (sinusoidal or learned tables), rotated into
// the queries and keys (RoPE) or added to the scores (ALiBi). This program
// prints the sinusoidal table and verifies its properties (unit norm, offsets
// expressible as a fixed linear map, dot products that depend only on the
// distance), shows the learned table receiving gradient only at used
// positions, checks RoPE against a hand-written rotation and its
// relative-position property, and prints the ALiBi bias and the recency
// preference it induces.
//
// TF -> datamunge: nn::SinusoidalTable / LearnedEncoding / ApplyRotary /
// AlibiSlopes / AlibiBias are nn::sinusoidal_encoding, nn::learned_encoding,
// nn::apply_rotary (interleaved pairs, tflow's convention), nn::alibi_slopes
// and nn::alibi_bias in datamunge/nn/attention.hpp. Random inputs come from the
// same std::mt19937_64 streams as TF, so every printed value matches TF's
// (the learned table's own values differ, but its gradient does not depend
// on them).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

Tensor Random(const nn::Shape& shape, std::mt19937_64& rng, float std = 1.f) {
  Tensor                          t = nn::empty(shape);
  std::normal_distribution<float> normal(0.f, std);
  float*                          p = t.data_ptr<float>();
  for (std::int64_t i = 0; i < t.numel(); ++i) p[i] = normal(rng);
  return t;
}

// ---------------------------------------------------------------------------
// 1. Sinusoidal encodings
// ---------------------------------------------------------------------------

void Sinusoidal() {
  const int    T = 64, d = 16;
  const Tensor pe = nn::sinusoidal_encoding(T, d);
  const auto   v  = pe.to_vector<float>();
  auto         at = [&](int t, int j) { return v[t * d + j]; };
  std::printf("== Sinusoidal table, d = %d (columns: sin/cos pairs with wavelengths 2 pi * 10000^(2i/d))\n", d);
  std::printf("  wavelengths:");
  for (int i = 0; i < d / 2; ++i) std::printf(" %.1f", 2 * M_PI * std::pow(10000.0, 2.0 * i / d));
  std::printf("\n  %5s", "pos");
  for (int j = 0; j < d; ++j) std::printf(" %6s", j % 2 ? "cos" : "sin");
  std::printf("\n");
  for (int pos : {0, 1, 2, 3, 10, 50}) {
    std::printf("  %5d", pos);
    for (int j = 0; j < d; ++j) std::printf(" %6.3f", at(pos, j));
    std::printf("\n");
  }
  // Norms are all sqrt(d/2).
  double nmin = 1e9, nmax = 0;
  for (int t = 0; t < T; ++t) {
    double n = 0;
    for (int j = 0; j < d; ++j) n += at(t, j) * at(t, j);
    nmin = std::min(nmin, std::sqrt(n)), nmax = std::max(nmax, std::sqrt(n));
  }
  std::printf("  ||PE_t|| in [%.4f, %.4f] (= sqrt(d/2) = %.4f)\n", nmin, nmax, std::sqrt(d / 2.0));
  check::Near(nmin, std::sqrt(d / 2.0), 1e-5, "sinusoidal row norm");
  check::Near(nmax, std::sqrt(d / 2.0), 1e-5, "sinusoidal row norm");
  // PE_{t+k} = M_k PE_t with M_k a rotation of each pair by k theta_i.
  double err = 0;
  for (int k : {1, 5, 20})
    for (int t = 0; t + k < T; ++t)
      for (int i = 0; i < d / 2; ++i) {
        const double theta = k / std::pow(10000.0, 2.0 * i / d);
        const double sn = at(t, 2 * i), cs = at(t, 2 * i + 1);
        const double sn2 = sn * std::cos(theta) + cs * std::sin(theta), cs2 = -sn * std::sin(theta) + cs * std::cos(theta);
        err = std::max(err, std::abs(sn2 - at(t + k, 2 * i)));
        err = std::max(err, std::abs(cs2 - at(t + k, 2 * i + 1)));
      }
  std::printf("  max |PE_{t+k} - M_k PE_t| over k in {1, 5, 20} = %.1e (the shift is a fixed rotation of each pair)\n", err);
  check::Require(err < 1e-6, "sinusoidal shift is not a rotation");
  // Dot products depend only on the offset.
  std::printf("  PE_t . PE_{t+k} for k = 0..8, at t = 0, 7, 30:\n");
  for (int t : {0, 7, 30}) {
    std::printf("    t = %2d:", t);
    for (int k = 0; k <= 8; ++k) {
      double dot = 0, dot0 = 0;
      for (int j = 0; j < d; ++j) dot += at(t, j) * at(t + k, j), dot0 += at(0, j) * at(k, j);
      std::printf(" %6.3f", dot);
      check::Near(dot, dot0, 1e-5, "sinusoidal dot product depends on the offset only");
    }
    std::printf("\n");
  }
}

// ---------------------------------------------------------------------------
// 2. Learned encodings
// ---------------------------------------------------------------------------

void Learned() {
  std::printf("\n== Learned table [max_len 10, d 4] added to a sequence of length 6: gradient reaches only positions 0-5\n");
  nn::ParameterStore p(1);
  const Tensor       table = nn::learned_encoding(p, "enc", 10, 4);
  const Tensor       used  = nn::slice(table, {0, 0}, {6, 4});  // the first T rows
  std::mt19937_64    rng(2);
  const Tensor       x = Random({3, 6, 4}, rng), r = Random({3, 6, 4}, rng);
  const Tensor       y    = x + used;  // broadcasts over the batch
  const Tensor       loss = nn::sum(y * r);
  const Tensor       g    = nn::grad(loss, {p.at("enc/pos")})[0];
  std::printf("  output shape %s; gradient row norms:", nn::shape_str(y.shape()).c_str());
  for (int t = 0; t < 10; ++t) {
    double n = 0;
    for (int j = 0; j < 4; ++j) n += g.at({t, j}) * g.at({t, j});
    std::printf(" %.2f", std::sqrt(n));
    if (t >= 6) check::Require(n == 0, "unused positions received a gradient");
    else check::Require(n > 0, "used positions received no gradient");
  }
  std::printf("\n  (a learned table cannot represent positions beyond max_len; the sinusoidal one extends to any length)\n");
}

// ---------------------------------------------------------------------------
// 3. Rotary position embedding
// ---------------------------------------------------------------------------

void Rotary() {
  const int T = 12, d = 8;
  std::printf("\n== RoPE on q, k [1, %d, %d]\n", T, d);
  std::mt19937_64 rng(3);
  const Tensor    qv = Random({1, T, d}, rng), kv = Random({1, T, d}, rng);
  const Tensor    rq = nn::apply_rotary(qv);
  // Hand rotation of every pair.
  float err = 0, nerr = 0;
  for (int t = 0; t < T; ++t) {
    double n0 = 0, n1 = 0;
    for (int j = 0; j < d / 2; ++j) {
      const double theta = t / std::pow(10000.0, 2.0 * j / d);
      const double a = qv.at({0, t, 2 * j}), b = qv.at({0, t, 2 * j + 1});
      const double ra = a * std::cos(theta) - b * std::sin(theta), rb = a * std::sin(theta) + b * std::cos(theta);
      const double oa = rq.at({0, t, 2 * j}), ob = rq.at({0, t, 2 * j + 1});
      err = std::max<float>(err, std::abs(ra - oa));
      err = std::max<float>(err, std::abs(rb - ob));
      n0 += a * a + b * b, n1 += oa * oa + ob * ob;
    }
    nerr = std::max<float>(nerr, std::abs(n0 - n1));
  }
  std::printf("  max |RoPE(q) - hand rotation| = %.1e; max | ||q_t||^2 - ||RoPE(q)_t||^2 | = %.1e (rotations preserve norms)\n",
              err, nerr);
  check::Require(err < 1e-5 && nerr < 1e-4, "RoPE differs from the hand rotation");
  // Relative position: use the same vectors a, b at different absolute positions; the score depends only on m - n.
  const Tensor same_q = nn::broadcast_to(nn::slice(qv, {0, 0, 0}, {1, 1, d}), {1, T, d});
  const Tensor same_k = nn::broadcast_to(nn::slice(kv, {0, 0, 0}, {1, 1, d}), {1, T, d});
  const Tensor scores = nn::matmul(nn::apply_rotary(same_q), nn::apply_rotary(same_k), false, true);  // [1, T, T]
  std::printf("  the same q and k placed at positions (m, n): RoPE(q)_m . RoPE(k)_n\n");
  for (int offset : {0, 1, 3}) {
    std::printf("    m - n = %d:", offset);
    for (int n = 0; n + offset < T; n += 2) {
      std::printf(" (%2d,%2d) %7.3f", n + offset, n, scores.at({0, n + offset, n}));
      check::Near(scores.at({0, n + offset, n}), scores.at({0, offset, 0}), 1e-4, "RoPE score depends on m - n only");
    }
    std::printf("\n");
  }
  std::printf("  (identical along each row: the score depends on the offset only, not on the absolute positions)\n");
}

// ---------------------------------------------------------------------------
// 4. ALiBi
// ---------------------------------------------------------------------------

void Alibi() {
  const int heads = 4, T = 8;
  std::printf("\n== ALiBi with %d heads, T = %d: slopes", heads, T);
  const auto slopes = nn::alibi_slopes(heads);
  for (double s : slopes) std::printf(" %.4f", s);
  std::printf("\n");
  const Tensor bias    = nn::alibi_bias(heads, T);
  const Tensor weights = nn::softmax(nn::zeros({heads, T, T}) + bias);  // content-free attention: the bias alone
  std::printf("  bias of head 0 (slope %.4f), rows = query position, columns = key position:\n", slopes[0]);
  for (int m = 0; m < T; ++m) {
    std::printf("    ");
    for (int n = 0; n < T; ++n) {
      const double b = bias.at({0, m, n});
      if (b < -1e8) std::printf("    -inf");
      else std::printf(" %7.3f", b);
    }
    std::printf("\n");
  }
  std::printf("  attention weights of the last query with equal content scores (the bias alone), per head:\n");
  for (int h = 0; h < heads; ++h) {
    std::printf("    head %d (slope %.4f):", h, slopes[h]);
    double z = 0;
    for (int n = 0; n < T; ++n) z += std::exp(-slopes[h] * (T - 1 - n));
    for (int n = 0; n < T; ++n) {
      std::printf(" %6.3f", weights.at({h, T - 1, n}));
      check::Near(weights.at({h, T - 1, n}), std::exp(-slopes[h] * (T - 1 - n)) / z, 1e-6, "ALiBi weights");
    }
    std::printf("\n");
  }
  std::printf("  (steep heads look only at recent keys, flat heads at everything; no parameters, and the\n"
              "   penalty is linear in the distance so it extends to lengths never seen in training)\n");
}

}  // namespace

int main() {
  try {
    Sinusoidal();
    Learned();
    Rotary();
    Alibi();
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
