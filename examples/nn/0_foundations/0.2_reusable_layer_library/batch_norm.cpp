// CMD: just run-nn batch_norm
// Batch normalization with fused batch statistics and moving statistics.
//
// Port of tflow's batch_norm. nn::batch_norm (datamunge/nn/layers.hpp)
// standardises each channel over the batch with nn::batch_norm_train (the
// FusedBatchNormV3 equivalent in conv.hpp), applies a learned scale and
// shift, and keeps exponential moving averages of the batch mean and
// variance for inference. The program checks the training output and the two
// statistics outputs against hand computations (and shows which variance
// estimator is reported), watches the moving statistics converge to the
// population values and the inference branch use them, demonstrates how
// batch norm rescues a badly initialised deep ReLU stack (activation scale
// and gradient norm), and verifies the backward pass through the batch
// statistics against the closed-form batch-norm gradient.
//
// TF -> datamunge: TF builds both branches in one graph and runs the moving
// average update as a separate op; here nn::batch_norm takes a `training`
// flag and, when training, updates "<name>/moving_mean" / "<name>/moving_var"
// in the stats store immediately. Section 1 therefore runs the inference
// branch before the training pass, and reads the batch statistics from
// nn::batch_norm_train. The data use the same std::mt19937_64 streams as TF,
// so sections 1, 2 and 4 print the same numbers; the deep-stack weights come
// from datamunge's generator, so section 3 matches statistically.

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

constexpr double kEps = 1e-3;

Tensor RandomMatrix(int n, int c, std::mt19937_64& rng, const std::vector<float>& mean, const std::vector<float>& std) {
  std::vector<float>              v(static_cast<std::size_t>(n) * c);
  std::normal_distribution<float> normal(0.f, 1.f);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < c; ++j) v[static_cast<std::size_t>(i) * c + j] = mean[j] + std[j] * normal(rng);
  return nn::from_vector(v, {n, c});
}

nn::BatchNormOptions Options(double momentum) {
  nn::BatchNormOptions o;
  o.momentum = momentum, o.eps = kEps;
  return o;
}

// ---------------------------------------------------------------------------
// 1. Training output and statistics against a hand computation
// ---------------------------------------------------------------------------

void HandCheck() {
  std::printf("== Training pass on x [8, 4]: y = (x - mu_B) / sqrt(var_B + eps), eps = %g\n", kEps);
  nn::ParameterStore p, stats;
  std::mt19937_64    rng(1);
  const Tensor       xv = RandomMatrix(8, 4, rng, {1, -2, 0, 5}, {1, 3, 0.1f, 2});
  // Inference first: the moving statistics are still (0, 1).
  const Tensor infer = nn::batch_norm(p, stats, "bn1", xv, false, Options(0.9));
  const Tensor train = nn::batch_norm(p, stats, "bn1", xv, true, Options(0.9));
  const nn::BatchNormResult fused = nn::batch_norm_train(xv, p.at("bn1/gamma"), p.at("bn1/beta"), kEps);
  const int N = 8, C = 4;
  float     err_y = 0, err_mean = 0, err_biased = 0, err_unbiased = 0, err_infer = 0, err_layer = 0, err_update = 0;
  std::printf("  %-8s %10s %12s %12s %14s\n", "channel", "mean_B", "var_B biased", "var_B N/(N-1)", "op batch_var");
  for (int c = 0; c < C; ++c) {
    double mu = 0, var = 0;
    for (int i = 0; i < N; ++i) mu += static_cast<float>(xv.at({i, c}));
    mu /= N;
    for (int i = 0; i < N; ++i) var += (static_cast<float>(xv.at({i, c})) - mu) * (static_cast<float>(xv.at({i, c})) - mu);
    var /= N;
    for (int i = 0; i < N; ++i) {
      const double x  = static_cast<float>(xv.at({i, c}));
      const double yh = (x - mu) / std::sqrt(var + kEps);
      err_y     = std::max<float>(err_y, std::abs(yh - fused.y.at({i, c})));
      err_layer = std::max<float>(err_layer, std::abs(train.at({i, c}) - fused.y.at({i, c})));
      err_infer = std::max<float>(err_infer, std::abs(x / std::sqrt(1 + kEps) - infer.at({i, c})));
    }
    const double bv = fused.variance.at({c});
    err_mean        = std::max<float>(err_mean, std::abs(mu - fused.mean.at({c})));
    err_biased      = std::max<float>(err_biased, std::abs(var - bv));
    err_unbiased    = std::max<float>(err_unbiased, std::abs(var * N / (N - 1) - bv));
    // The layer's update: mu <- 0.9 * 0 + 0.1 mu_B, var <- 0.9 * 1 + 0.1 var_B (unbiased).
    err_update = std::max<float>(err_update, std::abs(stats.at("bn1/moving_mean").at({c}) - 0.1 * fused.mean.at({c})));
    err_update = std::max<float>(err_update, std::abs(stats.at("bn1/moving_var").at({c}) - (0.9 + 0.1 * bv)));
    std::printf("  %-8d %10.4f %12.4f %12.4f %14.4f\n", c, mu, var, var * N / (N - 1), bv);
  }
  std::printf("  max |y - hand| = %.1e, |batch_mean - hand| = %.1e\n", err_y, err_mean);
  std::printf("  |batch_var - biased| = %.1e, |batch_var - unbiased| = %.1e  => the op reports the unbiased variance\n",
              err_biased, err_unbiased);
  std::printf("  inference branch with fresh moving stats (0, 1): max |y - x / sqrt(1 + eps)| = %.1e\n", err_infer);
  std::printf("  layer (training) vs batch_norm_train: %.1e; moving-average update vs hand: %.1e\n", err_layer, err_update);
  // TF: 3.3e-07, 1.8e-07, biased 1.5e+00, unbiased 9.2e-07, infer 9.5e-07.
  check::Require(err_y < 1e-5f && err_mean < 1e-5f, "batch-norm output / mean disagree with the hand computation");
  check::Require(err_unbiased < 1e-5f && err_biased > 0.1f, "batch_var should be the unbiased variance");
  check::Require(err_infer < 1e-5f, "inference branch with initial statistics");
  check::Require(err_layer == 0 && err_update < 1e-6f, "layer output / moving-average update");
}

// ---------------------------------------------------------------------------
// 2. Moving statistics and the inference branch
// ---------------------------------------------------------------------------

void MovingStatistics(int steps) {
  std::printf(
      "\n== Moving statistics, momentum 0.9, batches of 32 from N(m_c, s_c^2) with m = (1, 2, 3, 4), s = (1, 2, 0.5, "
      "3)\n");
  nn::ParameterStore       p, stats;
  std::mt19937_64          rng(2);
  const std::vector<float> mean = {1, 2, 3, 4}, std = {1, 2, 0.5f, 3};
  std::vector<float>       mm, mv;
  std::printf("  %6s | %-31s | %-31s\n", "step", "moving mean", "moving var");
  for (int step = 1; step <= steps; ++step) {
    nn::NoGradGuard g;
    nn::batch_norm(p, stats, "bn1", RandomMatrix(32, 4, rng, mean, std), true, Options(0.9));
    if (step == 1 || step == 10 || step == 30 || step == 100 || step == steps) {
      mm = stats.at("bn1/moving_mean").to_vector<float>(), mv = stats.at("bn1/moving_var").to_vector<float>();
      std::printf("  %6d |", step);
      for (int c = 0; c < 4; ++c) std::printf(" %7.3f", mm[c]);
      std::printf(" |");
      for (int c = 0; c < 4; ++c) std::printf(" %7.3f", mv[c]);
      std::printf("\n");
    }
  }
  std::printf("  %6s |   1.000   2.000   3.000   4.000 |   1.000   4.000   0.250   9.000\n", "target");
  // TF prints exactly these values at step 300 (same data stream).
  const float tf_mean[4] = {1.021f, 2.041f, 2.990f, 4.014f}, tf_var[4] = {1.102f, 4.113f, 0.234f, 8.317f};
  if (steps == 300) for (int c = 0; c < 4; ++c) {
    check::Near(mm[c], tf_mean[c], 2e-3, "moving mean vs TF at step 300");
    check::Near(mv[c], tf_var[c], 2e-3, "moving var vs TF at step 300");
  }
  // A fresh batch through the inference branch is standardised with the moving stats.
  const Tensor              fresh = RandomMatrix(4096, 4, rng, mean, std);
  const std::vector<float>  y     = nn::batch_norm(p, stats, "bn1", fresh, false, Options(0.9)).to_vector<float>();
  const float               tf_m[4] = {-0.041f, -0.021f, 0.028f, 0.002f}, tf_s[4] = {0.963f, 0.999f, 1.027f, 1.046f};
  std::printf("  inference on 4096 fresh rows: per-channel mean / std of the output =");
  for (int c = 0; c < 4; ++c) {
    double s = 0, ss = 0;
    for (int i = 0; i < 4096; ++i) s += y[i * 4 + c], ss += double(y[i * 4 + c]) * y[i * 4 + c];
    const double m = s / 4096, sd = std::sqrt(ss / 4096 - m * m);
    std::printf("  %.3f / %.3f", m, sd);
    if (steps == 300) {
      check::Near(m, tf_m[c], 2e-3, "inference output mean vs TF");
      check::Near(sd, tf_s[c], 2e-3, "inference output std vs TF");
    }
    for (int i = 0; i < 4096; ++i)
      check::Near(y[i * 4 + c], (fresh.at({i, c}) - mm[c]) / std::sqrt(mv[c] + kEps), 1e-5,
                  "inference uses the current moving statistics");
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// 3. A badly initialised deep ReLU stack, with and without batch norm
// ---------------------------------------------------------------------------

void DeepStack() {
  std::printf(
      "\n== 24-layer ReLU stack, width 256, weights N(0, 0.25 / fan_in) (half the LeCun std): activation std and "
      "gradient norm\n");
  const int depth = 24, width = 256, batch = 128;
  for (bool use_bn : {false, true}) {
    nn::ParameterStore p(3), stats;
    std::mt19937_64    rng(4);
    const Tensor       x = RandomMatrix(batch, width, rng, std::vector<float>(width, 0.f), std::vector<float>(width, 1.f));
    Tensor             h = x;
    std::vector<Tensor> acts;
    for (int l = 0; l < depth; ++l) {
      const Tensor W = p.get("l" + std::to_string(l) + "/W", {width, width}, nn::init::normal(0.5 / std::sqrt(width)));
      Tensor       z = nn::matmul(h, W);
      if (use_bn) z = nn::batch_norm(p, stats, "bn" + std::to_string(l), z, true, Options(0.99));
      h = nn::relu(z);
      acts.push_back(h);
    }
    const Tensor loss  = nn::mean(nn::square(h));
    const Tensor g     = nn::grad(loss, {p.at("l0/W")})[0];
    const double gnorm = nn::sqrt(nn::sum(nn::square(g))).item<float>();
    std::printf("  %-12s", use_bn ? "with BN" : "without BN");
    double last = 0;
    for (int l = 1; l <= depth; ++l) {
      if (!(l == 1 || l % 4 == 0)) continue;
      double ss = 0;
      for (float v : acts[l - 1].to_vector<float>()) ss += double(v) * v;
      last = std::sqrt(ss / (batch * width));
      std::printf("  L%-2d %.2e", l, last);
    }
    std::printf("  | ||dL/dW_1|| = %.2e\n", gnorm);
    // TF: without BN L24 1.3e-11 and gradient 0; with BN ~0.70 at every depth, gradient 9.3.
    if (use_bn) {
      check::Near(last, 0.70, 0.05, "with BN the activation scale stays ~0.70");
      check::Require(gnorm > 1 && gnorm < 100, "with BN the first-layer gradient is O(1)");
    } else {
      check::Require(last < 1e-8 && gnorm < 1e-12, "without BN the signal and gradient vanish");
    }
  }
}

// ---------------------------------------------------------------------------
// 4. Backward pass through the batch statistics
// ---------------------------------------------------------------------------

void Backward() {
  std::printf("\n== Gradient through the batch statistics: symbolic vs closed form vs finite differences\n");
  nn::ParameterStore p, stats;
  const int          N = 6, C = 3;
  std::mt19937_64    rng(5);
  Tensor             xv = RandomMatrix(N, C, rng, {0, 1, -1}, {1, 2, 0.5f});
  const Tensor       rv = RandomMatrix(N, C, rng, {0, 0, 0}, {1, 1, 1});
  auto loss_of = [&](const Tensor& x) { return nn::sum(nn::batch_norm(p, stats, "bn1", x, true, Options(0.9)) * rv); };
  xv.requires_grad_();
  const Tensor              loss  = loss_of(xv);  // dL/dy = r
  const std::vector<Tensor> grads = nn::grad(loss, {xv, p.at("bn1/gamma"), p.at("bn1/beta")});
  const std::vector<float>  x = xv.to_vector<float>(), r = rv.to_vector<float>(), dx = grads[0].to_vector<float>();
  // Closed form with gamma = 1: dx = (1/(N sigma)) (N dy - sum dy - xhat sum(dy xhat)).
  float err_x = 0, err_gamma = 0, err_beta = 0;
  for (int c = 0; c < C; ++c) {
    double mu = 0, var = 0, sdy = 0, sdyx = 0;
    for (int i = 0; i < N; ++i) mu += x[i * C + c];
    mu /= N;
    for (int i = 0; i < N; ++i) var += (x[i * C + c] - mu) * (x[i * C + c] - mu);
    var /= N;
    const double        sigma = std::sqrt(var + kEps);
    std::vector<double> xhat(N);
    for (int i = 0; i < N; ++i) {
      xhat[i] = (x[i * C + c] - mu) / sigma;
      sdy += r[i * C + c], sdyx += r[i * C + c] * xhat[i];
    }
    for (int i = 0; i < N; ++i) {
      const double d = (N * r[i * C + c] - sdy - xhat[i] * sdyx) / (N * sigma);
      err_x          = std::max<float>(err_x, std::abs(d - dx[i * C + c]));
    }
    err_gamma = std::max<float>(err_gamma, std::abs(sdyx - grads[1].at({c})));  // dL/dgamma = sum dy xhat
    err_beta  = std::max<float>(err_beta, std::abs(sdy - grads[2].at({c})));    // dL/dbeta = sum dy
  }
  // Finite differences on x (float32).
  const float     eps = 1e-2f;
  float           err_fd = 0, gmax = 0;
  nn::NoGradGuard no_grad;
  for (int i = 0; i < N * C; ++i) {
    std::vector<float> xp = x, xm = x;
    xp[i] += eps, xm[i] -= eps;
    const float lp = loss_of(nn::from_vector(xp, {N, C})).item<float>();
    const float lm = loss_of(nn::from_vector(xm, {N, C})).item<float>();
    err_fd         = std::max(err_fd, std::abs((lp - lm) / (2 * eps) - dx[i]));
    gmax           = std::max(gmax, std::abs(dx[i]));
  }
  std::printf(
      "  max |dL/dx symbolic - closed form| = %.1e, |dL/dgamma - sum(dy xhat)| = %.1e, |dL/dbeta - sum(dy)| = %.1e\n",
      err_x, err_gamma, err_beta);
  std::printf("  max |dL/dx symbolic - central difference (eps = %.0e)| = %.1e (%.1e relative to max |dL/dx| = %.2f)\n",
              eps, err_fd, err_fd / gmax, gmax);
  // The column sums of dL/dx vanish: BN output is invariant to a per-channel shift of x.
  std::printf("  column sums of dL/dx (invariance to shifting a channel):");
  float max_col = 0;
  for (int c = 0; c < C; ++c) {
    double s = 0;
    for (int i = 0; i < N; ++i) s += dx[i * C + c];
    std::printf(" %.1e", s);
    max_col = std::max<float>(max_col, std::abs(s));
  }
  std::printf("\n");
  // TF: 4.9e-07, 6.5e-07, 1.1e-07; FD 6.9e-05 (2.9e-05 relative, max 2.40); column sums <= 1.4e-06.
  check::Require(err_x < 1e-5f && err_gamma < 1e-5f && err_beta < 1e-5f, "batch-norm gradient vs closed form");
  check::Require(err_fd / gmax < 1e-3f, "batch-norm gradient vs finite differences");
  check::Near(gmax, 2.40, 0.01, "max |dL/dx| vs TF");
  check::Require(max_col < 1e-5f, "column sums of dL/dx should vanish");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int steps = 300;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument.rfind("epochs=", 0) == 0) steps = std::stoi(argument.substr(7));
      if (argument.rfind("steps=", 0) == 0) steps = std::stoi(argument.substr(6));
    }
    check::Require(steps > 0, "positive statistics-update budget required");
    HandCheck();
    MovingStatistics(steps);
    DeepStack();
    Backward();
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
