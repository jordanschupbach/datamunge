// CMD: just run-nn weight_spectral_norm
// Weight normalization and spectral normalization.
//
// Port of tflow's weight_spectral_norm. Two reparameterisations of a dense
// layer's weight matrix, from common/reparam.hpp. Weight normalisation writes
// each column as a direction times a gain, W_j = g_j v_j / ||v_j||, which
// decouples the scale of the weights from their direction; the program checks
// that the effective columns have norm g, derives and verifies the gradients
// with respect to g and v, and shows that the v-gradient is orthogonal to v.
// Spectral normalisation divides W by its largest singular value, estimated
// by power iteration; the program follows the estimate converging to the
// value from a full SVD (nn::svd, TF's Svd), verifies that the normalised
// matrix has spectral norm 1 and is therefore 1-Lipschitz, and shows what one
// iteration per training step buys when the weights drift slowly.
//
// The inputs x and r use the same std::mt19937_64 streams as TF; the weights
// come from nn's generator, so the numbers match TF statistically. The port
// turns the printed claims into checks.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>

#include "common/checks.hpp"
#include "common/reparam.hpp"

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
// 1. Weight normalisation: effective weights and gradients
// ---------------------------------------------------------------------------

void WeightNorm() {
  std::printf("== Weight normalisation: W_j = g_j v_j / ||v_j||, in = 5, out = 3\n");
  nn::ParameterStore p(1);
  std::mt19937_64    rng(1);
  const Tensor       xv = Random({4, 5}, rng), rv = Random({4, 3}, rng);
  const auto         wn   = reparam::WeightNormDense(p, "d", xv, 5, 3);
  const Tensor       loss = nn::sum(wn.y * rv, {0, 1});  // dL/dy = r
  const auto         grads = nn::grad(loss, {p.at("d/v"), p.at("d/g"), wn.W});
  const auto W = wn.W.to_vector<float>(), v = p.at("d/v").to_vector<float>(), g = p.at("d/g").to_vector<float>();
  const auto dv = grads[0].to_vector<float>(), dg = grads[1].to_vector<float>(), dW = grads[2].to_vector<float>();
  std::printf("  %-8s %10s %10s %10s %12s %14s\n", "column", "||v_j||", "g_j", "||W_j||", "W_j = v_j?", "v_j . dL/dv_j");
  float err_g = 0, err_dv = 0, err_dot = 0, err_norm = 0;
  for (int j = 0; j < 3; ++j) {
    double nv = 0, nW = 0, same = 0, dot_dW_v = 0, vdv = 0;
    for (int i = 0; i < 5; ++i) {
      const int k = i * 3 + j;
      nv += v[k] * v[k];
      nW += W[k] * W[k];
      same = std::max(same, std::abs(double(W[k]) - v[k]));
      dot_dW_v += dW[k] * v[k];
      vdv += v[k] * dv[k];
    }
    nv = std::sqrt(nv), nW = std::sqrt(nW);
    // Closed forms: dL/dg_j = dL/dW_j . v_j / ||v_j||;  dL/dv_j = (g_j/||v_j||) (dL/dW_j - (dL/dg_j) v_j / ||v_j||).
    const double dgj = dot_dW_v / nv;
    err_g            = std::max<float>(err_g, std::abs(dgj - dg[j]));
    for (int i = 0; i < 5; ++i) {
      const int    k    = i * 3 + j;
      const double dvij = g[j] / nv * (dW[k] - dgj * v[k] / nv);
      err_dv            = std::max<float>(err_dv, std::abs(dvij - dv[k]));
    }
    err_dot  = std::max<float>(err_dot, std::abs(vdv));
    err_norm = std::max<float>(err_norm, std::abs(nW - g[j]));
    std::printf("  %-8d %10.4f %10.4f %10.4f %12.1e %14.1e\n", j, nv, g[j], nW, same, vdv);
  }
  std::printf("  (at initialisation g = ||v||, so W = v; ||W_j|| = g_j always)\n");
  std::printf("  max |dL/dg symbolic - closed form| = %.1e, max |dL/dv symbolic - closed form| = %.1e\n", err_g, err_dv);
  std::printf("  max |v_j . dL/dv_j| = %.1e: the gradient of v is orthogonal to v (||v_j|| cannot change by that step)\n",
              err_dot);
  check::Require(err_norm < 1e-5f, "||W_j|| = g_j");
  check::Require(err_g < 1e-5f && err_dv < 1e-5f, "weight-norm gradients match the closed forms");
  check::Require(err_dot < 1e-5f, "v-gradient orthogonal to v");
}

// ---------------------------------------------------------------------------
// 2. Spectral normalisation: power iteration and the 1-Lipschitz property
// ---------------------------------------------------------------------------

// Largest singular value of a [in, out] matrix via nn::svd, for reference.
float SpectralNormSvd(const Tensor& W) { return nn::svd(W, /*compute_uv=*/false).s.to_vector<float>()[0]; }

void SpectralNorm() {
  std::printf("\n== Spectral normalisation: W / sigma(W), in = 64, out = 32\n");
  std::printf("  power iteration from a random u, sigma estimate after k iterations vs the SVD:\n");
  std::printf("  %4s %12s %12s\n", "k", "sigma_k", "rel. error");
  const Tensor none = nn::zeros({1, 64});
  float        sigma_true = 0, last_error = 1;
  for (int k : {1, 2, 3, 5, 10, 20}) {
    nn::ParameterStore p(2), stats(3);
    const auto         sn = reparam::SpectralNormDense(p, stats, "d", none, 64, 32, k);
    if (k == 1) sigma_true = SpectralNormSvd(p.at("d/W"));
    const float sigma = sn.sigma.item<float>();
    last_error        = std::abs(sigma - sigma_true) / sigma_true;
    std::printf("  %4d %12.6f %12.1e\n", k, sigma, last_error);
    check::Require(sigma <= sigma_true * (1 + 1e-5f), "power iteration underestimates sigma_max");
  }
  std::printf("  SVD: sigma_max = %.6f\n", sigma_true);
  check::Require(last_error < 1e-2f, "power iteration converges to the SVD value");

  // One iteration per step, with the vector u carried over: the usual training setup.
  std::printf("\n  one iteration per step with u persisted (as in training), same W:\n");
  nn::ParameterStore p(2), stats(3);
  std::printf("  %6s %12s %12s\n", "step", "sigma", "rel. error");
  reparam::SpectralNormOut sn;
  for (int step = 1; step <= 20; ++step) {
    sn = reparam::SpectralNormDense(p, stats, "d", none, 64, 32, 1);
    sn.update();
    const float sigma = sn.sigma.item<float>();
    if (step <= 3 || step == 5 || step == 10 || step == 20)
      std::printf("  %6d %12.6f %12.1e\n", step, sigma, std::abs(sigma - sigma_true) / sigma_true);
  }
  // The normalised matrix has spectral norm 1 and shrinks every vector.
  std::mt19937_64 rng(4);
  const Tensor    xv  = Random({1000, 64}, rng);
  sn                  = reparam::SpectralNormDense(p, stats, "d", xv, 64, 32, 1);
  const float sigma_sn = SpectralNormSvd(sn.W);
  const auto  x = xv.to_vector<float>(), y = sn.y.to_vector<float>();
  float       max_ratio = 0;
  for (int i = 0; i < 1000; ++i) {
    double nx = 0, ny = 0;
    for (int j = 0; j < 64; ++j) nx += x[i * 64 + j] * x[i * 64 + j];
    for (int j = 0; j < 32; ++j) ny += y[i * 32 + j] * y[i * 32 + j];
    max_ratio = std::max<float>(max_ratio, std::sqrt(ny / nx));
  }
  std::printf("  sigma_max(W / sigma) by SVD = %.6f; max ||x W_sn|| / ||x|| over 1000 random x = %.4f (<= 1: 1-Lipschitz)\n",
              sigma_sn, max_ratio);
  // Find the direction that attains the bound: the top right-singular vector v.
  std::printf("  (the ratio reaches 1 only along the top singular direction; random x are far from it in 64 dimensions)\n");
  check::Require(std::abs(sigma_sn - 1) < 1e-2f, "normalised spectral norm is 1");
  check::Require(max_ratio <= sigma_sn * (1 + 1e-5f), "spectrally normalised layer is Lipschitz with constant sigma_max");
}

}  // namespace

int main() {
  try {
    WeightNorm();
    SpectralNorm();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
