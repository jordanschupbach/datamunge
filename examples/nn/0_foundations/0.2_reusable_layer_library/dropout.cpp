// CMD: just run-nn dropout
// Dropout with a train / inference switch.
//
// Port of tflow's dropout. nn::dropout (datamunge/nn/layers.hpp) implements
// inverted dropout: every unit is zeroed with probability p and the survivors
// are divided by 1 - p, so the expected activation is unchanged and inference
// (training = false) is the plain identity. The program verifies the mask
// statistics, the expectation and variance of the output, the exact identity
// at inference, then trains a small MLP on a few noisy points with and
// without dropout to show the regularisation effect, and finally uses Monte
// Carlo dropout at test time as a crude uncertainty estimate.
//
// TF -> datamunge: TF feeds keep = 1 - p through a placeholder so one graph
// serves both modes; here the rate and a `training` flag are plain arguments
// of nn::dropout, and the masks come from an explicit nn::Generator (one per
// dropout layer, seeded like TF's per-op seeds). The training data use the
// same std::mt19937_64 stream as TF; weights and masks come from datamunge's
// generator, so the fits match TF statistically, not digit for digit.

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
using nn::Activation;
using nn::Tensor;

namespace {

// ---------------------------------------------------------------------------
// 1. Mask statistics, expectation, variance, and the inference identity
// ---------------------------------------------------------------------------

void Statistics() {
  std::printf("== Inverted dropout on x = 1 (n = 100000 units), keep = 1 - p\n");
  nn::Generator gen(7);
  const int     n = 100000;
  const Tensor  x = nn::ones({1, n});
  std::printf("  %6s %10s %10s %10s %14s %14s\n", "p", "P(y=0)", "mean y", "var y", "expect var", "values seen");
  for (float p : {0.f, 0.1f, 0.25f, 0.5f, 0.8f}) {
    const float              q = 1 - p;
    const std::vector<float> y = nn::dropout(x, p, true, gen).to_vector<float>();
    int                      zeros = 0;
    double                   sum = 0, sq = 0, maxv = 0;
    for (float v : y) zeros += v == 0, sum += v, sq += double(v) * v, maxv = std::max<double>(maxv, v);
    const double mean = sum / n, var = sq / n - mean * mean, pz = static_cast<double>(zeros) / n;
    std::printf("  %6.2f %10.4f %10.4f %10.4f %14.4f   {0, %.3f}\n", p, pz, mean, var, p / q, maxv);
    // TF: P(y=0) within 0.0012 of p, mean within 0.0024 of 1, var within 0.005 of p/q.
    check::Near(pz, p, 0.006, "fraction of dropped units");
    check::Near(mean, 1.0, 0.02, "dropout keeps the mean");
    check::Near(var, p / q, 0.03 * (1 + p / q), "dropout variance");
    check::Near(maxv, 1.0 / q, 1e-5, "survivors are scaled by 1/keep");
  }
  // Inference is exactly the identity, bit for bit.
  Tensor xr = nn::empty({1, n});
  {
    std::mt19937_64                 rng(0);
    std::normal_distribution<float> normal(0.f, 1.f);
    for (int i = 0; i < n; ++i) xr.data_ptr<float>()[i] = normal(rng);
  }
  const Tensor yr   = nn::dropout(xr, 0.5, false, gen);
  int          same = 0;
  for (int i = 0; i < n; ++i) same += yr.data_ptr<float>()[i] == xr.data_ptr<float>()[i];
  std::printf("  training = false (inference): %d of %d outputs identical to the input\n", same, n);
  check::Require(same == n, "inference dropout must be the identity");
}

// ---------------------------------------------------------------------------
// 2. Regularisation: an MLP fit to 40 noisy points, with and without dropout
// ---------------------------------------------------------------------------

struct Fit {
  float              train_mse, test_mse;
  std::vector<float> mc_std;  // MC-dropout predictive std on the test grid
};

Tensor Column(const std::vector<float>& v) { return nn::from_vector(v, {static_cast<std::int64_t>(v.size()), 1}); }

Fit TrainMlp(float p, std::uint64_t seed, const std::vector<float>& xtr, const std::vector<float>& ytr,
             const std::vector<float>& xte, const std::vector<float>& yte, int steps, bool mc) {
  nn::ParameterStore params(seed);
  nn::Generator      drop1(seed * 10 + 1), drop2(seed * 10 + 2);
  nn::DenseOptions   hidden;
  hidden.activation = Activation::Relu, hidden.init = nn::init::he_uniform();
  auto forward = [&](const Tensor& x, bool training) {
    Tensor h = nn::dense(params, "h1", x, 256, hidden);
    h        = nn::dropout(h, p, training, drop1);
    h        = nn::dense(params, "h2", h, 256, hidden);
    h        = nn::dropout(h, p, training, drop2);
    return nn::dense(params, "out", h, 1);
  };
  const Tensor X = Column(xtr), T = Column(ytr), Xte = Column(xte), Tte = Column(yte);
  { nn::NoGradGuard g; forward(X, false); }  // create the parameters before the optimizer
  nn::Optimizer adam(params, 1e-3, {nn::OptimizerKind::Adam});
  for (int step = 0; step < steps; ++step) adam.minimize(nn::mse(forward(X, true), T));
  nn::NoGradGuard no_grad;
  Fit             f;
  f.train_mse = nn::mse(forward(X, false), T).item<float>();
  f.test_mse  = nn::mse(forward(Xte, false), Tte).item<float>();
  if (mc) {  // Monte Carlo dropout: keep the mask on at test time and average
    const int           samples = 100;
    std::vector<double> s(xte.size(), 0), ss(xte.size(), 0);
    for (int k = 0; k < samples; ++k) {
      const std::vector<float> y = forward(Xte, true).to_vector<float>();
      for (size_t i = 0; i < xte.size(); ++i) s[i] += y[i], ss[i] += double(y[i]) * y[i];
    }
    for (size_t i = 0; i < xte.size(); ++i)
      f.mc_std.push_back(static_cast<float>(
          std::sqrt(std::max(0.0, ss[i] / samples - (s[i] / samples) * (s[i] / samples)))));
  }
  return f;
}

void Regularisation(int steps) {
  std::printf("\n== MLP 1-256-256-1 (ReLU) fit to 40 points of sin(3x) + N(0, 0.3^2) on [-1, 1], %d Adam steps\n",
              steps);
  std::mt19937_64                       rng(42);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  std::normal_distribution<float>       noise(0.f, 0.3f);
  std::vector<float>                    xtr, ytr, xte, yte;
  for (int i = 0; i < 40; ++i) {
    const float xv = U(rng);
    xtr.push_back(xv), ytr.push_back(std::sin(3 * xv) + noise(rng));
  }
  for (int i = 0; i < 201; ++i) {
    const float xv = -1.5f + 3.f * i / 200;  // extends beyond the training range
    xte.push_back(xv), yte.push_back(std::sin(3 * xv));
  }
  std::printf("  %6s %7s %14s %14s   (3 seeds)\n", "p", "seed", "train MSE", "test MSE [-1,1]");
  // Test error inside the training range only (the extrapolation region is for MC dropout).
  std::vector<float> xin, yin;
  for (size_t i = 0; i < xte.size(); ++i)
    if (std::abs(xte[i]) <= 1) xin.push_back(xte[i]), yin.push_back(yte[i]);
  float mean_train[2], mean_test[2];
  int   k = 0;
  for (float p : {0.f, 0.5f}) {
    float tr = 0, te = 0;
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      Fit f = TrainMlp(p, seed, xtr, ytr, xin, yin, steps, false);
      std::printf("  %6.2f %7llu %14.4f %14.4f\n", p, static_cast<unsigned long long>(seed), f.train_mse, f.test_mse);
      tr += f.train_mse / 3, te += f.test_mse / 3;
    }
    std::printf("  %6.2f %7s %14.4f %14.4f\n", p, "mean", tr, te);
    mean_train[k] = tr, mean_test[k] = te, ++k;
  }
  std::printf("  (noise variance 0.09 is the floor for the train MSE of a model that does not memorise noise)\n");
  std::printf("\n== Monte Carlo dropout (p = 0.5, 100 stochastic forward passes): predictive std along x\n");
  const Fit last = TrainMlp(0.5f, 1, xtr, ytr, xte, yte, steps, true);
  std::printf("  %8s", "x:");
  for (int i = 0; i <= 200; i += 20) std::printf(" %6.2f", xte[i]);
  std::printf("\n  %8s", "std:");
  for (int i = 0; i <= 200; i += 20) std::printf(" %6.3f", last.mc_std[i]);
  std::printf("\n  (training data cover [-1, 1]; the spread grows where there is no data)\n");
  // TF (3000 steps): p=0 train 0.060 / test 0.080; p=0.5 train 0.088 / test 0.049;
  // MC std 0.315 / 0.254 at x = -1.5 / 1.5 vs 0.147 at x = 0.
  check::Require(mean_train[0] < mean_train[1], "dropout should fit the training noise less closely");
  check::Require(mean_test[1] < mean_test[0], "dropout should generalise better on [-1, 1]");
  check::Require(mean_test[1] < 0.1f, "dropout test MSE");
  check::Require(last.mc_std[0] > last.mc_std[100] && last.mc_std[200] > last.mc_std[100],
                 "MC-dropout spread should grow outside the data");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int steps = 3000;  // steps=N overrides the Adam budget
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]).rfind("steps=", 0) == 0) steps = std::stoi(std::string(argv[i]).substr(6));
    Statistics();
    Regularisation(steps);
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
