// CMD: just run-nn activations
// Activation functions: ReLU, LeakyReLU, ELU, SELU, GELU, SiLU, Mish, softplus and friends.
//
// Port of tflow's activations. nn::activate (datamunge/nn/layers.hpp) maps an
// Activation enum to the ops of each non-linearity. This program evaluates
// every activation and its derivative on a grid, computing the derivative two
// ways: by autograd (nn::grad, TF: AddSymbolicGradients) and from the
// closed-form expression written out in C++, and reports the largest
// disagreement. It then summarises the qualitative properties (range,
// saturation, zero mean) and demonstrates the self-normalising behaviour of
// SELU with LeCun-normal weights.
//
// TF -> datamunge: no graph/session; the grid is an ordinary tensor with
// requires_grad. The grid values, the N(0, 1) samples and the deep-stack
// inputs use the same std::mt19937_64 streams as TF, so the first two tables
// match digit for digit; the deep-stack weights come from datamunge's
// generator, so that table matches TF statistically.

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

constexpr double kSeluLambda = 1.0507009873554804934193349852946;
constexpr double kSeluAlpha  = 1.6732632423543772848170429916717;
constexpr float  kLeakyAlpha = 0.2f;
const double     kPi         = std::acos(-1.0);

double Sigmoid(double x) { return 1 / (1 + std::exp(-x)); }
double Softplus(double x) { return x > 30 ? x : std::log1p(std::exp(x)); }
double NormalPdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2 * kPi); }
double NormalCdf(double x) { return 0.5 * (1 + std::erf(x / std::sqrt(2.0))); }

// Closed-form f(x) and f'(x) of each activation.
double F(Activation a, double x) {
  switch (a) {
    case Activation::None: return x;
    case Activation::Relu: return std::max(x, 0.0);
    case Activation::Relu6: return std::min(std::max(x, 0.0), 6.0);
    case Activation::LeakyRelu: return x > 0 ? x : kLeakyAlpha * x;
    case Activation::Elu: return x > 0 ? x : std::expm1(x);
    case Activation::Selu: return kSeluLambda * (x > 0 ? x : kSeluAlpha * std::expm1(x));
    case Activation::Gelu: return x * NormalCdf(x);
    case Activation::GeluTanh: return 0.5 * x * (1 + std::tanh(std::sqrt(2 / kPi) * (x + 0.044715 * x * x * x)));
    case Activation::Silu: return x * Sigmoid(x);
    case Activation::Mish: return x * std::tanh(Softplus(x));
    case Activation::Softplus: return Softplus(x);
    case Activation::Softsign: return x / (1 + std::abs(x));
    case Activation::Sigmoid: return Sigmoid(x);
    case Activation::Tanh: return std::tanh(x);
  }
  return x;
}
double dF(Activation a, double x) {
  switch (a) {
    case Activation::None: return 1;
    case Activation::Relu: return x > 0 ? 1 : 0;
    case Activation::Relu6: return x > 0 && x < 6 ? 1 : 0;
    case Activation::LeakyRelu: return x > 0 ? 1 : kLeakyAlpha;
    case Activation::Elu: return x > 0 ? 1 : std::exp(x);
    case Activation::Selu: return kSeluLambda * (x >= 0 ? 1 : kSeluAlpha * std::exp(x));  // TF uses the linear branch at 0
    case Activation::Gelu: return NormalCdf(x) + x * NormalPdf(x);
    case Activation::GeluTanh: {
      const double u = std::sqrt(2 / kPi) * (x + 0.044715 * x * x * x), du = std::sqrt(2 / kPi) * (1 + 3 * 0.044715 * x * x);
      const double th = std::tanh(u);
      return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * du;
    }
    case Activation::Silu: {
      const double s = Sigmoid(x);
      return s * (1 + x * (1 - s));
    }
    case Activation::Mish: {
      const double sp = Softplus(x), th = std::tanh(sp);
      return th + x * (1 - th * th) * Sigmoid(x);
    }
    case Activation::Softplus: return Sigmoid(x);
    case Activation::Softsign: return 1 / ((1 + std::abs(x)) * (1 + std::abs(x)));
    case Activation::Sigmoid: {
      const double s = Sigmoid(x);
      return s * (1 - s);
    }
    case Activation::Tanh: {
      const double t = std::tanh(x);
      return 1 - t * t;
    }
  }
  return 1;
}

const std::vector<Activation> kAll = {Activation::Relu,     Activation::Relu6,    Activation::LeakyRelu,
                                      Activation::Elu,      Activation::Selu,     Activation::Gelu,
                                      Activation::GeluTanh, Activation::Silu,     Activation::Mish,
                                      Activation::Softplus, Activation::Softsign, Activation::Sigmoid,
                                      Activation::Tanh};

// ---------------------------------------------------------------------------
// 1. Values and derivatives on a grid, autograd vs closed form
// ---------------------------------------------------------------------------

void Table() {
  const std::vector<float> grid = {-4, -2, -1, -0.5f, 0, 0.5f, 1, 2, 4};
  std::vector<std::vector<float>> ys, dys;
  for (Activation a : kAll) {
    Tensor x = nn::from_vector(grid).requires_grad_();
    Tensor y = nn::activate(x, a, kLeakyAlpha);
    // y is elementwise, so the gradient of sum(y) w.r.t. x is f'(x) elementwise.
    Tensor dy = nn::grad(nn::sum(y), {x})[0];
    ys.push_back(y.to_vector<float>()), dys.push_back(dy.to_vector<float>());
  }

  std::printf("== f(x) on the grid (rows: activation, columns: x)\n  %-11s", "");
  for (float g : grid) std::printf(" %7.1f", g);
  std::printf("\n");
  for (size_t k = 0; k < kAll.size(); ++k) {
    std::printf("  %-11s", nn::activation_name(kAll[k]).c_str());
    for (size_t i = 0; i < grid.size(); ++i) std::printf(" %7.3f", ys[k][i]);
    std::printf("\n");
  }
  std::printf("\n== f'(x) by autograd, and the largest gap to the closed form\n  %-11s", "");
  for (float g : grid) std::printf(" %7.1f", g);
  std::printf("   max|dy - f'|  max|y - f|\n");
  for (size_t k = 0; k < kAll.size(); ++k) {
    std::printf("  %-11s", nn::activation_name(kAll[k]).c_str());
    double err_d = 0, err_f = 0;
    for (size_t i = 0; i < grid.size(); ++i) {
      const float d = dys[k][i];
      std::printf(" %7.3f", d);
      err_d = std::max(err_d, std::abs(d - dF(kAll[k], grid[i])));
      err_f = std::max(err_f, std::abs(ys[k][i] - F(kAll[k], grid[i])));
    }
    std::printf("   %9.1e    %9.1e\n", err_d, err_f);
    // TF: every gap is below 1e-6 (gelu_tanh 9.3e-07 is the largest).
    check::Require(err_d < 2e-6 && err_f < 2e-6,
                   nn::activation_name(kAll[k]) + ": autograd / value disagrees with the closed form");
  }
  std::printf("  (ReLU's derivative at exactly 0 is taken as 0, as in TensorFlow; the closed form agrees)\n");
}

// ---------------------------------------------------------------------------
// 2. Qualitative properties
// ---------------------------------------------------------------------------

void Properties() {
  std::printf("\n== Properties on x ~ N(0, 1), 200000 samples\n");
  std::printf("  %-11s %9s %9s %10s %10s %12s %10s\n", "", "min f", "max f", "mean f", "std f", "P(f'(x)=0)", "max f'");
  std::mt19937_64                  rng(1);
  std::normal_distribution<double> normal(0, 1);
  std::vector<double>              xs(200000);
  for (double& v : xs) v = normal(rng);
  for (Activation a : kAll) {
    double lo = 1e9, hi = -1e9, sum = 0, sq = 0, dead = 0, maxd = 0;
    for (double v : xs) {
      const double f = F(a, v), d = dF(a, v);
      lo = std::min(lo, f), hi = std::max(hi, f), sum += f, sq += f * f;
      dead += d == 0 ? 1 : 0, maxd = std::max(maxd, d);
    }
    const double mean = sum / xs.size();
    const double sd   = std::sqrt(sq / xs.size() - mean * mean);
    std::printf("  %-11s %9.3f %9.3f %10.3f %10.3f %12.3f %10.3f\n", nn::activation_name(a).c_str(), lo, hi, mean, sd,
                dead / xs.size(), maxd);
    if (a == Activation::Selu) {  // TF: mean 0.005, std 1.003
      check::Near(mean, 0.0, 0.02, "SELU mean on N(0,1)");
      check::Near(sd, 1.0, 0.02, "SELU std on N(0,1)");
    }
    if (a == Activation::Relu) check::Near(dead / xs.size(), 0.5, 0.01, "ReLU dead fraction");
  }
}

// ---------------------------------------------------------------------------
// 3. SELU self-normalisation through a deep stack
// ---------------------------------------------------------------------------

void SeluDepth() {
  std::printf("\n== Mean / std of activations through 30 layers (width 256, LeCun-normal weights, no bias)\n");
  constexpr int   depth = 30, width = 256, batch = 256;
  nn::NoGradGuard no_grad;  // forward only
  for (Activation act : {Activation::Selu, Activation::Elu, Activation::Tanh, Activation::Relu}) {
    nn::ParameterStore p(5);
    Tensor             xv = nn::empty({batch, width});
    {
      std::mt19937_64                 rng(2);
      std::normal_distribution<float> normal(0.f, 1.f);
      float*                          px = xv.data_ptr<float>();
      for (int i = 0; i < batch * width; ++i) px[i] = normal(rng);
    }
    Tensor h = xv;
    std::vector<double> means, stds;
    for (int l = 0; l < depth; ++l) {
      nn::DenseOptions o;
      o.init = nn::init::lecun_normal(), o.bias = false, o.activation = act;
      h               = nn::dense(p, "l" + std::to_string(l), h, width, o);
      const Tensor hd = h.to(nn::DType::Float64);
      const double m  = nn::mean(hd).item();
      means.push_back(m), stds.push_back(std::sqrt(nn::mean(nn::square(hd)).item() - m * m));
    }
    std::printf("  %-6s", nn::activation_name(act).c_str());
    for (int l : {1, 2, 5, 10, 20, 30}) std::printf("  L%-2d mean %6.3f std %5.3f |", l, means[l - 1], stds[l - 1]);
    std::printf("\n");
    // TF: selu std 0.96-1.01 at every reported depth; elu 0.079, tanh 0.121, relu 0.000 at L30.
    if (act == Activation::Selu)
      for (int l = 0; l < depth; ++l) {
        check::Require(std::abs(means[l]) < 0.1, "SELU mean drifts");
        check::Require(stds[l] > 0.85 && stds[l] < 1.15, "SELU std drifts");
      }
    else
      check::Require(stds[depth - 1] < 0.25, nn::activation_name(act) + " should lose scale with depth");
  }
}

}  // namespace

int main() {
  try {
    Table();
    Properties();
    SeluDepth();
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
