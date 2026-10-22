// CMD: just run-nn dense_layer
// Dense layer with activation, bias and initialiser options.
//
// Port of tflow's dense_layer. nn::dense (datamunge/nn/layers.hpp) computes
// y = act(x W + b) with its weights created in a ParameterStore under
// "<name>/W" and "<name>/b", drawn from one of the Glorot, He or LeCun
// initialisers. This program checks the layer against a hand-written forward
// pass, shows weight sharing and application to sequences, measures how each
// initialiser propagates the scale of activations through a deep stack, and
// verifies the gradients of the loss with respect to W and b against finite
// differences and the closed-form expressions.
//
// TF -> datamunge: the graph, placeholders, session and init op disappear
// (layers run eagerly); nn::Params is nn::ParameterStore, the input width is
// read from x instead of being passed, AddSymbolicGradients is nn::grad, and
// the Assign ops used for finite differences become in-place copies under a
// NoGradGuard. Weights come from datamunge's own generator, so the printed
// weights differ from TF's; the checks compare against the hand / closed-form
// computations and the scale thresholds, which hold for any seed.

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

void PrintMatrix(const char* label, const Tensor& t) {
  const Tensor m = t.contiguous();
  std::printf("  %s [%lld x %lld]\n", label, static_cast<long long>(m.size(0)), static_cast<long long>(m.size(1)));
  for (std::int64_t i = 0; i < m.size(0); ++i) {
    std::printf("    ");
    for (std::int64_t j = 0; j < m.size(1); ++j) std::printf("%8.4f", m.at({i, j}));
    std::printf("\n");
  }
}

// ---------------------------------------------------------------------------
// 1. Forward pass against a hand computation, bias and activation options
// ---------------------------------------------------------------------------

void ForwardPass() {
  Header("Forward pass: y = act(x W + b), checked by hand");
  nn::ParameterStore p(1);
  nn::DenseOptions   linear;   // identity, bias
  nn::DenseOptions   no_bias;  // identity, no bias
  no_bias.bias = false;
  nn::DenseOptions relu;  // ReLU, bias started at 0.1
  relu.activation = Activation::Relu, relu.bias_value = 0.1;
  const Tensor x     = nn::tensor({{1, 2, 3}, {0, -1, 1}, {2, 2, 2}, {-1, 0, 1}});
  const Tensor y_lin = nn::dense(p, "lin", x, 2, linear);
  const Tensor y_nob = nn::dense(p, "nob", x, 2, no_bias);
  const Tensor y_rel = nn::dense(p, "relu", x, 2, relu);
  PrintMatrix("x", x);
  PrintMatrix("W (lin)", p.at("lin/W"));
  PrintMatrix("y = x W + b (b = 0 at init)", y_lin);
  PrintMatrix("y = x W (no bias; different W)", y_nob);
  PrintMatrix("y = relu(x W + 0.1)", y_rel);
  // Hand computation of the ReLU layer.
  const Tensor W = p.at("relu/W"), b = p.at("relu/b");
  float        max_err = 0;
  for (std::int64_t i = 0; i < 4; ++i)
    for (std::int64_t j = 0; j < 2; ++j) {
      float acc = static_cast<float>(b.at({j}));
      for (std::int64_t k = 0; k < 3; ++k) acc += static_cast<float>(x.at({i, k}) * W.at({k, j}));
      max_err = std::max(max_err, std::abs(std::max(acc, 0.f) - static_cast<float>(y_rel.at({i, j}))));
    }
  std::printf("  max |layer - hand computation| = %.2e\n", max_err);
  std::printf("  parameters: %lld (= 3*2+2 + 3*2 + 3*2+2)\n", static_cast<long long>(p.count()));
  check::Require(max_err < 1e-5f, "dense forward pass disagrees with the hand computation");
  check::Require(p.count() == 22, "parameter count");
  check::Near(b.at({0}), 0.1, 1e-7, "bias_value");
}

// ---------------------------------------------------------------------------
// 2. Weight sharing and sequence inputs
// ---------------------------------------------------------------------------

void SharingAndSequences() {
  Header("Weight sharing (same name => same W) and [batch, time, in] inputs");
  nn::ParameterStore p(2);
  const Tensor       same = nn::tensor({{1, 2, 3}, {0, -1, 1}});
  Tensor             s    = nn::empty({2, 5, 3});
  for (int i = 0; i < 30; ++i) s.data_ptr<float>()[i] = 0.1f * (i % 7) - 0.3f;
  const Tensor ya = nn::dense(p, "shared", same, 2);
  const Tensor yb = nn::dense(p, "shared", same.clone(), 2);  // reuses shared/W, shared/b
  const Tensor ys = nn::dense(p, "shared", s, 2);             // [2, 5, 3] -> [2, 5, 2]
  const float  diff = nn::max(nn::abs(ya - yb)).item<float>();
  std::printf("  parameters after three applications: %lld (one W [3,2] and one b [2])\n",
              static_cast<long long>(p.count()));
  std::printf("  same input through both applications: max difference %.1e\n", diff);
  std::printf("  sequence input [2, 5, 3] -> output shape %s\n", nn::shape_str(ys.shape()).c_str());
  std::printf("  (every [batch, time] position is mapped by the same affine map)\n");
  // Position (1, 3) of the sequence equals the layer applied to that row alone.
  const Tensor row  = nn::reshape(nn::select(nn::select(s, 0, 1), 0, 3), {1, 3});
  const float  dpos = nn::max(nn::abs(nn::dense(p, "shared", row, 2) - nn::select(nn::select(ys, 0, 1), 0, 3)))
                         .item<float>();
  check::Require(p.count() == 8 && p.size() == 2, "shared layer created extra parameters");
  check::Require(diff == 0, "shared applications differ");
  check::Require(ys.shape() == nn::Shape{2, 5, 2}, "sequence output shape");
  check::Require(dpos < 1e-6f, "sequence position differs from the row-wise map");
}

// ---------------------------------------------------------------------------
// 3. Initialisers and the scale of activations through a deep stack
// ---------------------------------------------------------------------------

void InitialiserScales() {
  Header("Initialisers: std of pre-activations through 24 layers of width 256");
  struct Case {
    const char*     label;
    nn::Initializer init;
    Activation      act;
  };
  const Case cases[] = {
      {"glorot_uniform + tanh", nn::init::glorot_uniform(), Activation::Tanh},
      {"glorot_uniform + relu", nn::init::glorot_uniform(), Activation::Relu},
      {"he_normal + relu", nn::init::he_normal(), Activation::Relu},
      {"lecun_normal + selu", nn::init::lecun_normal(), Activation::Selu},
      {"lecun_normal + relu", nn::init::lecun_normal(), Activation::Relu},  // variance 1/fan_in: too small for ReLU
  };
  constexpr int depth = 24, width = 256, batch = 512;
  std::printf("  %-28s", "layer:");
  for (int l = 1; l <= depth; ++l)
    if (l == 1 || l % 4 == 0) std::printf(" %6d", l);
  std::printf("\n");
  // The same std::mt19937_64(0) input as TF.
  Tensor xv = nn::empty({batch, width});
  {
    std::mt19937_64                 rng(0);
    std::normal_distribution<float> normal(0.f, 1.f);
    float*                          px = xv.data_ptr<float>();
    for (int i = 0; i < batch * width; ++i) px[i] = normal(rng);
  }
  std::vector<std::vector<double>> stds;
  nn::NoGradGuard                  no_grad;  // forward only
  for (const Case& c : cases) {
    nn::ParameterStore p(3);
    Tensor             h = xv;
    std::vector<double> row;
    std::printf("  %-28s", c.label);
    for (int l = 1; l <= depth; ++l) {
      nn::DenseOptions o;
      o.init = c.init, o.bias = false;
      const Tensor z = nn::dense(p, "l" + std::to_string(l - 1), h, width, o);  // pre-activation
      h              = nn::activate(z, c.act);
      const double sd = std::sqrt(nn::mean(nn::square(z.to(nn::DType::Float64))).item());
      row.push_back(sd);
      if (l == 1 || l % 4 == 0) std::printf(" %6.3f", sd);
    }
    std::printf("\n");
    stds.push_back(row);
  }
  std::printf("  (He keeps the scale under ReLU; Glorot and LeCun lose about sqrt(1/2) per ReLU layer,\n"
              "   so the signal vanishes geometrically; LeCun + SELU self-normalises)\n");
  // TF: tanh 1.003 -> 0.147, glorot+relu -> 0.000, he+relu 1.25, selu 1.007, lecun+relu -> 0.000.
  check::Near(stds[0][0], 1.0, 0.1, "glorot+tanh layer-1 std");
  check::Require(stds[0][23] > 0.08 && stds[0][23] < 0.3, "glorot+tanh decays slowly");
  check::Require(stds[1][23] < 0.01, "glorot+relu vanishes");
  check::Require(stds[2][23] > 0.6 && stds[2][23] < 2.5, "he+relu keeps the scale");
  for (double sd : stds[3]) check::Require(sd > 0.85 && sd < 1.15, "lecun+selu self-normalises");
  check::Require(stds[4][23] < 0.01, "lecun+relu vanishes");
}

// ---------------------------------------------------------------------------
// 4. Gradients of a squared loss w.r.t. W and b: symbolic, numeric, closed form
// ---------------------------------------------------------------------------

void Gradients() {
  Header("Gradients of L = mean((x W + b - t)^2): autograd vs finite differences vs closed form");
  nn::ParameterStore p(4);
  const Tensor       xv   = nn::tensor({{1, 2, 3}, {0, -1, 1}, {2, 2, 2}, {-1, 0, 1}});
  const Tensor       tv   = nn::tensor({{1, 0}, {0, 1}, {1, 1}, {0, 0}});
  auto               loss = [&]() -> std::pair<Tensor, Tensor> {
    Tensor y = nn::dense(p, "d", xv, 2);
    return {nn::mean(nn::square(y - tv), {0, 1}), y};
  };
  auto [L, yv]        = loss();
  const Tensor W      = p.at("d/W"), b = p.at("d/b");
  auto         grads  = nn::grad(L, {W, b});
  const Tensor gW     = grads[0], gb = grads[1];
  const float  L0     = L.item<float>();
  std::printf("  loss = %.6f\n", L0);
  // Closed form: dL/dW = (2/(N m)) x^T (y - t), dL/db = (2/(N m)) sum_rows (y - t).
  const float scale      = 2.f / (4 * 2);
  float       err_closed = 0;
  for (std::int64_t k = 0; k < 3; ++k)
    for (std::int64_t j = 0; j < 2; ++j) {
      float g = 0;
      for (std::int64_t i = 0; i < 4; ++i) g += static_cast<float>(xv.at({i, k}) * (yv.at({i, j}) - tv.at({i, j})));
      err_closed = std::max(err_closed, std::abs(scale * g - static_cast<float>(gW.at({k, j}))));
    }
  for (std::int64_t j = 0; j < 2; ++j) {
    float g = 0;
    for (std::int64_t i = 0; i < 4; ++i) g += static_cast<float>(yv.at({i, j}) - tv.at({i, j}));
    err_closed = std::max(err_closed, std::abs(scale * g - static_cast<float>(gb.at({j}))));
  }
  // Central finite differences, one parameter at a time, perturbing the
  // stored parameters in place (TF: Assign ops fed from placeholders).
  nn::NoGradGuard no_grad;
  const float     eps    = 1e-3f;
  float           err_fd = 0;
  auto            fd_of  = [&](Tensor param, const Tensor& g) {
    const Tensor base = param.clone();
    float*       pp   = param.data_ptr<float>();
    for (std::int64_t i = 0; i < param.numel(); ++i) {
      pp[i] += eps;
      const float lp = loss().first.item<float>();
      param.copy_(base);
      pp[i] -= eps;
      const float lm = loss().first.item<float>();
      param.copy_(base);
      const float fd = (lp - lm) / (2 * eps);
      err_fd         = std::max(err_fd, std::abs(fd - g.contiguous().data_ptr<float>()[i]));
    }
  };
  fd_of(W, gW);
  fd_of(b, gb);
  PrintMatrix("dL/dW (autograd)", gW);
  std::printf("  dL/db (autograd) = [%.4f, %.4f]\n", gb.at({0}), gb.at({1}));
  const float gmax = nn::max(nn::abs(gW)).item<float>();
  std::printf("  max |autograd - closed form| = %.1e\n", err_closed);
  std::printf("  max |autograd - central difference (eps = %.0e, float32)| = %.1e, i.e. %.1e relative to max |dL/dW|\n",
              eps, err_fd, err_fd / gmax);
  // TF: closed form 0.0e+00, finite differences 5.8e-04 relative.
  check::Require(err_closed < 1e-5f, "gradient vs closed form");
  check::Require(err_fd / gmax < 5e-3f, "gradient vs finite differences");
}

}  // namespace

int main() {
  try {
    ForwardPass();
    SharingAndSequences();
    InitialiserScales();
    Gradients();
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
