/// @file nn_advanced_variants_tests.cpp
/// @brief Tests for newton_schulz_orthogonalize(), Muon, Lookahead and sam_gradients().

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {
const auto F64 = nn::DType::Float64;

double poly_iterate(double x, int steps, nn::NewtonSchulzCoefficients k) {
  for (int i = 0; i < steps; ++i) x = k.a * x + k.b * x * x * x + k.c * x * x * x * x * x;
  return x;
}
}  // namespace

TEST(NnAdvancedVariants, NewtonSchulzMapsSingularValuesByThePolynomial) {
  // G = U diag(s) V^T with known s; after k steps the singular values must be p^(k)(s / ||G||_F) and the singular
  // vectors unchanged, for both a wide and a tall matrix.
  for (bool tall : {false, true}) {
    const Tensor A = nn::from_vector(std::vector<double>{0.3, -1.2, 0.7, 2.0, 0.1, -0.4, 1.5, 0.9, -0.6, 0.2, 1.1, -1.7}, {3, 4});
    const Tensor G = tall ? nn::transpose(A) : A;
    const auto d = nn::svd(G);
    const auto s = d.s.to_vector<double>();
    double fro = 0;
    for (double v : s) fro += v * v;
    fro = std::sqrt(fro);
    for (auto k : {nn::NewtonSchulzCoefficients::cubic(), nn::NewtonSchulzCoefficients::quintic()}) {
      const Tensor X = nn::newton_schulz_orthogonalize(G, 4, k);
      ASSERT_EQ(X.shape(), G.shape());
      // U^T X V should be diagonal with entries p^(4)(s_i / ||G||_F).
      const Tensor D = nn::matmul(nn::matmul(d.u, X, true, false), d.v);
      const auto dv = D.to_vector<double>();
      const std::int64_t r = D.size(1);
      for (std::int64_t i = 0; i < D.size(0); ++i)
        for (std::int64_t j = 0; j < r; ++j) {
          const double expected = i == j ? poly_iterate(s[i] / (fro + 1e-7), 4, k) : 0.0;
          EXPECT_NEAR(dv[i * r + j], expected, 1e-10) << "tall=" << tall << " i=" << i << " j=" << j;
        }
    }
  }
  // The cubic iteration converges to the exact polar factor U V^T.
  const Tensor G = nn::from_vector(std::vector<double>{2.0, 1.0, 0.0, 1.0, 3.0, 1.0}, {2, 3});
  const Tensor X = nn::newton_schulz_orthogonalize(G, 40, nn::NewtonSchulzCoefficients::cubic());
  const auto I = nn::matmul(X, X, false, true).to_vector<double>();
  EXPECT_NEAR(I[0], 1.0, 1e-12);
  EXPECT_NEAR(I[1], 0.0, 1e-12);
  EXPECT_NEAR(I[3], 1.0, 1e-12);
  EXPECT_THROW((void)nn::newton_schulz_orthogonalize(nn::zeros({3}, F64)), std::invalid_argument);
  EXPECT_THROW((void)nn::newton_schulz_orthogonalize(G, 0), std::invalid_argument);
}

TEST(NnAdvancedVariants, MuonMatchesReferenceUpdate) {
  Tensor W = nn::from_vector(std::vector<double>{1.0, 0.0, -1.0, 0.5, 2.0, 0.0}, {3, 2});
  Tensor b = nn::from_vector(std::vector<double>{0.1, -0.1}, {2});
  W.requires_grad_(), b.requires_grad_();
  nn::MuonConfig cfg;
  cfg.momentum = 0.9, cfg.weight_decay = 0.1;
  nn::Muon opt({W, b}, 0.02, cfg);
  Tensor Wref = W.detach().clone(), bref = b.detach().clone();
  Tensor mW = nn::zeros({3, 2}, F64), mb = nn::zeros({2}, F64);
  for (int t = 1; t <= 3; ++t) {
    const Tensor gW = nn::from_vector(std::vector<double>{0.5 * t, -1.0, 0.2, 0.3, -0.1 * t, 0.7}, {3, 2});
    const Tensor gb = nn::from_vector(std::vector<double>{1.0, -2.0 * t}, {2});
    opt.step({gW, gb});
    mW = 0.9 * mW + gW, mb = 0.9 * mb + gb;
    Wref = Wref - 0.02 * 0.1 * Wref - 0.02 * std::sqrt(1.5) * nn::newton_schulz_orthogonalize(gW + 0.9 * mW);
    bref = bref - 0.02 * 0.1 * bref - 0.02 * (gb + 0.9 * mb);
  }
  EXPECT_EQ(opt.iterations(), 3);
  const auto w = W.to_vector<double>(), wr = Wref.to_vector<double>();
  for (std::size_t i = 0; i < w.size(); ++i) EXPECT_NEAR(w[i], wr[i], 1e-14);
  const auto bv = b.to_vector<double>(), br = bref.to_vector<double>();
  for (std::size_t i = 0; i < bv.size(); ++i) EXPECT_NEAR(bv[i], br[i], 1e-14);
  EXPECT_THROW(nn::Muon({W}, -1.0), std::invalid_argument);
  EXPECT_THROW(nn::Muon({W}, 0.1, nn::MuonConfig{.momentum = 1.0}), std::invalid_argument);
}

TEST(NnAdvancedVariants, LookaheadContractionAndSamGradients) {
  // Gradient descent on 0.5 * h * w^2 multiplies w by c = 1 - lr * h per step; Lookahead's cycle by 1 - a + a c^k.
  Tensor w = nn::from_vector(std::vector<double>{1.0}, {1});
  w.requires_grad_();
  nn::OptimizerConfig sgd;
  sgd.kind = nn::OptimizerKind::Sgd;
  nn::Optimizer inner({w}, 0.5, sgd);
  nn::Lookahead la({w}, 4, 0.5);
  const double h = 3.0, c = 1 - 0.5 * h, factor = 1 - 0.5 + 0.5 * std::pow(c, 4);
  double expected = 1.0;
  for (int cycle = 0; cycle < 3; ++cycle) {
    for (int i = 0; i < 4; ++i) {
      inner.step({h * w.detach()});
      EXPECT_EQ(la.after_step(), i == 3);
    }
    expected *= factor;
    EXPECT_NEAR(w.to_vector<double>()[0], expected, 1e-14);
    EXPECT_NEAR(la.slow_weights()[0].to_vector<double>()[0], expected, 1e-14);
  }
  EXPECT_THROW(nn::Lookahead({w}, 0), std::invalid_argument);

  // L = 0.5 (3 x^2 + y^2): the SAM gradient is A (w + rho g / |g|) and the parameters are restored.
  Tensor x = nn::from_vector(std::vector<double>{1.0}, {1}), y = nn::from_vector(std::vector<double>{2.0}, {1});
  x.requires_grad_(), y.requires_grad_();
  auto loss = [&] { return 0.5 * nn::sum(3.0 * nn::square(x) + nn::square(y)); };
  const auto r = nn::sam_gradients({x, y}, loss, 0.1);
  const double gx = 3.0, gy = 2.0, n = std::sqrt(gx * gx + gy * gy);
  const double px = 1.0 + 0.1 * gx / n, py = 2.0 + 0.1 * gy / n;
  EXPECT_NEAR(r.gradients[0].item(), 3.0 * px, 1e-14);
  EXPECT_NEAR(r.gradients[1].item(), py, 1e-14);
  EXPECT_NEAR(r.loss.item(), 3.5, 1e-14);
  EXPECT_NEAR(r.sharp_loss.item(), 0.5 * (3 * px * px + py * py), 1e-14);
  EXPECT_EQ(x.item(), 1.0);
  EXPECT_EQ(y.item(), 2.0);
  const auto r0 = nn::sam_gradients({x, y}, loss, 0.0);
  EXPECT_NEAR(r0.gradients[0].item(), 3.0, 1e-14);
  EXPECT_THROW((void)nn::sam_gradients({x}, loss, -1.0), std::invalid_argument);
}
