// CMD: just run-nn principal_components
// PCA through thin SVD, and RBF kernel PCA with out-of-sample projection.
//
// Port of tflow's principal_components (PCA/centering in common/pca.hpp, RBF
// kernel in common/kernels.hpp; nn::svd and nn::eigh replace TF's Svd and
// SelfAdjointEig). A rotated three-dimensional Fourier dataset has known
// eigenvectors and variances. Recover them up to sign, check whitened
// covariance, and verify the rank-two reconstruction error on training and
// held-out points. Kernel PCA centers a ring-data Gram matrix, keeps three
// positive eigenvalues and projects new points using training statistics.
// Check the eigenproblem, training reprojection and invariance to query
// batching. Kernel coordinates have no input-space inverse here; no pre-image
// reconstruction is claimed. No files are written.

#include <array>
#include <cmath>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "common/checks.hpp"
#include "common/kernels.hpp"
#include "common/pca.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void Linear() {
  constexpr int                                       n = 128;
  const double                                        a = .6, b = .4;
  const std::array<std::array<double, 3>, 3>          rotation{{{{std::cos(a), -std::sin(a) * std::cos(b), std::sin(a) * std::sin(b)}},
                                                                {{std::sin(a), std::cos(a) * std::cos(b), -std::cos(a) * std::sin(b)}},
                                                                {{0, std::sin(b), std::cos(b)}}}};
  const double                                        offset[]{1, -2, .5};
  auto data = [&](double shift) {
    std::vector<double> x(n * 3);
    for (int i = 0; i < n; ++i) {
      const double t = 2 * std::acos(-1.) * (i + shift) / n;
      const double latent[]{std::sqrt(18.) * std::cos(t), std::sqrt(2.) * std::sin(t), std::sqrt(.08) * std::cos(2 * t)};
      for (int d = 0; d < 3; ++d) {
        x[i * 3 + d] = offset[d];
        for (int j = 0; j < 3; ++j) x[i * 3 + d] += rotation[d][j] * latent[j];
      }
    }
    return nn::from_vector(x, {n, 3});
  };
  const Tensor            training = data(0), held_out = data(.5);
  ml::PrincipalComponents pca(training, 2);
  auto                    reconstruct = [&](const Tensor& q) { return pca.Reconstruct(pca.Transform(q)); };
  const Tensor            whitened    = pca.scores / nn::sqrt(pca.variances);
  const auto              covariance  = (nn::matmul(whitened, whitened, true, false) / double(n - 1)).to_vector<double>();
  const auto              mean        = pca.mean.to_vector<double>();
  const auto              components  = pca.components.to_vector<double>();
  const auto              variances   = pca.variances.to_vector<double>();
  for (int d = 0; d < 3; ++d) check::Near(mean[d], offset[d], 1e-12, "PCA training mean");
  for (int c = 0; c < 2; ++c) {
    double dot = 0;
    for (int d = 0; d < 3; ++d) dot += components[d * 2 + c] * rotation[d][c];
    check::Near(std::abs(dot), 1, 1e-12, "principal axis up to sign");
    check::Near(variances[c], (c == 0 ? 9. : 1.) * n / (n - 1), 1e-11, "known sample variance");
  }
  for (int i = 0; i < 4; ++i) check::Near(covariance[i], i % 3 == 0 ? 1 : 0, 1e-12, "whitened covariance identity");
  auto mse = [](const Tensor& x, const Tensor& restored) { return nn::mean(nn::square(x - restored)).item<double>(); };
  const double train_mse = mse(training, reconstruct(training));
  check::Near(train_mse, .04 / 3, 1e-12, "discarded-component reconstruction energy");
  std::cout << "  PCA variances=" << variances[0] << ',' << variances[1] << "; explained fraction=" << 10 / 10.04
            << "; train MSE=" << train_mse;
  const double test_mse = mse(held_out, reconstruct(held_out));
  check::Near(test_mse, .04 / 3, 1e-12, "held-out reconstruction energy");
  std::cout << "; test MSE=" << test_mse << '\n';
}

void Kernel() {
  constexpr int      n = 80, retained = 3;
  std::vector<float> xs(n * 2), qs(6 * 2);
  for (int i = 0; i < n; ++i) {
    const double angle = 2 * std::acos(-1.) * (i / 2) / (n / 2), radius = i % 2 ? 2 : .5;
    xs[i * 2]          = float(radius * std::cos(angle));
    xs[i * 2 + 1]      = float(radius * std::sin(angle));
  }
  for (int i = 0; i < 6; ++i) {
    const double angle = .13 + .7 * i, radius = i % 2 ? 2 : .5;
    qs[i * 2]          = float(radius * std::cos(angle));
    qs[i * 2 + 1]      = float(radius * std::sin(angle));
  }
  const Tensor x        = nn::from_vector(xs, {n, 2}), query_value = nn::from_vector(qs, {6, 2});
  const Tensor kernel   = ml::RbfKernel(x, x, .5f);
  const Tensor centered = ml::CenterKernel(kernel);
  const auto   eigen    = nn::eigh(centered);
  // eigh returns ascending eigenvalues. Reverse the top slice.
  const Tensor values  = nn::reverse(nn::slice(eigen.e, {n - retained}, {retained}), {0});
  const Tensor vectors = nn::reverse(nn::slice(eigen.v, {0, n - retained}, {n, retained}), {1});
  const Tensor root    = nn::expand_dims(nn::sqrt(values), 0);
  const Tensor scaled = vectors / root, training_scores = vectors * root;
  auto project = [&](const Tensor& q) { return nn::matmul(ml::CenterCrossKernel(ml::RbfKernel(q, x, .5f), kernel), scaled); };
  const auto residual   = (nn::matmul(centered, vectors) - vectors * nn::expand_dims(values, 0)).to_vector<float>();
  const auto centered_v = centered.to_vector<float>();
  const auto spectrum = eigen.e.to_vector<float>(), top = values.to_vector<float>();
  const auto scores = training_scores.to_vector<float>(), projection = project(x).to_vector<float>();
  for (int i = 0; i < n; ++i) {
    double sum = 0;
    for (int j = 0; j < n; ++j) sum += centered_v[i * n + j];
    check::Near(sum, 0, 2e-5, "centered kernel row sums");
  }
  check::Require(spectrum[0] > -1e-5 && top[retained - 1] > .1, "PSD kernel and positive retained spectrum");
  for (int i = 0; i < n * retained; ++i) {
    check::Near(scores[i], projection[i], 2e-5, "training projection equals eigen embedding");
    check::Near(residual[i], 0, 2e-5, "kernel eigenproblem residual");
  }
  std::cout << "  kernel PCA retained eigenvalues:";
  for (int c = 0; c < retained; ++c) std::cout << ' ' << top[c];
  const auto batch_projection = project(query_value).to_vector<float>();
  for (int i = 0; i < 6; ++i) {
    const auto row = project(nn::from_vector(std::vector<float>{qs[i * 2], qs[i * 2 + 1]}, {1, 2})).to_vector<float>();
    for (int c = 0; c < retained; ++c)
      check::Near(row[c], batch_projection[i * retained + c], 2e-6, "projection independent of query batch");
  }
  std::cout << "\n  centered Gram, eigenproblem, training reprojection and six independent query projections passed\n";
}

}  // namespace

int main() {
  try {
    std::cout << "== Principal components\n";
    Linear();
    Kernel();
    std::cout << "All PCA and kernel-PCA checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
