#pragma once
#include "nn_interpretation_check_support.hpp"
namespace {
void DerivativeChecks() {
  std::vector<float> input(16), kernel(8), bias{.03f, -.02f}, head(54);
  for (int i = 0; i < 16; ++i)
    input[i] = .3f * std::cos(.41 * i);
  for (int i = 0; i < 8; ++i)
    kernel[i] = .4f * std::sin(.53 * i + .2);
  for (int i = 0; i < 54; ++i)
    head[i] = .2f * std::sin(.71 * i + .4);
  const Tensor k = sal::FromF(kernel, {2, 2, 1, 2}), b = sal::FromF(bias, {2}),
               h = sal::FromF(head, {18, 3});
  auto logits_of = [&](const Tensor& x) {
    auto feature =
        nn::relu(nn::bias_add(nn::conv2d(x, k, nn::conv_options(1, nn::Padding::Valid)), b));
    return nn::matmul(nn::reshape(feature, {1, 18}), h);
  };
  const Tensor x        = sal::FromF(input, {1, 4, 4, 1});
  const auto   logits   = sal::F(logits_of(x));
  const auto   gradient = sal::F(sal::VanillaInputGradient(logits_of, x, sal::FromI({1}), 3));
  const auto   other    = sal::F(sal::VanillaInputGradient(logits_of, x, sal::FromI({0}), 3));
  std::vector<double> values(input.begin(), input.end());
  auto                host = [&](const std::vector<double>& v, int target) {
    double score = 0;
    for (int y = 0; y < 3; ++y)
      for (int a = 0; a < 3; ++a)
        for (int c = 0; c < 2; ++c) {
          double pre = bias[c];
          for (int ky = 0; ky < 2; ++ky)
            for (int kx = 0; kx < 2; ++kx)
              pre += v[(y + ky) * 4 + a + kx] * kernel[(ky * 2 + kx) * 2 + c];
          score += std::max(pre, 0.) * head[((y * 3 + a) * 2 + c) * 3 + target];
        }
    return score;
  };
  double error = 0, delta = 0;
  int    positive = 0, negative = 0;
  for (int i = 0; i < 16; ++i) {
    auto plus = values, minus = values;
    plus[i] += 1e-5;
    minus[i] -= 1e-5;
    error = std::max(error, std::abs((host(plus, 1) - host(minus, 1)) / 2e-5 - gradient[i]));
    delta += std::abs(double(gradient[i]) - other[i]);
    positive += gradient[i] > 0;
    negative += gradient[i] < 0;
  }
  check::Near(logits[1], host(values, 1), 1e-7, "raw target score independent host reference");
  check::Near(error, 0, 2e-7, "ALL16 raw-score input finite differences");
  check::Require(
      positive > 0 && negative > 0 && delta > 1e-4,
      "signed upstream derivatives preserved,target changes derivative");
  std::cout << "vanilla class-score fixture:ALL16 independent hostFD error=" << error
            << ",positive/negative derivatives=" << positive << '/' << negative
            << ",targetdelta=" << delta << ";rawscore and standard signed backprop passed\n";
}

/// TF's ShapeRefiner pass becomes eager shapes of every endpoint for a batch of two.
void GraphChecks() {
  nn::ParameterStore p(1117);
  Tensor             x = nn::zeros({2, 28, 28, 1});
  x.requires_grad_(true);
  const auto net   = sal::SaliencyCnn(p, x);
  const auto grad  = nn::grad(nn::sum(sal::SaliencyScore(net.logits, sal::FromI({0, 1}))), {x})[0];
  auto       shape = [](const Tensor& t, const nn::Shape& dims) {
    check::Require(t.dim() == std::int64_t(dims.size()), "classifier endpoint rank");
    check::Require(t.shape() == dims, "classifier/gradient endpoint dimensions");
  };
  shape(net.conv1, {2, 28, 28, 8});
  shape(net.pool1, {2, 14, 14, 8});
  shape(net.conv2, {2, 14, 14, 16});
  shape(net.pool2, {2, 7, 7, 16});
  shape(net.hidden, {2, 32});
  shape(net.logits, {2, 10});
  shape(grad, {2, 28, 28, 1});
  check::Require(
      p.count() == 5 * 5 * 8 + 8 + 3 * 3 * 8 * 16 + 16 + 784 * 32 + 32 + 32 * 10 + 10 &&
          p.count() == 26826 && p.size() == 8,
      "independent parameter inventory");
  std::cout << "saliency CNN all6endpoint/input-gradient shapes,26826parameters/8tensors passed\n";
}

} // namespace
