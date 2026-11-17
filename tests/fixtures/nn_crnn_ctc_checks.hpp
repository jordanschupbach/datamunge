// Independent numeric and model-geometry checks; experiment driver omitted.
#pragma once
#include "nn_interpretation_check_support.hpp"
namespace inpaint {
using namespace dm::nn::models;
}
namespace denoise {
using namespace dm::nn::models;
}
namespace flow {
using namespace dm::nn::models;
}
namespace color {
using namespace dm::nn::models;
}
namespace depth {
using namespace dm::nn::models;
}
namespace pose {
using namespace dm::nn::models;
}
namespace style {
using namespace dm::nn::models;
}
namespace crnn {
using namespace dm::nn::models;
}
namespace ocr {
using namespace dm::nn::models;
}
namespace cifar {
using namespace dm::nn::models;
}

#include <chrono>
#include <datamunge/nn/image_io.hpp>
#include <datamunge/nn/sequence.hpp>
#include <fstream>
#include <memory>
namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

Tensor Ints(const std::vector<std::int64_t>& v, nn::Shape shape = {}) {
  return nn::from_vector(v, shape);
}

// Probabilities of a [T][C] block of logits.
std::vector<std::vector<double>> Softmax(const std::vector<float>& z, int T, int C, int B, int n) {
  std::vector<std::vector<double>> p(T, std::vector<double>(C));
  for (int t = 0; t < T; ++t) {
    double sum = 0;
    for (int c = 0; c < C; ++c)
      sum += std::exp(z[(t * B + n) * C + c]);
    for (int c = 0; c < C; ++c)
      p[t][c] = std::exp(z[(t * B + n) * C + c]) / sum;
  }
  return p;
}

void CtcChecks() {
  const std::vector<float> values{.2f, -.3f, .4f, -.1f, .5f, .1f, .7f, -.2f, .3f};
  const Tensor             logits = nn::from_vector(values, {3, 1, 3});
  const auto out   = nn::ctc_loss_with_gradient(logits, Ints({0, 0}, {1, 2}), Ints({2}), Ints({3}));
  const auto p     = Softmax(values, 3, 3, 1, 0);
  double     total = 0, occupation[3][3]{};
  int        paths = 0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c) {
        const std::vector<std::int64_t> path{a, b, c};
        if (nn::ctc_collapse(path, 2) != std::vector<std::int64_t>{0, 0})
          continue;
        ++paths;
        const double pr = p[0][a] * p[1][b] * p[2][c];
        total += pr;
        for (int t = 0; t < 3; ++t)
          occupation[t][path[t]] += pr;
      }
  check::Require(paths == 1, "repeated label needs separating blank,unique minimal path");
  check::Near(out.loss.item<double>(), -std::log(total), 5e-7, "CTC sum over all 27 paths");
  const auto g     = out.gradient.to_vector<double>();
  double     error = 0;
  for (int t = 0; t < 3; ++t)
    for (int c = 0; c < 3; ++c)
      error = std::max(error, std::abs(p[t][c] - occupation[t][c] / total - g[t * 3 + c]));
  check::Near(error, 0, 1e-6, "EVERY CTC logit derivative versus exhaustive path posterior");
  check::Require(
      ocr::CtcCollapse({1, 1, 36, 1, 36, 1}) == "111" && ocr::CtcCollapse({36, 36}).empty(),
      "merge repeats BEFORE removing blank");
  check::Require(
      ocr::OcrDistance("coffee", "cofee") == 1 && ocr::OcrDistance("abc", "xyz") == 3 &&
          ocr::OcrDistance("", "abc") == 3,
      "independent edit distance cases");
  std::cout << " exhaustive repeated-label CTC value/all-logit adjoints,blank/repeat "
               "order/edit-distance fixtures passed,error="
            << error << std::endl;
}

void VariableCtcChecks() {
  std::vector<float> values(18);
  for (int i = 0; i < 18; ++i)
    values[i] = static_cast<float>(.3 * std::sin(i * .7));
  Tensor     z      = nn::from_vector(values, {3, 2, 3});
  const auto labels = Ints({0, -1, 0, 0}, {2, 2}), label_lengths = Ints({1, 2}),
             lengths = Ints({2, 3});
  z.requires_grad_(true);
  const Tensor mean     = nn::mean(nn::ctc_loss(z, labels, label_lengths, lengths));
  const auto   adjoint  = nn::grad(mean, {z})[0].to_vector<double>();
  double       expected = 0, error = 0;
  int          accepted = 0;
  for (int n = 0; n < 2; ++n) {
    const int  length = n ? 3 : 2;
    const auto p      = Softmax(values, length, 3, 2, n);
    double     posterior[3][3]{}, total = 0;
    int        combinations = 1;
    for (int t = 0; t < length; ++t)
      combinations *= 3;
    const std::vector<std::int64_t> target =
        n ? std::vector<std::int64_t>{0, 0} : std::vector<std::int64_t>{0};
    for (int k = 0; k < combinations; ++k) {
      int                       number = k;
      std::vector<std::int64_t> path;
      double                    pr = 1;
      for (int t = 0; t < length; ++t) {
        const int c = number % 3;
        number /= 3;
        path.push_back(c);
        pr *= p[t][c];
      }
      if (nn::ctc_collapse(path, 2) != target)
        continue;
      ++accepted;
      total += pr;
      for (int t = 0; t < length; ++t)
        posterior[t][path[t]] += pr;
    }
    expected -= std::log(total) / 2;
    for (int t = 0; t < 3; ++t)
      for (int c = 0; c < 3; ++c) {
        const double want = t < length ? (p[t][c] - posterior[t][c] / total) / 2 : 0;
        error             = std::max(error, std::abs(want - adjoint[(t * 2 + n) * 3 + c]));
      }
  }
  check::Require(
      accepted == 4, "multiple alternative alignments retained plus repeated-character path");
  check::Near(mean.item<double>(), expected, 1e-6, "batch mean CTC exhaustive-path NLL");
  check::Near(error, 0, 1e-6, "all mean CTC derivatives and zero padded-time gradient");
  // Independent finite differences of the scalar mean verify the reduction too.
  double      fd_error = 0;
  const float delta    = .002f;
  for (int i = 0; i < 18; ++i) {
    auto plus = values, minus = values;
    plus[i] += delta;
    minus[i] -= delta;
    nn::NoGradGuard guard;
    const double    hi =
        nn::mean(nn::ctc_loss(nn::from_vector(plus, {3, 2, 3}), labels, label_lengths, lengths))
            .item<double>();
    const double lo =
        nn::mean(nn::ctc_loss(nn::from_vector(minus, {3, 2, 3}), labels, label_lengths, lengths))
            .item<double>();
    fd_error = std::max(
        fd_error,
        std::abs(
            static_cast<double>(static_cast<float>(hi) - static_cast<float>(lo)) / (2 * delta) -
            adjoint[i]));
  }
  check::Near(fd_error, 0, 2e-4, "every variable-time batch-mean CTC logit finite difference");
  std::cout << " multiple-alignment/variable-time/mean reduction/padded-gradient exhaustive error="
            << error << ",all18 finite differences error=" << fd_error << std::endl;
}

void FullChecks() {
  nn::ParameterStore p(317), state(319);
  std::vector<float> input(32 * 100);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>(.5 + .4 * std::sin(i * .17));
  const auto   model     = crnn::Crnn(p, state, nn::from_vector(input, {1, 32, 100, 1}), {}, false);
  const Tensor objective = nn::sum(nn::square(model.logits));
  const auto   gradients = nn::grad(objective, p.parameters());
  const std::vector<nn::Shape> shapes{
      {1, 32, 100, 64},
      {1, 16, 50, 128},
      {1, 8, 25, 256},
      {1, 8, 25, 256},
      {1, 4, 26, 512},
      {1, 4, 26, 512},
      {1, 1, 26, 512}};
  for (int i = 0; i < 7; ++i)
    check::Require(model.stages[i].shape() == shapes[i], "official CRNN stage shape");
  check::Require(model.logits.shape() == nn::Shape{26, 1, 37}, "official CRNN logits shape");
  check::Require(p.count() == 8327205, "independent sum of conv,BN,LSTM and projection parameters");
  check::Require(
      p.size() == 36 && state.size() == 6, "seven biased convs,three BN,two projected BLSTMs");
  for (const auto& g : gradients) {
    const double norm = nn::sum(nn::square(g)).item<double>();
    check::Require(
        std::isfinite(norm) && norm > 0, "every full-width learned tensor has nonzero derivative");
  }
  std::cout << " full CRNN:" << p.count()
            << " parameters/36 tensors,6 BN state,all seven convolution shapes/T26/two BLSTMs and "
               "all-tensor gradients passed\n";
}

} // namespace
