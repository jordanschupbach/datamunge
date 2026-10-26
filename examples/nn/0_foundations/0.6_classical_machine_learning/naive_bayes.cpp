// CMD: just run-nn naive_bayes
// Gaussian, multinomial and Bernoulli naive Bayes with nn ops.
//
// Port of tflow's naive_bayes (model in common/naive_bayes.hpp). Fit Gaussian
// class statistics from training observations, validate a constant feature's
// variance floor and evaluate independent test data. A small count-document
// fixture has hand-computed Laplace-smoothed word probabilities; its
// binary-presence version checks Bernoulli likelihoods, including absent
// words. Compare every log score against host arithmetic, verify posterior
// normalization and evaluate an empty document. Samples come from the same
// std::mt19937 stream as the TF version. No files are written.

#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/naive_bayes.hpp"

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

void Documents() {
  const Tensor counts = nn::from_vector(
      std::vector<float>{3, 1, 0, 0, 2, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 3, 0, 1, 0, 2, 1, 0, 1, 2}, {6, 4});
  const Tensor query  = nn::from_vector(std::vector<float>{2, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0}, {3, 4});
  const Tensor labels = nn::from_vector(std::vector<std::int32_t>{0, 0, 0, 1, 1, 1});
  ml::NaiveBayes multinomial(counts, labels, 2, ml::NaiveBayesKind::kMultinomial);
  const auto binary_training = nn::cast(nn::greater(counts, 0.0), DType::Float32);
  const auto binary_query    = nn::cast(nn::greater(query, 0.0), DType::Float32);
  ml::NaiveBayes bernoulli(binary_training, labels, 2, ml::NaiveBayesKind::kBernoulli);
  const auto mscore = multinomial.Scores(query), bscore = bernoulli.Scores(binary_query);
  const auto mp = nn::softmax(mscore), bp = nn::softmax(bscore);
  const double multinomial_probability[2][4]{{7. / 13, 3. / 13, 2. / 13, 1. / 13}, {2. / 15, 2. / 15, 3. / 15, 8. / 15}};
  const double bernoulli_probability[2][4]{{.8, .6, .4, .2}, {.4, .4, .6, .8}};
  for (int c = 0; c < 2; ++c)
    for (int j = 0; j < 4; ++j) {
      check::Near(multinomial.probability.at({c, j}), multinomial_probability[c][j], 1e-7, "Laplace word probability");
      check::Near(bernoulli.probability.at({c, j}), bernoulli_probability[c][j], 1e-7, "Laplace presence probability");
    }
  for (int row = 0; row < 3; ++row) {
    for (int c = 0; c < 2; ++c) {
      double multi = std::log(.5), bern = std::log(.5);
      for (int j = 0; j < 4; ++j) {
        const double value = query.at({row, j});
        multi += value * std::log(multinomial_probability[c][j]);
        bern += std::log(value > 0 ? bernoulli_probability[c][j] : 1 - bernoulli_probability[c][j]);
      }
      check::Near(mscore.at({row, c}), multi, 1e-6, "multinomial host log score");
      check::Near(bscore.at({row, c}), bern, 1e-6, "Bernoulli host log score");
    }
    for (const Tensor* p : {&mp, &bp})
      check::Near(p->at({row, 0}) + p->at({row, 1}), 1, 1e-6, "document posterior normalization");
  }
  check::Require(mp.at({0, 0}) > .9 && mp.at({1, 1}) > .9, "multinomial document classification");
  check::Require(bp.at({0, 0}) > .85 && bp.at({1, 1}) > .85, "Bernoulli document classification");
  check::Near(mp.at({2, 0}), .5, 1e-7, "empty multinomial document yields prior");
  check::Near(bp.at({2, 0}), 4. / 7, 1e-6, "empty Bernoulli document includes absent-word evidence");
  std::cout << "  multinomial P(correct)=" << mp.at({0, 0}) << ',' << mp.at({1, 1}) << "; Bernoulli=" << bp.at({0, 0})
            << ',' << bp.at({1, 1})
            << "\n  smoothed probabilities, all log scores and empty-document posteriors match hand calculations\n";
}

void Gaussian() {
  std::mt19937                    rng(97);
  std::normal_distribution<float> normal;
  auto sample = [&](int n) {
    std::vector<float>        x(n * 3);
    std::vector<std::int32_t> y(n);
    for (int i = 0; i < n; ++i) {
      const int c  = i % 2;
      x[i * 3 + 0] = (c ? 1.5f : -1.5f) + .5f * normal(rng);
      x[i * 3 + 1] = c + .8f * normal(rng);
      x[i * 3 + 2] = 5;
      y[i]         = c;
    }
    return std::pair<Tensor, Tensor>{nn::from_vector(x, {n, 3}), nn::from_vector(y)};
  };
  const auto [training, labels] = sample(600);
  const auto [query, truth]     = sample(300);
  ml::NaiveBayes model(training, labels, 2, ml::NaiveBayesKind::kGaussian);
  const auto     scores     = model.Scores(query);
  const auto     posterior  = nn::softmax(scores);
  const auto     prediction = nn::argmax(scores, 1).to_vector<std::int64_t>();
  const auto     tx         = training.to_vector<float>();
  for (int c = 0; c < 2; ++c)
    for (int j = 0; j < 3; ++j) {
      double mean = 0, variance = 0;
      for (int i = c; i < 600; i += 2) mean += tx[i * 3 + j] / 300.;
      for (int i = c; i < 600; i += 2) variance += std::pow(tx[i * 3 + j] - mean, 2) / 300;
      check::Near(model.mean.at({c, j}), mean, 2e-6, "independent class mean");
      check::Near(model.variance.at({c, j}), std::max(variance, 1e-4), 1e-6, "independent class variance/floor");
    }
  const auto qy = truth.to_vector<std::int32_t>();
  int        correct = 0;
  for (int i = 0; i < 300; ++i) {
    correct += prediction[i] == qy[i];
    check::Near(posterior.at({i, 0}) + posterior.at({i, 1}), 1, 1e-6, "Gaussian posterior normalization");
    for (int c = 0; c < 2; ++c) {
      double expected = std::log(.5);
      for (int j = 0; j < 3; ++j) {
        const double mean = model.mean.at({c, j}), variance = model.variance.at({c, j});
        expected -= .5 * (std::log(2 * std::acos(-1.) * variance) + std::pow(query.at({i, j}) - mean, 2) / variance);
      }
      check::Near(scores.at({i, c}), expected, 1e-5, "Gaussian host log likelihood");
    }
  }
  check::Require(correct >= 294, "Gaussian held-out accuracy");
  std::cout << "  Gaussian test accuracy=" << correct / 300. << "; constant-feature variance=" << model.variance.at({0, 2})
            << '\n';
}

}  // namespace

int main() {
  try {
    std::cout << "== Naive Bayes\n";
    Gaussian();
    Documents();
    std::cout << "All naive-Bayes checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
