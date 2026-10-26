// datamunge::nn port of tflow_examples src/common/ml/naive_bayes.h.
// Gaussian, multinomial and Bernoulli naive Bayes fitted with nn ops.
// Inputs are float32 [N,D] and int32 labels [N], with every class represented.
// Multinomial features are nonnegative counts; Bernoulli features are 0/1.
// Gaussian uses population variances with an absolute positive floor.
// Count variants use additive smoothing; priors use empirical class counts.
// Fitted statistics depend only on training tensors. Scores are log-joints,
// omitting the multinomial coefficient shared by all classes for each row.
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>

namespace ml {
namespace nn = datamunge::nn;

enum class NaiveBayesKind { kGaussian, kMultinomial, kBernoulli };

struct NaiveBayes {
  NaiveBayesKind kind;
  nn::Tensor     log_prior, mean, variance, probability;

  NaiveBayes(const nn::Tensor& training, const nn::Tensor& labels, int classes, NaiveBayesKind model,
             float smoothing = 1.f, float variance_floor = 1e-4f)
      : kind(model) {
    if (classes < 2 || !std::isfinite(smoothing) || smoothing <= 0 || !std::isfinite(variance_floor) ||
        variance_floor <= 0)
      throw std::invalid_argument("positive naive-Bayes smoothing/floor and at least two classes required");
    const auto assignment = nn::one_hot(labels, classes, 1.0, 0.0, training.dtype());
    const auto count      = nn::sum(assignment, {0});
    log_prior             = nn::log(count / nn::sum(count, {0}));
    const auto mass       = nn::matmul(assignment, training, /*transpose_a=*/true);
    if (kind == NaiveBayesKind::kGaussian) {
      mean                = mass / nn::expand_dims(count, 1);
      const auto centered = nn::expand_dims(training, 1) - nn::expand_dims(mean, 0);
      const auto spread =
          nn::sum(nn::square(centered) * nn::expand_dims(assignment, 2), {0}) / nn::expand_dims(count, 1);
      variance = nn::maximum(spread, nn::scalar_like(double(variance_floor), spread));
    } else if (kind == NaiveBayesKind::kMultinomial) {
      const auto smoothed = mass + double(smoothing);
      probability         = smoothed / nn::sum(smoothed, {1}, true);
    } else {
      probability = (mass + double(smoothing)) / nn::expand_dims(count + 2.0 * smoothing, 1);
    }
  }

  nn::Tensor Scores(const nn::Tensor& query) const {
    if (kind == NaiveBayesKind::kGaussian) {
      const auto delta      = nn::expand_dims(query, 1) - nn::expand_dims(mean, 0);
      const auto exponent   = nn::square(delta) / nn::expand_dims(variance, 0);
      const auto normalizer = nn::log(double(2 * std::acos(-1.f)) * variance);
      return log_prior - 0.5 * nn::sum(exponent + normalizer, {2});
    }
    const auto scores = nn::matmul(query, nn::log(probability), false, /*transpose_b=*/true);
    if (kind == NaiveBayesKind::kMultinomial) return scores + log_prior;
    const auto absent = nn::matmul(1.0 - query, nn::log1p(-probability), false, true);
    return log_prior + (scores + absent);
  }
};

}  // namespace ml
