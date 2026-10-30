// datamunge::nn port of tflow_examples src/common/ml/naive_bayes.h.
// Gaussian, multinomial and Bernoulli naive Bayes fitted with nn ops.
// Inputs are float32 [N,D] and int32 labels [N], with every class represented.
// Multinomial features are nonnegative counts; Bernoulli features are 0/1.
// Gaussian uses population variances with an absolute positive floor.
// Count variants use additive smoothing; priors use empirical class counts.
// Fitted statistics depend only on training tensors. Scores are log-joints,
// omitting the multinomial coefficient shared by all classes for each row.
#pragma once

/// @file naive_bayes.hpp
/// @brief Gaussian, multinomial and Bernoulli tensor naive Bayes.

#include <cmath>
#include <datamunge/nn/ops.hpp>
#include <stdexcept>

namespace datamunge::nn::ml {
namespace nn = datamunge::nn;

/// @brief Conditionally independent Gaussian, multinomial-count or Bernoulli-presence likelihoods.
enum class NaiveBayesKind {
  kGaussian,    ///< Population means and absolute-floored variances.
  kMultinomial, ///< Smoothed nonnegative count likelihood.
  kBernoulli    ///< Smoothed binary presence and absence likelihood.
};

/// @brief Training-fitted tensor naive Bayes with empirical class priors.
///
/// Caller supplies finite floating [N,D] features and integer [N] labels in
/// 0..classes-1, with every class represented. Count features must be nonnegative;
/// Bernoulli features must be binary. Those support/range preconditions are not
/// fully checked. Gaussian uses an absolute population-variance floor. Multinomial
/// scores omit the query-dependent class-common factorial term. Training/scoring
/// use tensor operations; do not pass requires-grad training state unless a
/// differentiable fitted-statistics computation is intended.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::ml;
/// const auto x=n::tensor({2,0,1,0,0,1,0,2},{4,2});
/// const auto y=n::from_vector(std::vector<int32_t>{0,0,1,1});
/// const m::NaiveBayes model(x,y,2,m::NaiveBayesKind::kMultinomial);
/// const auto probabilities=n::softmax(model.Scores(n::tensor({1,0},{1,2})),1);
/// @endcode
///
/// @par References
/// - Manning, Raghavan and Schuetze (2008), Introduction to Information Retrieval,
/// https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
/// @see Scores()
struct NaiveBayes {
  NaiveBayesKind kind;      ///< Fitted likelihood family.
  nn::Tensor     log_prior; ///< Empirical log class priors [K].
  nn::Tensor     mean;      ///< Gaussian means [K,D], undefined for other families.
  nn::Tensor     variance;  ///< Gaussian floored variances [K,D], otherwise undefined.
  nn::Tensor probability;   ///< Multinomial/Bernoulli probabilities [K,D], undefined for Gaussian.

  /// @brief Fit class priors and conditional feature likelihoods from training rows.
  /// @param training Finite floating [N,D] features, satisfying the selected model support.
  /// @param labels Integer [N] IDs, every configured class represented.
  /// @param classes At least two classes.
  /// @param model Gaussian, multinomial or Bernoulli kind; caller supplies a valid enum.
  /// @param smoothing Positive finite additive count smoothing, default 1.
  /// @param variance_floor Positive finite absolute variance floor, default 1e-4.
  /// @throws std::invalid_argument On invalid controls or incompatible tensor operations.
  ///
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace m=n::ml;
  /// const auto x=n::tensor({2,0,1,0,0,1,0,2},{4,2});
  /// const auto y=n::from_vector(std::vector<int32_t>{0,0,1,1});
  /// const m::NaiveBayes model(x,y,2,m::NaiveBayesKind::kMultinomial);
  /// @endcode
  /// @see Scores()
  NaiveBayes(
      const nn::Tensor& training,
      const nn::Tensor& labels,
      int               classes,
      NaiveBayesKind    model,
      float             smoothing      = 1.f,
      float             variance_floor = 1e-4f)
      : kind(model) {
    if (classes < 2 || !std::isfinite(smoothing) || smoothing <= 0 ||
        !std::isfinite(variance_floor) || variance_floor <= 0)
      throw std::invalid_argument(
          "positive naive-Bayes smoothing/floor and at least two classes required");
    const auto assignment = nn::one_hot(labels, classes, 1.0, 0.0, training.dtype());
    const auto count      = nn::sum(assignment, {0});
    log_prior             = nn::log(count / nn::sum(count, {0}));
    const auto mass       = nn::matmul(assignment, training, /*transpose_a=*/true);
    if (kind == NaiveBayesKind::kGaussian) {
      mean                = mass / nn::expand_dims(count, 1);
      const auto centered = nn::expand_dims(training, 1) - nn::expand_dims(mean, 0);
      const auto spread   = nn::sum(nn::square(centered) * nn::expand_dims(assignment, 2), {0}) /
                          nn::expand_dims(count, 1);
      variance = nn::maximum(spread, nn::scalar_like(double(variance_floor), spread));
    } else if (kind == NaiveBayesKind::kMultinomial) {
      const auto smoothed = mass + double(smoothing);
      probability         = smoothed / nn::sum(smoothed, {1}, true);
    } else {
      probability = (mass + double(smoothing)) / nn::expand_dims(count + 2.0 * smoothing, 1);
    }
  }

  /// @brief Evaluate class log-joint scores using the fitted training state.
  ///
  /// Bernoulli scores include absent features. An all-zero multinomial query
  /// reduces to log priors; no multinomial coefficient is included.
  /// @param query Finite floating [Q,D], same feature semantics/dtype as training.
  /// @return Floating [Q,classes] scores; softmax along axis 1 yields posterior shares.
  /// @throws std::invalid_argument On incompatible tensor operations.
  ///
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace m=n::ml;
  /// const auto x=n::tensor({2,0,1,0,0,1,0,2},{4,2});
  /// const auto y=n::from_vector(std::vector<int32_t>{0,0,1,1});
  /// const m::NaiveBayes model(x,y,2,m::NaiveBayesKind::kMultinomial);
  /// const auto score=model.Scores(n::zeros({1,2})); // log class priors
  /// @endcode
  /// @see NaiveBayes()
  nn::Tensor Scores(const nn::Tensor& query) const {
    if (kind == NaiveBayesKind::kGaussian) {
      const auto delta      = nn::expand_dims(query, 1) - nn::expand_dims(mean, 0);
      const auto exponent   = nn::square(delta) / nn::expand_dims(variance, 0);
      const auto normalizer = nn::log(double(2 * std::acos(-1.0)) * variance);
      return log_prior - 0.5 * nn::sum(exponent + normalizer, {2});
    }
    const auto scores = nn::matmul(query, nn::log(probability), false, /*transpose_b=*/true);
    if (kind == NaiveBayesKind::kMultinomial)
      return scores + log_prior;
    const auto absent = nn::matmul(1.0 - query, nn::log1p(-probability), false, true);
    return log_prior + (scores + absent);
  }
};

} // namespace datamunge::nn::ml
