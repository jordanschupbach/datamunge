#pragma once

/// @file losses.hpp
/// @brief Loss functions. All are differentiable and numerically stable.
///
/// Losses that compare predictions with targets return the per-example (or
/// per-element) loss reduced by `reduction`: Mean (default), Sum or None.

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

enum class Reduction { None, Mean, Sum };

/// Applies a reduction over all elements.
Tensor reduce(const Tensor& loss, Reduction reduction);

/// Cross-entropy between target distributions and softmax(logits) along
/// `axis`: -sum(targets * log_softmax(logits)). With label smoothing the
/// targets become (1 - s) targets + s / classes.
Tensor softmax_cross_entropy(const Tensor& logits, const Tensor& targets, Reduction reduction = Reduction::Mean,
                             double label_smoothing = 0.0, std::int64_t axis = -1);

/// Cross-entropy with integer class labels (shape = logits shape without
/// the last axis). TF SparseSoftmaxCrossEntropyWithLogits.
Tensor sparse_softmax_cross_entropy(const Tensor& logits, const Tensor& labels, Reduction reduction = Reduction::Mean,
                                    double label_smoothing = 0.0);

/// Elementwise binary cross-entropy from logits with labels in [0, 1]:
/// stable for any |logit|, and exactly sigmoid(z) - y as the derivative at z = 0.
Tensor sigmoid_cross_entropy(const Tensor& logits, const Tensor& labels, Reduction reduction = Reduction::Mean);

Tensor mse(const Tensor& prediction, const Tensor& target, Reduction reduction = Reduction::Mean);
Tensor mae(const Tensor& prediction, const Tensor& target, Reduction reduction = Reduction::Mean);

/// Elementwise Huber function of an error: 0.5 e^2 for |e| <= delta, delta (|e| - delta / 2) beyond.
Tensor huber(const Tensor& error, double delta = 1.0);
Tensor huber_loss(const Tensor& prediction, const Tensor& target, double delta = 1.0,
                  Reduction reduction = Reduction::Mean);

/// Elementwise KL(N(mu_q, exp(logvar_q)) || N(mu_p, exp(logvar_p))).
Tensor kl_gaussian(const Tensor& mu_q, const Tensor& logvar_q, const Tensor& mu_p, const Tensor& logvar_p);
/// Elementwise KL(N(mu, exp(logvar)) || N(0, 1)) (the VAE term).
Tensor kl_standard_normal(const Tensor& mu, const Tensor& logvar);

/// max(0, 1 - y * score) with labels y in {-1, +1}.
Tensor hinge(const Tensor& scores, const Tensor& labels, Reduction reduction = Reduction::Mean);

}  // namespace datamunge::nn
