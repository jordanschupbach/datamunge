#include "datamunge/nn/losses.hpp"

#include <stdexcept>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

Tensor reduce(const Tensor& loss, Reduction reduction) {
  switch (reduction) {
    case Reduction::None: return loss;
    case Reduction::Mean: return mean(loss);
    case Reduction::Sum: return sum(loss);
  }
  return loss;
}

namespace {
void check_same_shape(const Tensor& a, const Tensor& b, const char* what) {
  if (a.shape() != b.shape())
    throw std::invalid_argument(std::string(what) + ": shapes " + shape_str(a.shape()) + " and " +
                                shape_str(b.shape()) + " differ");
}
}  // namespace

Tensor softmax_cross_entropy(const Tensor& logits, const Tensor& targets, Reduction reduction, double label_smoothing,
                             std::int64_t axis) {
  check_same_shape(logits, targets, "softmax_cross_entropy");
  if (!(label_smoothing >= 0 && label_smoothing < 1))
    throw std::invalid_argument("softmax_cross_entropy: label_smoothing must be in [0, 1)");
  Tensor t = cast(targets, to_floating(logits.dtype()));
  if (label_smoothing > 0) {
    const double classes = static_cast<double>(logits.size(axis));
    t                    = add(mul(t, 1.0 - label_smoothing), label_smoothing / classes);
  }
  return reduce(neg(sum(mul(t, log_softmax(logits, axis)), {axis})), reduction);
}

Tensor sparse_softmax_cross_entropy(const Tensor& logits, const Tensor& labels, Reduction reduction,
                                    double label_smoothing) {
  if (logits.dim() != labels.dim() + 1)
    throw std::invalid_argument("sparse_softmax_cross_entropy: labels " + shape_str(labels.shape()) +
                                " must have one axis fewer than logits " + shape_str(logits.shape()));
  const Tensor targets = one_hot(labels, logits.shape().back(), 1.0, 0.0, to_floating(logits.dtype()));
  return softmax_cross_entropy(logits, targets, reduction, label_smoothing, -1);
}

Tensor sigmoid_cross_entropy(const Tensor& logits, const Tensor& labels, Reduction reduction) {
  const Tensor y                  = cast(labels, to_floating(logits.dtype()));
  const Tensor positive           = greater_equal(logits, 0.0);
  const Tensor negative_magnitude = where(positive, neg(logits), logits);
  const Tensor linear             = where(positive, mul(logits, sub(1.0, y)), neg(mul(logits, y)));
  return reduce(add(linear, log1p(exp(negative_magnitude))), reduction);
}

Tensor mse(const Tensor& prediction, const Tensor& target, Reduction reduction) {
  return reduce(squared_difference(prediction, target), reduction);
}

Tensor mae(const Tensor& prediction, const Tensor& target, Reduction reduction) {
  return reduce(abs(sub(prediction, target)), reduction);
}

Tensor huber(const Tensor& error, double delta) {
  if (!(delta > 0)) throw std::invalid_argument("huber: delta must be positive");
  const Tensor d = abs(error);
  const Tensor q = minimum(d, scalar_like(delta, d));
  return add(mul(square(q), 0.5), mul(sub(d, q), delta));
}

Tensor huber_loss(const Tensor& prediction, const Tensor& target, double delta, Reduction reduction) {
  return reduce(huber(sub(prediction, target), delta), reduction);
}

Tensor kl_gaussian(const Tensor& mu_q, const Tensor& logvar_q, const Tensor& mu_p, const Tensor& logvar_p) {
  // 0.5 (logvar_p - logvar_q + (exp(logvar_q) + (mu_q - mu_p)^2) / exp(logvar_p) - 1)
  return mul(sub(add(sub(logvar_p, logvar_q), div(add(exp(logvar_q), squared_difference(mu_q, mu_p)), exp(logvar_p))),
                 1.0),
             0.5);
}

Tensor kl_standard_normal(const Tensor& mu, const Tensor& logvar) {
  // 0.5 (exp(logvar) + mu^2 - 1 - logvar)
  return mul(sub(sub(add(exp(logvar), square(mu)), 1.0), logvar), 0.5);
}

Tensor hinge(const Tensor& scores, const Tensor& labels, Reduction reduction) {
  return reduce(relu(sub(1.0, mul(cast(labels, to_floating(scores.dtype())), scores))), reduction);
}

}  // namespace datamunge::nn
