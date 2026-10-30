#pragma once
/// @file capsule_models.hpp
/// @brief Tensor capsule squash/routing/margin and full MNIST CapsNet with borrowed decoder.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <cmath>
#include <functional>
#include <limits>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Squash last-axis capsule vectors while preserving direction and a finite origin derivative.
///
/// The exact unregularized length norm²/(1+norm²) is approached when
/// norm² is large relative to epsilon. Epsilon makes the origin finite;
/// zero vectors stay zero with zero first derivative. This is not a
/// componentwise sigmoid. O(numel) tensor work, no mutation or fitting.
///
/// @param x Nonempty Float32/64 tensor with positive last vector dimension.
/// @param epsilon Finite positive denominator regularizer, default1e-9.
/// @return Same-shaped differentiable x*norm²/(1+norm²)/sqrt(norm²+epsilon).
/// @throws std::invalid_argument For invalid shapes/dtypes/dimensions/iterations, margin settings or incompatible stored leaves.
/// @par Example
/// @code
/// const auto y = dm::nn::models::Squash(dm::nn::tensor({3., 4.}, {1, 2}));
/// @endcode
/// @par References
/// - Sabour, Frosst and Hinton (2017), https://arxiv.org/abs/1710.09829
/// @see CapsNet(), DynamicRouting(), Squash(), MarginLoss()
inline nn::Tensor Squash(const nn::Tensor &x, double epsilon = 1e-9) {
  if (x.dim() < 1 || x.numel() == 0 || !(epsilon > 0) || !std::isfinite(epsilon) ||
      (x.dtype() != nn::DType::Float32 && x.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid capsule squash tensor/epsilon");
  const nn::Tensor squared = nn::sum(nn::square(x), {-1}, true);
  return squared / (squared + 1.0) * (x * nn::rsqrt(squared + epsilon));
}

/// @brief Votes, normalized routing coefficients, final capsules and iteration history.
///
/// Tensor handles own storage references and retain every routing iteration
/// in the differentiation graph. No parameter updates or fitting occur.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto routed =
///     n::models::DynamicRouting(n::ones({2, 6, 2}), n::ones({6, 9, 2}), 6, 3, 2, 3, 3);
/// @endcode
/// @par References
/// - Sabour, Frosst and Hinton (2017), https://arxiv.org/abs/1710.09829
/// @see DynamicRouting(), Squash()
struct RoutingOutputs {
  nn::Tensor votes;     ///< [N,I,J,Dout] per-input/output transformed capsule votes.
  nn::Tensor couplings; ///< Final [N,I,J] nonnegative softmax rows, sum over J is one.
  nn::Tensor capsules;  ///< [N,J,Dout] final squashed output vectors.
  std::vector<nn::Tensor>
      history; ///< One [N,I,J] coupling tensor per iteration, in execution order.
};

/// @brief Build differentiable routing by agreement through every requested iteration.
///
/// One batched product forms votes[N,I,J,Dout]. Start logits b=0;
/// each iteration computes c=softmax_J(b), s=sum_I(c*votes), v=Squash(s),
/// then b+=sum_Dout(votes*v) except after the last iteration. All iterations
/// retain gradients; no stop_gradient or learned-store mutation. The loop
/// is forward routing, not an optimizer/fitting loop. Nonnegative c rows
/// sum to one; extreme logits can underflow entries to zero.
/// Work O(N*I*J*Din*Dout + iterations*N*I*J*Dout); retained history
/// uses O(iterations*N*I*J) coefficients plus graphs/votes.
///
/// @param u Nonempty floating input capsules [N,I,Din].
/// @param weights Matching floating vote matrices [I,J*Dout,Din].
/// @param inputs Positive I matching u and weights.
/// @param outputs Positive J, product J*Dout representable.
/// @param in_dim Positive Din matching both tensors.
/// @param out_dim Positive output vector width Dout.
/// @param iterations Positive number of routing iterations, normally three.
/// @return Votes, each coupling history, final softmax couplings and squashed capsules.
/// @throws std::invalid_argument For invalid shapes/dtypes/dimensions/iterations, margin settings or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y =
///     n::models::DynamicRouting(n::ones({2, 6, 2}), n::ones({6, 9, 2}), 6, 3, 2, 3, 3);
/// @endcode
/// @par References
/// - Sabour, Frosst and Hinton (2017), https://arxiv.org/abs/1710.09829
/// @see CapsNet(), DynamicRouting(), Squash(), MarginLoss()
inline RoutingOutputs DynamicRouting(const nn::Tensor &u, const nn::Tensor &weights,
                                     std::int64_t inputs, std::int64_t outputs,
                                     std::int64_t in_dim, std::int64_t out_dim,
                                     int iterations) {
  if (inputs < 1 || outputs < 1 || in_dim < 1 || out_dim < 1 || iterations < 1 ||
      outputs > std::numeric_limits<int64_t>::max() / out_dim || u.dim() != 3 ||
      u.numel() == 0 || u.size(1) != inputs || u.size(2) != in_dim || weights.dim() != 3 ||
      weights.shape() != nn::Shape{inputs, outputs * out_dim, in_dim} ||
      weights.dtype() != u.dtype() ||
      (u.dtype() != nn::DType::Float32 && u.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid capsule routing");
  RoutingOutputs r;
  // Votes [N, I, J, out_dim] from one batched product per input capsule.
  const nn::Tensor per_capsule = nn::transpose(u, {1, 2, 0});   // [I, in_dim, N]
  const nn::Tensor products = nn::matmul(weights, per_capsule); // [I, J*out, N]
  r.votes = nn::reshape(nn::transpose(products, {2, 0, 1}), {-1, inputs, outputs, out_dim});
  nn::Tensor logits = nn::zeros_like(nn::sum(r.votes, {3})); // [N, I, J]
  for (int t = 0; t < iterations; ++t) {
    r.couplings = nn::softmax(logits, -1); // over j
    const nn::Tensor weighted =
        nn::sum(nn::expand_dims(r.couplings, 3) * r.votes, {1}); // [N, J, out]
    r.capsules = Squash(weighted);
    r.history.push_back(r.couplings);
    if (t + 1 < iterations)
      logits = logits + nn::sum(r.votes * nn::expand_dims(r.capsules, 1), {3});
  }
  return r;
}

/// @brief Compute stabilized last-axis capsule magnitudes for scores and margin loss.
///
/// Zero vectors get sqrt(1e-9), not exactly zero, and finite zero
/// input derivatives. This stabilizes CapsNet log scores. Does not
/// normalize across classes or mutate inputs. O(numel) tensor work.
///
/// @param capsules Nonempty floating vectors on the last axis.
/// @return sqrt(sum_last(capsules²)+1e-9), with the last axis removed.
/// @throws std::invalid_argument For invalid shapes/dtypes/dimensions/iterations, margin settings or incompatible stored leaves.
/// @par Example
/// @code
/// const auto lengths = dm::nn::models::CapsuleLengths(dm::nn::ones({2, 10, 16}));
/// @endcode
/// @par References
/// - Sabour, Frosst and Hinton (2017), https://arxiv.org/abs/1710.09829
/// @see CapsNet(), DynamicRouting(), Squash(), MarginLoss()
inline nn::Tensor CapsuleLengths(const nn::Tensor &capsules) {
  if (capsules.dim() < 1 || capsules.numel() == 0 ||
      (capsules.dtype() != nn::DType::Float32 && capsules.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid capsule lengths tensor");
  return nn::sqrt(nn::sum(nn::square(capsules), {-1}) + 1e-9);
}

/// @brief Compute mean per-example capsule present/absent squared-hinge margin loss.
///
/// No logarithm/softmax or reconstruction term is hidden here.
/// Soft/weighted masks are evaluated algebraically; callers supply valid
/// targets. Gradients flow through squared ReLU hinges, with the core
/// ReLU boundary convention. O(N*classes) work, no parameter mutation.
///
/// @param lengths Nonempty Float32/64 [N,classes] stabilized capsule lengths.
/// @param one_hot Same-shape/dtype target masks, normally one-hot class indicators.
/// @param plus Present-class threshold in (minus,1], default .9.
/// @param minus Absent-class threshold in [0,plus), default .1.
/// @param down Finite nonnegative absent-class weight, default .5.
/// @return Scalar mean_N sum_classes(T*relu(plus-length)² + down*(1-T)*relu(length-minus)²).
/// @throws std::invalid_argument For invalid shapes/dtypes/dimensions/iterations, margin settings or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::MarginLoss(n::full({2, 10}, .2),
///                                         n::one_hot(n::tensor({1, 2}).to(n::DType::Int64), 10));
/// @endcode
/// @par References
/// - Sabour, Frosst and Hinton (2017), https://arxiv.org/abs/1710.09829
/// @see CapsNet(), DynamicRouting(), Squash(), MarginLoss()
inline nn::Tensor MarginLoss(const nn::Tensor &lengths, const nn::Tensor &one_hot,
                             double plus = .9, double minus = .1, double down = .5) {
  if (lengths.dim() != 2 || lengths.numel() == 0 || one_hot.shape() != lengths.shape() ||
      one_hot.dtype() != lengths.dtype() ||
      (lengths.dtype() != nn::DType::Float32 && lengths.dtype() != nn::DType::Float64) ||
      !(minus >= 0 && minus < plus && plus <= 1) || !(down >= 0) || !std::isfinite(down))
    throw std::invalid_argument("invalid capsule margin loss shapes/settings");
  const nn::Tensor present = nn::square(nn::relu(plus - lengths));
  const nn::Tensor absent = nn::square(nn::relu(lengths - minus));
  const nn::Tensor per_class = one_hot * present + (1.0 - one_hot) * absent * down;
  return nn::mean(nn::sum(per_class, {1}));
}

/// @brief Primary/digit capsules, scores and an optional true/predicted-mask decoder.
///
/// Tensor handles retain graphs. The decoder closure captures the digit tensor
/// and a POINTER to the borrowed parameter store; that store and unchanged
/// leaves must outlive closure calls/backward. A new forward is needed after
/// optimizer/checkpoint mutation. The callable is empty when decoder=false.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::CapsNet(p, n::ones({1, 28, 28, 1}));
/// const auto image = y.reconstruct(n::one_hot(n::argmax(y.logits, 1), 10));
/// @endcode
/// @par References
/// - Sabour, Frosst and Hinton (2017), https://arxiv.org/abs/1710.09829
/// @see CapsNet(), CapsuleLengths(), MarginLoss()
struct CapsNetOutputs {
  nn::Tensor primary; ///< [N,1152,8] squashed 32-type primary capsules.
  nn::Tensor digits;  ///< [N,10,16] routed digit vectors.
  nn::Tensor lengths; ///< [N,10] stabilized capsule lengths, sqrt(sum(v²)+1e-9).
  nn::Tensor
      logits; ///< log(lengths); softmax gives normalized lengths, not calibrated probabilities.
  std::function<nn::Tensor(const nn::Tensor &mask)>
      reconstruct; ///< Optional matching-dtype [N,10] mask -> [N,784] sigmoid pixels; borrows p.
};

/// @brief Build the complete MNIST capsule network and optional reconstruction decoder without fitting.
///
/// VALID biased9x9/256/ReLU gives20x20; biased9x9/2/256 forms
/// 6x6x32 primary 8D vectors, squashed into 1152 capsules. Normal(.01)
/// vote matrices route to ten 16D digit capsules. log(stabilized length)
/// serves as logits: softmax normalizes lengths and is not a calibration
/// guarantee. The decoder masks digit vectors, flattens160 features,
/// then Glorot dense 512/ReLU,1024/ReLU,784/sigmoid. All decoder
/// variables are created before returning, so optimizer construction may
/// follow an inference warmup without calling the closure first.
/// Fresh stores learn 6804224 scalars without decoder or 8215568 with it.
/// Convolution kernels use He-uniform. No BN/dropout, preprocessing,
/// margin/reconstruction loss, optimizer or fitting is hidden. The closure
/// borrows p; keep it and unchanged parameter leaves alive until backward.
/// Synchronize shared store access and create fresh graphs after updates.
///
/// @param p Mutable floating learned store; all requested leaves lazily created before return.
/// @param image Nonempty matching-dtype [N,28,28,1] grayscale images, caller supplies pixel scaling.
/// @param iterations Positive routing count, default three.
/// @param decoder True by default creates decoder leaves/closure; false leaves callable empty.
/// @return Primary/digit capsules, stabilized lengths/log scores and optional masked reconstruction.
/// @throws std::invalid_argument For invalid shapes/dtypes/dimensions/iterations, margin settings or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::CapsNet(p, n::ones({1, 28, 28, 1}));
/// const auto reconstruction = y.reconstruct(n::one_hot(n::argmax(y.logits, 1), 10));
/// @endcode
/// @par References
/// - Sabour, Frosst and Hinton (2017), https://arxiv.org/abs/1710.09829
/// @see CapsNet(), DynamicRouting(), Squash(), MarginLoss()
inline CapsNetOutputs CapsNet(nn::ParameterStore &p, const nn::Tensor &image,
                              int iterations = 3, bool decoder = true) {
  if (image.dim() != 4 || image.numel() == 0 || image.size(1) != 28 || image.size(2) != 28 ||
      image.size(3) != 1 || image.dtype() != p.dtype() || iterations < 1)
    throw std::invalid_argument(
        "CapsNet needs matching floating NHWC28x28 grayscale images and positive iterations");
  CapsNetOutputs out;
  nn::ConvLayerOptions cfg;
  cfg.init = nn::init::he_uniform();
  cfg.padding = nn::Padding::Valid;
  cfg.activation = nn::Activation::Relu;
  const nn::Tensor h = nn::conv2d(p, "conv1", image, 256, 9, cfg); // 20x20x256
  cfg.activation = nn::Activation::None;
  cfg.stride = 2;
  const nn::Tensor primary = nn::conv2d(p, "primary", h, 256, 9, cfg); // 6x6x(32x8)
  out.primary = Squash(nn::reshape(primary, {-1, 1152, 8}));
  const nn::Tensor w = p.get("digit/W", {1152, 160, 8}, nn::init::normal(.01));
  out.digits = DynamicRouting(out.primary, w, 1152, 10, 8, 16, iterations).capsules;
  out.lengths = CapsuleLengths(out.digits);
  out.logits = nn::log(out.lengths); // softmax(log |v|) = |v| / sum |v|
  if (decoder) {
    // Create the decoder's variables now (tflow builds it once on the unmasked capsules).
    const std::int64_t widths[4] = {160, 512, 1024, 784};
    for (int i = 0; i < 3; ++i) {
      const std::string name = "decoder/fc" + std::to_string(i + 1);
      p.get(name + "/W", {widths[i], widths[i + 1]}, nn::init::glorot_uniform());
      p.get(name + "/b", {widths[i + 1]}, nn::init::zeros());
    }
    const nn::Tensor digits = out.digits;
    nn::ParameterStore *store = &p;
    out.reconstruct = [store, digits](const nn::Tensor &mask) {
      if (mask.dim() != 2 || mask.size(0) != digits.size(0) || mask.size(1) != 10 ||
          mask.dtype() != digits.dtype())
        throw std::invalid_argument(
            "CapsNet decoder mask must match batch,10 classes and digit dtype");
      nn::Tensor d = nn::reshape(digits * nn::expand_dims(mask, 2), {-1, 160});
      d = nn::relu(nn::dense(*store, "decoder/fc1", d, 512));
      d = nn::relu(nn::dense(*store, "decoder/fc2", d, 1024));
      return nn::sigmoid(nn::dense(*store, "decoder/fc3", d, 784));
    };
  }
  return out;
}
} // namespace datamunge::nn::models
