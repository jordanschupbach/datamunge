/// @file
/// @brief Paired image fixtures and differentiable SimCLR components.
#pragma once

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/losses.hpp>
#include <datamunge/nn/ops.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Own a synthetic image batch and two independently augmented views.
/// @par Example
/// @code
/// std::mt19937 rng(7);
/// const auto batch = dm::nn::synthetic_contrastive_views(rng);
/// @endcode
/// @see synthetic_contrastive_views()
struct ContrastiveViews {
  Tensor images;    ///< Source image surfaces [B,8,8].
  Tensor views;     ///< Flattened crops [2B,36], with pairs B rows apart.
  Tensor transform; ///< Crop row, crop column, gain and offset [2B,4].
};

/// @brief Generate paired crops of random polynomial grayscale surfaces.
///
/// Each image uses four independent coefficients sampled uniformly from
/// [-0.7,0.7]. Each view has its own 6x6 crop and affine intensity transform.
/// All outputs are Float32. The caller owns and advances the random engine.
///
/// @param rng Random engine, consumed in image-major then view-major order.
/// @param count Number of source images, default 32.
/// @param stronger Use gain [0.65,1.35] and offset [-0.22,0.22] when true.
/// The default uses gain [0.8,1.2] and offset [-0.15,0.15].
/// @return Source images, flattened views and augmentation metadata.
/// @throws std::invalid_argument If count is nonpositive.
/// @par Example
/// @code
/// std::mt19937 rng(7);
/// const auto batch = dm::nn::synthetic_contrastive_views(rng, 4);
/// @endcode
/// @see ContrastiveViews
inline ContrastiveViews synthetic_contrastive_views(std::mt19937 &rng, int count = 32,
                                                     bool stronger = false) {
  if (count < 1)
    throw std::invalid_argument("synthetic_contrastive_views: positive image count required");
  const std::int64_t batch = count;
  std::uniform_real_distribution<float> coefficient(-.7f, .7f);
  std::uniform_real_distribution<float> gain(stronger ? .65f : .8f, stronger ? 1.35f : 1.2f);
  std::uniform_real_distribution<float> offset(stronger ? -.22f : -.15f, stronger ? .22f : .15f);
  std::vector<float> images(batch * 64), views(2 * batch * 36), transforms;
  for (std::int64_t row = 0; row < batch; ++row) {
    const float a = coefficient(rng), b = coefficient(rng);
    const float c = coefficient(rng), d = coefficient(rng);
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        const float u = (x - 3.5f) / 4, v = (y - 3.5f) / 4;
        images[row * 64 + y * 8 + x] = a*u + b*v + c*u*v + d*(u*u-v*v);
      }
  }
  transforms.reserve(2 * batch * 4);
  for (std::int64_t row = 0; row < 2 * batch; ++row) {
    const int y0 = rng() % 3, x0 = rng() % 3;
    const float g = gain(rng), b = offset(rng);
    transforms.insert(transforms.end(), {float(y0), float(x0), g, b});
    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 6; ++x)
        views[row * 36 + y * 6 + x] =
            g * images[(row % batch) * 64 + (y + y0) * 8 + x + x0] + b;
  }
  return {from_vector(images, {batch, 8, 8}), from_vector(views, {2 * batch, 36}),
          from_vector(transforms, {2 * batch, 4})};
}

/// @brief Own SimCLR's normalized embeddings, similarity scores and training loss.
/// @par Example
/// @code
/// const auto path = dm::nn::nt_xent(dm::nn::ones({4, 3}));
/// @endcode
/// @see nt_xent()
struct ContrastiveObjective {
  Tensor unit;       ///< Row-normalized projections [2B,D].
  Tensor similarity; ///< Temperature-scaled pair scores [2B,2B] before masking.
  Tensor nearest;    ///< Row index of the highest-scoring other view [2B].
  Tensor loss;       ///< Scalar symmetric NT-Xent mean over all anchors.
};

/// @brief Calculate symmetric normalized temperature-scaled cross entropy.
///
/// Positive pairs are B rows apart. The denominator includes the positive
/// and every other view except the anchor. Squared row norms have a floor of
/// 1e-8. Work and score storage scale quadratically in batch size.
///
/// @param projection Finite floating tensor [2B,D], with B and D positive.
/// @param temperature Positive finite divisor for cosine similarities, default 0.2.
/// @return Differentiable loss and projections with nearest-view diagnostics.
/// @throws std::invalid_argument For a non-floating projection, invalid shape or invalid temperature.
/// @par Example
/// @code
/// const auto path = dm::nn::nt_xent(dm::nn::ones({4, 3}));
/// @endcode
/// @par References
/// - Chen et al. (2020), A Simple Framework for Contrastive Learning of Visual Representations.
///   https://proceedings.mlr.press/v119/chen20j.html
/// @see simclr_projection(), synthetic_contrastive_views()
inline ContrastiveObjective nt_xent(const Tensor &projection, double temperature = .2) {
  if (projection.shape().size() != 2 || projection.shape()[0] < 2 ||
      projection.shape()[0] % 2 || projection.shape()[1] < 1 ||
      !is_floating(projection.dtype()) || !std::isfinite(temperature) || temperature <= 0)
    throw std::invalid_argument("nt_xent: expected two equal view groups and positive temperature");
  const auto rows = projection.shape()[0];
  const auto type = projection.dtype();
  const auto unit = projection * rsqrt(maximum(sum(square(projection), {1}, true),
                                                scalar(1e-8, type)));
  const auto similarity = matmul(unit, unit.transpose()) / temperature;
  const auto row = arange(0, rows, 1., DType::Int64);
  const auto positive = concat({slice_axis(row, 0, rows / 2, rows),
                                slice_axis(row, 0, 0, rows / 2)}, 0);
  const auto mask = equal(row.reshape({rows, 1}), row.reshape({1, rows}));
  const auto eligible = where(mask, scalar(-std::numeric_limits<double>::infinity(), type),
                              similarity);
  const auto paired = sum(similarity * one_hot(positive, rows, 1., 0., type), {1});
  return {unit, similarity, argmax(eligible.detach(), 1),
          mean(logsumexp(eligible, {1}) - paired)};
}

/// @brief Encode flattened views and apply the compact SimCLR projection MLP.
///
/// The fixture architecture uses a ReLU encoder with 32 outputs, then a
/// 24-unit ReLU hidden layer and a linear 12-unit projection. Parameter
/// storage is reused when called again with the same prefix and input width.
///
/// @param store Owns the trainable weights and biases.
/// @param name Prefix for encoder and projector parameter names.
/// @param views Floating tensor [2B,D] of flattened views.
/// @return Unnormalized projections [2B,12].
/// @throws std::invalid_argument If the dense layer cannot use the input shape.
/// @par Example
/// @code
/// dm::nn::ParameterStore store(7);
/// const auto z = dm::nn::simclr_projection(store, "simclr", dm::nn::ones({4,36}));
/// @endcode
/// @par References
/// - Chen et al. (2020). https://proceedings.mlr.press/v119/chen20j.html
/// @see nt_xent()
inline Tensor simclr_projection(ParameterStore &store, const std::string &name,
                                const Tensor &views) {
  const auto encoded = dense(store, name + "/encoder", views, 32, Activation::Relu);
  const auto hidden = dense(store, name + "/project", encoded, 24, Activation::Relu);
  return dense(store, name + "/project_out", hidden, 12);
}

/// @brief Differentiable VICReg loss, individual penalties and batch covariances.
/// @par Example
/// @code
/// const auto result = dm::nn::vicreg(dm::nn::zeros({4, 3}));
/// @endcode
/// @see vicreg()
struct VICRegResult {
  Tensor loss;       ///< Weighted scalar objective.
  Tensor invariance; ///< Mean squared difference between paired projections.
  Tensor variance;   ///< Average variance hinge over both views and all columns.
  Tensor covariance; ///< Sum of both off-diagonal covariance penalties.
  Tensor cov_left;   ///< Sample covariance of the first view group [D,D].
  Tensor cov_right;  ///< Sample covariance of the second view group [D,D].
};

/// @brief Compute variance, invariance and covariance regularization on paired projections.
///
/// The variance hinge targets standard deviation one, with 1e-4 added to sample
/// variances. The covariance penalty sums squared off-diagonal entries divided
/// by the width for each view group. The three penalties use weights 25, 25, 1.
///
/// @param projection Finite floating [2B,D] tensor, pairs B rows apart, B >= 2 and D >= 1.
/// @return Scalar objective, its terms and sample covariance matrices.
/// @throws std::invalid_argument For invalid shape or nonfloating input.
/// @par Example
/// @code
/// const auto result = dm::nn::vicreg(dm::nn::zeros({4, 3}));
/// @endcode
/// @par References
/// - Bardes et al. (2022), https://arxiv.org/abs/2105.04906
/// @see barlow_twins()
inline VICRegResult vicreg(const Tensor &projection) {
  if (!is_floating(projection.dtype()) || projection.dim() != 2 ||
      projection.shape()[0] < 4 || projection.shape()[0] % 2 || projection.shape()[1] < 1)
    throw std::invalid_argument("vicreg: expected two groups of at least two floating rows");
  const auto batch = projection.shape()[0] / 2, width = projection.shape()[1];
  const auto left = slice_axis(projection, 0, 0, batch);
  const auto right = slice_axis(projection, 0, batch, 2 * batch);
  const auto off = 1. - eye(width, projection.dtype());
  const auto stats = [&](const Tensor &view) {
    const auto centered = view - mean(view, {0}, true);
    const auto covariance = matmul(centered.transpose(), centered) / double(batch - 1);
    const auto variance = sum(square(centered), {0}) / double(batch - 1);
    return std::array{covariance, mean(relu(1. - sqrt(variance + 1e-4))),
                      sum(square(off * covariance)) / double(width)};
  };
  const auto a = stats(left), b = stats(right);
  const auto invariance = mean(square(left - right));
  const auto variance = .5 * (a[1] + b[1]);
  const auto covariance = a[2] + b[2];
  return {25. * (invariance + variance) + covariance, invariance, variance, covariance, a[0], b[0]};
}

/// @brief Differentiable Barlow Twins loss and cross-view diagnostics.
/// @par Example
/// @code
/// const auto result = dm::nn::barlow_twins(dm::nn::zeros({4, 3}));
/// @endcode
/// @see barlow_twins()
struct BarlowTwinsResult {
  Tensor loss;         ///< Diagonal penalty plus weighted off-diagonal penalty.
  Tensor diagonal;     ///< Sum of squared diagonal errors from one.
  Tensor off_diagonal; ///< Sum of squared off-diagonal correlations.
  Tensor correlation;  ///< Cross-view feature correlation [D,D].
  Tensor left;         ///< Standardized first view group [B,D].
  Tensor right;        ///< Standardized second view group [B,D].
};

/// @brief Penalize departures of the cross-view correlation from the identity.
///
/// Each view group is centered and scaled columnwise using its population
/// variance plus 1e-5. Off-diagonal errors receive the given relative weight.
///
/// @param projection Finite floating [2B,D] tensor, pairs B rows apart, B >= 2 and D >= 1.
/// @param redundancy_weight Finite nonnegative off-diagonal weight, default 0.02.
/// @return Scalar objective, its terms, standardized views and their correlation.
/// @throws std::invalid_argument For invalid shape, dtype or weight.
/// @par Example
/// @code
/// const auto result = dm::nn::barlow_twins(dm::nn::zeros({4, 3}));
/// @endcode
/// @par References
/// - Zbontar et al. (2021), https://proceedings.mlr.press/v139/zbontar21a.html
/// @see vicreg()
inline BarlowTwinsResult barlow_twins(const Tensor &projection, double redundancy_weight = .02) {
  if (!is_floating(projection.dtype()) || projection.dim() != 2 ||
      projection.shape()[0] < 4 || projection.shape()[0] % 2 || projection.shape()[1] < 1 ||
      !std::isfinite(redundancy_weight) || redundancy_weight < 0)
    throw std::invalid_argument("barlow_twins: invalid projection or redundancy weight");
  const auto batch = projection.shape()[0] / 2;
  const auto normalize = [](const Tensor &view) {
    const auto centered = view - mean(view, {0}, true);
    return centered / sqrt(mean(square(centered), {0}, true) + 1e-5);
  };
  const auto left = normalize(slice_axis(projection, 0, 0, batch));
  const auto right = normalize(slice_axis(projection, 0, batch, 2 * batch));
  const auto correlation = matmul(left.transpose(), right) / double(batch);
  const auto identity = eye(projection.shape()[1], projection.dtype());
  const auto errors = square(correlation - identity);
  const auto diagonal = sum(identity * errors);
  const auto off_diagonal = sum((1. - identity) * errors);
  return {diagonal + redundancy_weight * off_diagonal, diagonal, off_diagonal,
          correlation, left, right};
}

/// @brief Implementation functions for paired representation models.
namespace contrastive_detail {

/// @brief Validate paired geometry and apply the shared 32/32/12 projection.
/// @param store Owns the parameters.
/// @param name Layer prefix.
/// @param views Paired flattened floating inputs.
/// @return Raw projections.
/// @throws std::invalid_argument On invalid input geometry.
inline Tensor siamese_projection(ParameterStore &store, const std::string &name,
                                  const Tensor &views) {
  if (!is_floating(views.dtype()) || views.dim() != 2 || views.shape()[0] < 2 ||
      views.shape()[0] % 2 || views.shape()[1] < 1)
    throw std::invalid_argument("siamese projection: expected equal floating view groups");
  const auto encoded = dense(store, name + "/encoder", views, 32, Activation::Relu);
  const auto hidden = dense(store, name + "/project", encoded, 32, Activation::Relu);
  return dense(store, name + "/output", hidden, 12);
}

/// @brief Normalize rows with a squared norm floor of 1e-8.
/// @param x Floating row matrix.
/// @return Row-normalized matrix.
inline Tensor row_unit(const Tensor &x) {
  return x * rsqrt(maximum(sum(square(x), {1}, true), scalar(1e-8, x.dtype())));
}

/// @brief Swap the two view groups of a paired matrix.
/// @param x Matrix with an even number of rows.
/// @return The second half followed by the first half.
inline Tensor swap_views(const Tensor &x) {
  const auto batch = x.shape()[0] / 2;
  return concat({slice_axis(x, 0, batch, 2 * batch), slice_axis(x, 0, 0, batch)}, 0);
}

}  // namespace contrastive_detail

/// @brief Own the compact SimSiam forward path and its symmetric objective.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto result = dm::nn::simsiam(model, "siam", dm::nn::ones({4, 36}));
/// @endcode
/// @see simsiam()
struct SimSiamResult {
  Tensor projection; ///< Raw shared encoder and projector output [2B,12].
  Tensor online;     ///< Unit-normalized predictor output [2B,12].
  Tensor target;     ///< Detached normalized projection in original view order [2B,12].
  Tensor per_view;   ///< Negative cosine with the paired target [2B].
  Tensor loss;       ///< Mean negative cosine over both view directions.
};

/// @brief Build the compact SimSiam model with stopped cross-view targets.
///
/// A shared 32/32/12 encoder and projector processes both view groups.
/// A 24/12 predictor operates on the raw projections. The objective
/// differentiates through the predictor and its inputs. Target rows are
/// detached before matching to their partner B rows away. Normalization
/// floors squared norms at 1e-8, so zero rows remain finite.
///
/// @param store Owns the trainable weights and biases.
/// @param name Prefix for parameters that will be reused on later calls.
/// @param views Finite floating matrix [2B,D] with positive B and D.
/// @return Projection, normalized outputs and negative cosine losses.
/// @throws std::invalid_argument For invalid view geometry or incompatible parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto result = dm::nn::simsiam(model, "siam", dm::nn::ones({4, 36}));
/// @endcode
/// @par References
/// - Chen and He (2021), Exploring Simple Siamese Representation Learning.
///   https://arxiv.org/abs/2011.10566
/// @see synthetic_contrastive_views()
inline SimSiamResult simsiam(ParameterStore &store, const std::string &name,
                             const Tensor &views) {
  const auto projection = contrastive_detail::siamese_projection(store, name, views);
  const auto predicted = dense(store, name + "/predictor_hidden", projection, 24, Activation::Relu);
  const auto online = contrastive_detail::row_unit(
      dense(store, name + "/predictor_output", predicted, 12));
  const auto target = contrastive_detail::row_unit(projection.detach());
  const auto per_view = -sum(online * contrastive_detail::swap_views(target), {1});
  return {projection, online, target, per_view, mean(per_view)};
}

/// @brief Own normalized BYOL representations and squared alignment losses.
/// @par Example
/// @code
/// dm::nn::ParameterStore online(7), target(8), predictor(9);
/// const auto result = dm::nn::byol(online, target, predictor, dm::nn::ones({4,36}));
/// @endcode
/// @see byol()
struct BYOLResult {
  Tensor online;   ///< Normalized predictor rows [2B,12].
  Tensor target;   ///< Detached normalized target projection rows [2B,12].
  Tensor per_view; ///< Squared distance to partner target [2B].
  Tensor loss;     ///< Mean squared distance over both directions.
};

/// @brief Compare online predictions with detached momentum-encoder targets.
///
/// Use a first forward pass to allocate both encoders, then copy online
/// parameters to target with ParameterStore::copy_from(). After each optimizer
/// update on online and predictor only, call target.copy_from(online, 1-momentum).
/// This function initializes missing parameters and otherwise reads their values.
///
/// @param online Owns the trainable 32/32/12 encoder and projector.
/// @param target Owns the independently stored EMA encoder of the same geometry.
/// @param predictor Owns the trainable 24/12 prediction layers.
/// @param views Finite floating matrix [2B,D] with positive B and D.
/// @return Detached target and differentiable online path and objective.
/// @throws std::invalid_argument For aliased stores, invalid view geometry or parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore online(7), target(8), predictor(9);
/// const auto views = dm::nn::ones({4,36});
/// dm::nn::byol(online, target, predictor, views);
/// target.copy_from(online);
/// const auto result = dm::nn::byol(online, target, predictor, views);
/// @endcode
/// @par References
/// - Grill et al. (2020), Bootstrap Your Own Latent. https://arxiv.org/abs/2006.07733
/// @see simsiam(), ParameterStore::copy_from()
inline BYOLResult byol(ParameterStore &online, ParameterStore &target,
                       ParameterStore &predictor, const Tensor &views) {
  if (&online == &target || &online == &predictor || &target == &predictor)
    throw std::invalid_argument("byol: stores must be independent");
  const auto projection = contrastive_detail::siamese_projection(online, "byol", views);
  Tensor target_unit;
  {
    NoGradGuard guard;
    target_unit = contrastive_detail::row_unit(
        contrastive_detail::siamese_projection(target, "byol", views));
  }
  const auto hidden = dense(predictor, "byol/hidden", projection, 24, Activation::Relu);
  const auto online_unit = contrastive_detail::row_unit(dense(predictor, "byol/output", hidden, 12));
  const auto per_view = sum(square(online_unit - contrastive_detail::swap_views(target_unit)), {1});
  return {online_unit, target_unit, per_view, mean(per_view)};
}

/// @brief Return the next fixed-capacity FIFO dictionary after appending keys.
///
/// The returned dictionary owns detached storage. If the new keys fill the
/// queue, only the newest capacity rows are retained.
///
/// @param queue Existing floating dictionary [K,D], K and D positive.
/// @param keys Floating append batch [B,D] of the same dtype, B positive.
/// @return Updated dictionary [K,D] in oldest-to-newest order.
/// @throws std::invalid_argument For incompatible queue or keys.
/// @par Example
/// @code
/// const auto next = dm::nn::moco_enqueue(dm::nn::zeros({4,3}), dm::nn::ones({2,3}));
/// @endcode
/// @see moco()
inline Tensor moco_enqueue(const Tensor &queue, const Tensor &keys) {
  if (queue.dim() != 2 || keys.dim() != 2 || queue.shape()[0] < 1 ||
      keys.shape()[0] < 1 || queue.shape()[1] < 1 || queue.shape()[1] != keys.shape()[1] ||
      !is_floating(queue.dtype()) || queue.dtype() != keys.dtype())
    throw std::invalid_argument("moco_enqueue: incompatible queue or keys");
  const auto capacity = queue.shape()[0], count = keys.shape()[0];
  NoGradGuard guard;
  if (count >= capacity)
    return slice_axis(keys, 0, count - capacity, count).detach().clone();
  return concat({slice_axis(queue.detach(), 0, count, capacity), keys.detach()}, 0);
}

/// @brief Compute the compact MoCo contrastive loss for normalized queries and keys.
///
/// Versions 1 and 2 use positives in column zero and detached FIFO negatives.
/// Version 3 uses both directions of in-batch classification and multiplies
/// their summed cross-entropies by twice the temperature. Keys are detached.
///
/// @param query Floating [2B,D] normalized queries in paired view order.
/// @param key Floating [2B,D] normalized keys in the same order and dtype.
/// @param queue Floating [K,D] negatives for versions 1 and 2, ignored for version 3.
/// @param version 1, 2 or 3.
/// @param temperature Positive finite softmax temperature, default 0.2.
/// @return Scalar differentiable contrastive loss.
/// @throws std::invalid_argument For invalid geometry, version or temperature.
/// @par Example
/// @code
/// const auto q = dm::nn::ones({4,3});
/// const auto loss = dm::nn::moco_loss(q, q, dm::nn::zeros({4,3}), 1);
/// @endcode
/// @par References
/// - He et al. (2020), https://arxiv.org/abs/1911.05722
/// - Chen et al. (2021), https://arxiv.org/abs/2104.02057
/// @see moco(), moco_enqueue()
inline Tensor moco_loss(const Tensor &query, const Tensor &key, const Tensor &queue,
                         int version, double temperature = .2) {
  if (query.dim() != 2 || query.shape() != key.shape() || query.shape()[0] < 2 ||
      query.shape()[0] % 2 || query.shape()[1] < 1 || !is_floating(query.dtype()) ||
      query.dtype() != key.dtype() || version < 1 || version > 3 ||
      !std::isfinite(temperature) || temperature <= 0)
    throw std::invalid_argument("moco_loss: invalid projections, version or temperature");
  const auto batch = query.shape()[0] / 2;
  const auto first = slice_axis(query, 0, 0, batch);
  const auto positive_keys = slice_axis(key.detach(), 0, batch, 2 * batch);
  if (version == 3) {
    const auto second = slice_axis(query, 0, batch, 2 * batch);
    const auto reverse_keys = slice_axis(key.detach(), 0, 0, batch);
    const auto labels = arange(0, batch, 1., DType::Int64);
    return (2. * temperature) *
           (sparse_softmax_cross_entropy(matmul(first, positive_keys.transpose()) / temperature, labels) +
            sparse_softmax_cross_entropy(matmul(second, reverse_keys.transpose()) / temperature, labels));
  }
  if (queue.dim() != 2 || queue.shape()[0] < 1 || queue.shape()[1] != query.shape()[1] ||
      queue.dtype() != query.dtype())
    throw std::invalid_argument("moco_loss: invalid negative dictionary");
  const auto positives = sum(first * positive_keys, {1}, true);
  const auto negatives = matmul(first, queue.detach().transpose());
  return sparse_softmax_cross_entropy(concat({positives, negatives}, 1) / temperature,
                                      zeros({batch}, DType::Int64));
}

/// @brief Own the current MoCo representations and loss.
/// @par Example
/// @code
/// dm::nn::ParameterStore online(7), target(8), predictor(9);
/// auto out = dm::nn::moco(online, target, predictor, dm::nn::ones({4,36}), {}, 3);
/// @endcode
/// @see moco()
struct MoCoResult {
  Tensor query;        ///< Normalized online rows [2B,12].
  Tensor key;          ///< Detached normalized target rows [2B,12].
  Tensor forward_keys; ///< Target rows for the second group [B,12].
  Tensor loss;         ///< Scalar contrastive loss.
};

/// @brief Evaluate the compact MoCo v1, v2 or v3 forward path.
///
/// The v1 projector is linear after a ReLU encoder. Versions 2 and 3 add
/// a ReLU projection layer. Version 3 also adds an online predictor.
/// First allocate both encoders with a forward pass, then synchronize with
/// target.copy_from(online). Before each training forward, update target
/// with target.copy_from(online, 1-momentum). Optimize online and predictor
/// only. For queue versions append forward_keys after the optimizer step.
///
/// @param online Trainable encoder and projector store.
/// @param target Independently owned EMA encoder and projector store.
/// @param predictor Independent predictor store, used by version 3.
/// @param views Finite floating [2B,D] paired crop matrix.
/// @param queue Negative dictionary [K,12] for versions 1 and 2.
/// @param version Architecture and objective version, 1, 2 or 3.
/// @return Normalized query and key tensors, enqueueable keys and loss.
/// @throws std::invalid_argument For aliased stores or incompatible arguments.
/// @par Example
/// @code
/// dm::nn::ParameterStore online(7), target(8), predictor(9);
/// auto out = dm::nn::moco(online, target, predictor, dm::nn::ones({4,36}), {}, 3);
/// @endcode
/// @par References
/// - He et al. (2020), https://arxiv.org/abs/1911.05722
/// - Chen et al. (2020), https://arxiv.org/abs/2003.04297
/// - Chen et al. (2021), https://arxiv.org/abs/2104.02057
/// @see moco_loss(), moco_enqueue(), ParameterStore::copy_from()
inline MoCoResult moco(ParameterStore &online, ParameterStore &target, ParameterStore &predictor,
                       const Tensor &views, const Tensor &queue, int version) {
  if (&online == &target || &online == &predictor || &target == &predictor ||
      !is_floating(views.dtype()) || views.dim() != 2 || views.shape()[0] < 2 ||
      views.shape()[0] % 2 || views.shape()[1] < 1 || version < 1 || version > 3)
    throw std::invalid_argument("moco: invalid stores, views or version");
  const auto project = [&](ParameterStore &store) {
    if (version >= 2)
      return contrastive_detail::siamese_projection(store, "moco", views);
    const auto encoded = dense(store, "moco/encoder", views, 32, Activation::Relu);
    return dense(store, "moco/output", encoded, 12);
  };
  auto q = project(online);
  Tensor key;
  {
    NoGradGuard guard;
    key = contrastive_detail::row_unit(project(target));
  }
  if (version == 3) {
    const auto hidden = dense(predictor, "moco/hidden", q, 24, Activation::Relu);
    q = dense(predictor, "moco/output", hidden, 12);
  }
  const auto query = contrastive_detail::row_unit(q);
  const auto forward_keys = slice_axis(key, 0, views.shape()[0] / 2, views.shape()[0]);
  return {query, key, forward_keys, moco_loss(query, key, queue, version)};
}

/// @brief Compute detached Sinkhorn assignments independently for both view groups.
///
/// Work in Float64 log space. Each iteration scales column mass toward
/// B/P and then normalizes every row to unit mass. Column balance is
/// approximate after a finite number of iterations.
///
/// @param scores Finite floating similarity matrix [2B,P] in paired view order.
/// @param epsilon Positive finite entropy scale, default 0.12.
/// @param iterations Positive number of column/row scaling sweeps, default 40.
/// @return Detached row probabilities [2B,P] in the score dtype.
/// @throws std::invalid_argument For invalid geometry, dtype or scaling options.
/// @par Example
/// @code
/// const auto assignments = dm::nn::swav_assignments(dm::nn::zeros({4, 3}));
/// @endcode
/// @par References
/// - Caron et al. (2020), https://arxiv.org/abs/2006.09882
/// @see swav()
inline Tensor swav_assignments(const Tensor &scores, double epsilon = .12, int iterations = 40) {
  if (scores.dim() != 2 || scores.shape()[0] < 2 || scores.shape()[0] % 2 ||
      scores.shape()[1] < 1 || !is_floating(scores.dtype()) ||
      !std::isfinite(epsilon) || epsilon <= 0 || iterations < 1)
    throw std::invalid_argument("swav_assignments: invalid scores or scaling options");
  NoGradGuard guard;
  const auto batch = scores.shape()[0] / 2, prototypes = scores.shape()[1];
  auto mass = scores.detach().to(DType::Float64).reshape({2, batch, prototypes});
  mass = (mass - max(mass, {1, 2}, true)) / epsilon;
  const auto column_mass = std::log(double(batch) / double(prototypes));
  for (int step = 0; step < iterations; ++step) {
    mass = mass - logsumexp(mass, {1}, true) + column_mass;
    mass = mass - logsumexp(mass, {2}, true);
  }
  return exp(mass).reshape(scores.shape()).to(scores.dtype());
}

/// @brief Own the compact SwAV training forward path and detached assignment targets.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto result = dm::nn::swav(model, "swav", dm::nn::ones({4,36}));
/// @endcode
/// @see swav()
struct SwAVResult {
  Tensor unit;        ///< Row-normalized projections [2B,12].
  Tensor prototypes;  ///< Column-normalized prototypes [12,P].
  Tensor scores;      ///< Cosine scores before scaling [2B,P].
  Tensor assignments; ///< Detached balanced assignments in original view order [2B,P].
  Tensor loss;        ///< Mean swapped cross-entropy over both view directions.
};

/// @brief Predict balanced prototype assignments across paired crops.
///
/// Projects both groups with the compact 32/24/12 MLP. Forms scores using
/// normalized prototype columns, computes independent Sinkhorn assignments
/// for the two groups, then swaps them before cross-entropy. Gradients flow
/// through scores but stop at the assignment computation.
///
/// @param model Owns encoder, projector and prototype parameters.
/// @param name Parameter prefix, reused across training calls.
/// @param views Finite floating paired view matrix [2B,D], with B and D positive.
/// @param count Positive prototype count, default 8.
/// @param temperature Positive finite prediction temperature, default 0.2.
/// @return Forward intermediates and differentiable scalar loss.
/// @throws std::invalid_argument For invalid geometry, options or incompatible parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto result = dm::nn::swav(model, "swav", dm::nn::ones({4,36}));
/// @endcode
/// @par References
/// - Caron et al. (2020), https://arxiv.org/abs/2006.09882
/// @see swav_assignments(), simclr_projection()
inline SwAVResult swav(ParameterStore &model, const std::string &name, const Tensor &views,
                       int count = 8, double temperature = .2) {
  if (views.dim() != 2 || views.shape()[0] < 2 || views.shape()[0] % 2 ||
      views.shape()[1] < 1 || !is_floating(views.dtype()) || count < 1 ||
      !std::isfinite(temperature) || temperature <= 0)
    throw std::invalid_argument("swav: invalid view geometry or options");
  const auto unit = contrastive_detail::row_unit(simclr_projection(model, name, views));
  const auto raw = model.get(name + "/prototype", {12, count}, init::uniform(.35));
  const auto prototypes = raw / sqrt(sum(square(raw), {0}, true) + 1e-8);
  const auto scores = matmul(unit, prototypes);
  const auto assignments = swav_assignments(scores);
  const auto loss = softmax_cross_entropy(scores / temperature,
                                           contrastive_detail::swap_views(assignments));
  return {unit, prototypes, scores, assignments, loss};
}

/// @brief Own causal contexts and in-batch future prediction scores.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto fit = dm::nn::contrastive_predictive_coding(model, "cpc", dm::nn::ones({3,6,4}));
/// @endcode
/// @see contrastive_predictive_coding()
struct ContrastivePredictiveCodingResult {
  Tensor latent;  ///< Per-step encoded features [B,T,D].
  Tensor context; ///< GRU states through the context prefix [B,C,D].
  Tensor scores;  ///< Time-major and then horizon-major logits [2C,B,B].
  Tensor loss;    ///< Scalar mean InfoNCE over contexts, horizons and examples.
};

/// @brief Build a causal GRU context to distinguish future latents across sequences.
///
/// A shared tanh projection encodes each time independently. The GRU reads
/// the prefix in temporal order. Two learned bilinear heads compare each
/// context with all examples one or two steps ahead. The matching sequence
/// is the positive and other sequences in the batch supply negatives.
///
/// @param store Owns all trainable parameters.
/// @param name Prefix for the encoder, GRU and horizon matrices.
/// @param input Finite floating sequence [B,T,F], with B >= 2 and F positive.
/// @param context Number of prefix steps, default 4. Requires T >= context + 2.
/// @param width Width of latents and context states, default 12.
/// @return Encoded path, causal contexts, logits and differentiable loss.
/// @throws std::invalid_argument If input geometry, options or parameters are incompatible.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto fit = dm::nn::contrastive_predictive_coding(model, "cpc", dm::nn::ones({3,6,4}));
/// @endcode
/// @par References
/// - van den Oord et al. (2018), https://arxiv.org/abs/1807.03748
/// @see gru_step(), nt_xent()
inline ContrastivePredictiveCodingResult contrastive_predictive_coding(
    ParameterStore &store, const std::string &name, const Tensor &input,
    int context = 4, int width = 12) {
  if (input.dim() != 3 || !is_floating(input.dtype()) || input.shape()[0] < 2 ||
      input.shape()[1] - 2 < context || input.shape()[2] < 1 || context < 1 || width < 1)
    throw std::invalid_argument("contrastive_predictive_coding: invalid sequence geometry");
  const auto batch = input.shape()[0];
  const auto latent = dense(store, name + "/encoder", input, width, Activation::Tanh);
  auto h = zeros({batch, width}, latent.dtype());
  std::vector<Tensor> states, scores;
  for (int t = 0; t < context; ++t) {
    h = gru_step(store, name + "/context", select(latent, 1, t), h);
    states.push_back(h);
    for (int horizon = 1; horizon <= 2; ++horizon) {
      const auto w = store.get(name + "/horizon" + std::to_string(horizon),
                               {width, width}, init::uniform(.25));
      const auto future = select(latent, 1, t + horizon);
      scores.push_back(matmul(matmul(h, w), future.transpose()));
    }
  }
  const auto logits = stack(scores, 0);
  const auto labels = broadcast_to(arange(0, batch, 1, DType::Int64),
                                   {2 * std::int64_t(context), batch});
  return {latent, stack(states, 1), logits, sparse_softmax_cross_entropy(logits, labels)};
}

/// @brief Own pair distances and squared contrastive margin losses.
/// @par Example
/// @code
/// const auto fit = dm::nn::contrastive_margin(dm::nn::zeros({4,2,3}), dm::nn::ones({4}));
/// @endcode
/// @see contrastive_margin()
struct ContrastiveMarginResult {
  Tensor distances; ///< Stabilized Euclidean distances [N].
  Tensor per_pair;  ///< Squared margin objective for each pair [N].
  Tensor loss;      ///< Scalar mean of the pair losses.
};

/// @brief Pull matching embeddings together and penalize negatives inside a margin.
///
/// Matching pairs contribute squared Euclidean distance. Nonmatching pairs
/// contribute the squared positive part of margin minus stabilized distance.
/// A 1e-8 offset inside the square root gives finite derivatives at zero.
///
/// @param embedding Floating [N,2,D] paired embeddings, N and D positive.
/// @param same Boolean or numeric [N] indicators, one for a match and zero otherwise.
/// @param margin Positive finite distance margin, default one.
/// @return Differentiable distances and pair losses, with labels treated as constants.
/// @throws std::invalid_argument For invalid geometry, labels or margin.
/// @par Example
/// @code
/// const auto fit = dm::nn::contrastive_margin(dm::nn::zeros({4,2,3}), dm::nn::ones({4}));
/// @endcode
/// @par References
/// - Hadsell et al. (2006), https://doi.org/10.1109/CVPR.2006.100
/// @see nt_xent()
inline ContrastiveMarginResult contrastive_margin(const Tensor &embedding, const Tensor &same,
                                                   double margin = 1.) {
  if (embedding.dim() != 3 || embedding.shape()[1] != 2 || embedding.shape()[0] < 1 ||
      embedding.shape()[2] < 1 || !is_floating(embedding.dtype()) ||
      same.shape() != Shape{embedding.shape()[0]} || !std::isfinite(margin) || margin <= 0)
    throw std::invalid_argument("contrastive_margin: invalid pair geometry or margin");
  if (!all(logical_or(equal(same.detach(), 0.), equal(same.detach(), 1.))).item<bool>())
    throw std::invalid_argument("contrastive_margin: expected binary labels");
  const auto squared = sum(square(select(embedding, 1, 0) - select(embedding, 1, 1)), {1});
  const auto distances = sqrt(squared + 1e-8);
  const auto per_pair = where(equal(same.detach(), 1.), squared,
                              square(relu(margin - distances)));
  return {distances, per_pair, mean(per_pair)};
}

/// @brief Own one contribution per matched pair and the scalar training objective.
/// @par Example
/// @code
/// const auto fit = dm::nn::npair_loss(dm::nn::ones({4,3}));
/// @endcode
/// @see npair_loss(), lifted_structured_loss()
struct StructuredMetricResult {
  Tensor terms; ///< Per-pair objective before any shared regularization [B].
  Tensor loss;  ///< Mean per-pair objective plus regularization if applicable.
};

/// @brief Compare each anchor against one positive per distinct class.
///
/// Forms cross-entropy on anchor-to-positive dot products with matching
/// columns on the diagonal, then adds half the regularization coefficient
/// times the mean squared norm of all embedding rows.
///
/// @param paired Floating [2B,D], B >= 2 distinct classes in each half in matching order.
/// @param regularization Finite nonnegative L2 coefficient, default 0.002.
/// @return Differentiable pair terms and total loss.
/// @throws std::invalid_argument For invalid geometry or regularization.
/// @par Example
/// @code
/// const auto fit = dm::nn::npair_loss(dm::nn::ones({4,3}));
/// @endcode
/// @par References
/// - Sohn (2016), Improved Deep Metric Learning with Multi-class N-pair Loss Objective.
/// @see lifted_structured_loss()
inline StructuredMetricResult npair_loss(const Tensor &paired, double regularization = .002) {
  if (paired.dim() != 2 || paired.shape()[0] < 4 || paired.shape()[0] % 2 ||
      paired.shape()[1] < 1 || !is_floating(paired.dtype()) ||
      !std::isfinite(regularization) || regularization < 0.)
    throw std::invalid_argument("npair_loss: invalid pair geometry or regularization");
  const auto batch = paired.shape()[0] / 2;
  const auto anchor = slice_axis(paired, 0, 0, batch);
  const auto positive = slice_axis(paired, 0, batch, 2 * batch);
  const auto terms = sparse_softmax_cross_entropy(matmul(anchor, positive.transpose()),
      arange(0, batch, 1., DType::Int64), Reduction::None);
  return {terms, mean(terms) + (.5 * regularization) * mean(sum(square(paired), {1}))};
}

/// @brief Apply the lifted structured hinge to all negatives around both ends of each pair.
///
/// Distances use a 1e-8 squared-distance offset. For each pair, sum the
/// exponentiated margin-minus-distance scores of its two endpoints against
/// all different-class rows. Add the log of that mass (floored at 1e-30)
/// to the positive distance and apply half the squared positive part.
///
/// @param paired Floating [2B,D], B >= 2 distinct classes in each half in matching order.
/// @param margin Positive finite distance margin, default one.
/// @return Differentiable pair terms and their mean.
/// @throws std::invalid_argument For invalid geometry or margin.
/// @par Example
/// @code
/// const auto fit = dm::nn::lifted_structured_loss(dm::nn::ones({4,3}));
/// @endcode
/// @par References
/// - Song et al. (2016), https://arxiv.org/abs/1511.06452
/// @see npair_loss(), contrastive_margin()
inline StructuredMetricResult lifted_structured_loss(const Tensor &paired, double margin = 1.) {
  if (paired.dim() != 2 || paired.shape()[0] < 4 || paired.shape()[0] % 2 ||
      paired.shape()[1] < 1 || !is_floating(paired.dtype()) ||
      !std::isfinite(margin) || margin <= 0.)
    throw std::invalid_argument("lifted_structured_loss: invalid pair geometry or margin");
  const auto rows = paired.shape()[0], batch = rows / 2, width = paired.shape()[1];
  const auto classes = arange(0, batch, 1., DType::Int64);
  const auto ids = concat({classes, classes}, 0);
  const auto delta = paired.reshape({rows, 1, width}) - paired.reshape({1, rows, width});
  const auto distance = sqrt(sum(square(delta), {2}) + 1e-8);
  const auto eligible = not_equal(ids.reshape({rows, 1}), ids.reshape({1, rows}));
  const auto negatives = where(eligible, margin - distance,
                                scalar(-std::numeric_limits<double>::infinity(), paired.dtype()));
  const auto grouped = concat({slice_axis(negatives, 0, 0, batch),
                                slice_axis(negatives, 0, batch, rows)}, 1);
  const auto log_mass = maximum(logsumexp(grouped, {1}), scalar(std::log(1e-30), paired.dtype()));
  const auto difference = slice_axis(paired, 0, 0, batch) - slice_axis(paired, 0, batch, rows);
  const auto positive = sqrt(sum(square(difference), {1}) + 1e-8);
  const auto terms = .5 * square(relu(positive + log_mass));
  return {terms, mean(terms)};
}

/// @brief Own cosine scores and a differentiable proxy metric objective.
/// @par Example
/// @code
/// const auto fit = dm::nn::proxy_metric_loss(dm::nn::ones({2,3}), dm::nn::ones({2,3}),
///     dm::nn::arange(0,2,1,dm::nn::DType::Int64));
/// @endcode
/// @see proxy_metric_loss()
struct ProxyMetricResult {
  Tensor similarity; ///< Cosine scores between samples and proxies [N,K].
  Tensor loss;       ///< Scalar objective with gradients to both input tables.
};

/// @brief Compute Proxy-NCA or Proxy Anchor over normalized embeddings and proxies.
///
/// Row norms use a squared-norm offset of 1e-8. Proxy-NCA compares the
/// positive squared distance with negative proxies only in the denominator.
/// Proxy Anchor averages the positive penalty over present classes and the
/// negative penalty over all classes, with cosine margin 0.1 and scale 32.
///
/// @param embedding Floating samples [N,D], N and D positive.
/// @param proxies Floating class representatives [K,D], K >= 2, same dtype.
/// @param labels Integral target IDs [N] in [0,K).
/// @param anchor Use Proxy Anchor when true, Proxy-NCA otherwise (default).
/// @return Cosine scores and differentiable scalar loss.
/// @throws std::invalid_argument For incompatible geometry, labels or dtype.
/// @par Example
/// @code
/// const auto fit = dm::nn::proxy_metric_loss(dm::nn::ones({2,3}), dm::nn::ones({2,3}),
///     dm::nn::arange(0,2,1,dm::nn::DType::Int64));
/// @endcode
/// @par References
/// - Movshovitz-Attias et al. (2017), https://arxiv.org/abs/1703.07464
/// - Kim et al. (2020), https://arxiv.org/abs/2003.13911
/// @see npair_loss(), lifted_structured_loss()
inline ProxyMetricResult proxy_metric_loss(const Tensor &embedding, const Tensor &proxies,
                                           const Tensor &labels, bool anchor = false) {
  if (embedding.dim() != 2 || proxies.dim() != 2 || embedding.shape()[0] < 1 ||
      embedding.shape()[1] < 1 || proxies.shape()[0] < 2 ||
      proxies.shape()[1] != embedding.shape()[1] || !is_floating(embedding.dtype()) ||
      proxies.dtype() != embedding.dtype() || labels.shape() != Shape{embedding.shape()[0]} ||
      !is_integral(labels.dtype()) || min(labels).item<std::int64_t>() < 0 ||
      max(labels).item<std::int64_t>() >= proxies.shape()[0])
    throw std::invalid_argument("proxy_metric_loss: invalid sample, proxy or label geometry");
  const auto unit = [](const Tensor &x) { return x / sqrt(sum(square(x), {1}, true) + 1e-8); };
  const auto x = unit(embedding), p = unit(proxies), similarity = matmul(x, p.transpose());
  const auto n = x.shape()[0], k = p.shape()[0], d = x.shape()[1];
  const auto match = one_hot(labels, k, 1., 0., embedding.dtype());
  const auto target = equal(match, 1.);
  const auto exclude = scalar(-std::numeric_limits<double>::infinity(), embedding.dtype());
  if (anchor) {
    const auto base = zeros({1, k}, embedding.dtype());
    const auto positive = logsumexp(concat({base, where(target, -32. * (similarity - .1), exclude)}, 0), {0});
    const auto negative = logsumexp(concat({base, where(target, exclude, 32. * (similarity + .1))}, 0), {0});
    const auto present = count_nonzero(sum(match, {0})).item();
    return {similarity, sum(positive) / present + mean(negative)};
  }
  const auto delta = x.reshape({n, 1, d}) - p.reshape({1, k, d});
  const auto distance = sum(square(delta), {2});
  const auto positive = sum(match * distance, {1});
  return {similarity, mean(positive + logsumexp(where(target, exclude, -distance), {1}))};
}

}  // namespace datamunge::nn
