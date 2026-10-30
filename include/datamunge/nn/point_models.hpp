#pragma once
/// @file point_models.hpp
/// @brief Complete PointNet with bothT-Nets and PointNet++SSG with tensor geometric grouping.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/point_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Configure complete PointNet or PointNet++SSG geometry and widths.
///
/// PointNet ignores hierarchical center/neighbor fields. Width scaling preserves
/// both T-Nets/shared MLPs or all3set abstractions and the two-layer classifier.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PointConfig c{4, 3, 256, 64, 16, 16, 32};
/// @endcode
/// @par References
/// - Qi,Su,Mo and Guibas(2017),PointNet:Deep Learning on Point Sets for3DClassification and Segmentation, https://arxiv.org/abs/1612.00593;Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching units use Glorot dense/bias/populationBN eps.001/decay.9;compact width/4/256point/threeclass fits and seed0/strict-radius input-order grouping are disclosed adaptations,not full benchmark reproduction.
/// @see PointNet(), PointNetPlusPlus()
struct PointConfig {
  int divisor = 1;           ///< Width divisor1..16;compact4,sourcefull1.
  int classes = 40;          ///< At least2head classes;compact3.
  int points = 1024;         ///< Exact source point count;compact256.
  int first_centers = 512;   ///< FirstFPS centers<=points;compact64.
  int second_centers = 128;  ///< SecondFPS centers<=first;compact16.
  int first_neighbors = 32;  ///< Positive first-radius output slots;compact16.
  int second_neighbors = 64; ///< Positive second-radius output slots;compact32.
};

/// @brief Own logits,global features and model-specific transform/grouping endpoints.
///
/// All tensor handles own graph/storage. PointNet populates two transforms;
/// PointNet++populates centers/neighbor identities/counts instead. Unused fields
/// remain undefined;neighborhood choices are detached categorical metadata.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(971), s(977);
/// const auto out = m::PointNet(p, s, n::ones({1, 256, 3}), {4, 3, 256, 64, 16, 16, 32});
/// @endcode
/// @par References
/// - Qi,Su,Mo and Guibas(2017),PointNet:Deep Learning on Point Sets for3DClassification and Segmentation, https://arxiv.org/abs/1612.00593;Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching units use Glorot dense/bias/populationBN eps.001/decay.9;compact width/4/256point/threeclass fits and seed0/strict-radius input-order grouping are disclosed adaptations,not full benchmark reproduction.
/// @see PointNet(), PointNetPlusPlus()
struct PointOutput {
  Tensor logits;            ///< Raw[N,classes] scores.
  Tensor global;            ///< Point/global-abstraction max-pooled[N,1024/divisor]features.
  Tensor input_transform;   ///< PointNet[B,3,3]learned transform;empty for++.
  Tensor feature_transform; ///< PointNet[B,64/divisor,64/divisor]transform;empty for++.
  Tensor centroids1;        ///< PointNet++firstFPS[B,S1,3];empty forPointNet.
  Tensor centroids2;        ///< PointNet++secondFPS[B,S2,3];empty forPointNet.
  Tensor neighbors1;        ///< PointNet++detached[B,S1,K1]localsource IDs.
  Tensor neighbors2;        ///< PointNet++detached[B,S2,K2]first-center IDs.
  Tensor counts1;           ///< PointNet++first capped strict-radius counts.
  Tensor counts2;           ///< PointNet++second capped strict-radius counts.
};

/// @brief Bind non-owning stores/modes for source shared point MLP,BN and T-Net units.
///
/// Caller stores and optional generator must outlive this lightweight adapter.
/// Calls lazily create leaves/update explicitly requested moving moments,but no
/// optimizer or fit loop is hidden. Output tensor handles own storage/graph.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(971), s(977);
/// m::PointLayers layer(p, s, false, false, 4, nullptr);
/// const auto y = layer.Unit("u", n::ones({1, 4, 3}), 8);
/// @endcode
/// @par References
/// - Qi,Su,Mo and Guibas(2017),PointNet:Deep Learning on Point Sets for3DClassification and Segmentation, https://arxiv.org/abs/1612.00593;Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching units use Glorot dense/bias/populationBN eps.001/decay.9;compact width/4/256point/threeclass fits and seed0/strict-radius input-order grouping are disclosed adaptations,not full benchmark reproduction.
/// @see PointNet(), PointNetPlusPlus()
class PointLayers {
public:
  /// @brief Bind explicit learned/state ownership and training/update policy.
  ///
  /// References do not transfer ownership;inference always freezes state.
  ///
  /// @param parameters Mutable matching floating learned store.
  /// @param stats Mutable moving-population store,same dtype.
  /// @param train True uses batch moments and classifier dropout.
  /// @param update True permits moving updates only when train is also true.
  /// @param div Width divisor1..16.
  /// @param dropout_gen Required mutable generator when train=true;otherwise unused.
  /// @throws std::invalid_argument For invalid divisor,store dtype mismatch or missing training generator.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(971), s(977);
/// m::PointLayers l(p, s, false, false, 4, nullptr);
/// @endcode
  /// @par References
  /// - Qi,Su,Mo and Guibas(2017),PointNet:Deep Learning on Point Sets for3DClassification and Segmentation, https://arxiv.org/abs/1612.00593;Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching units use Glorot dense/bias/populationBN eps.001/decay.9;compact width/4/256point/threeclass fits and seed0/strict-radius input-order grouping are disclosed adaptations,not full benchmark reproduction.
  /// @see PointNet(), PointNetPlusPlus()
  PointLayers(nn::ParameterStore &parameters, nn::ParameterStore &stats, bool train,
              bool update, int div, nn::Generator *dropout_gen)
      : p(parameters), state(stats), training(train), update_stats(train && update),
        divisor(div), dropout(dropout_gen) {
    if (p.dtype() != state.dtype())
      throw std::invalid_argument("matching point store dtypes required");
    if (divisor < 1 || divisor > 16)
      throw std::invalid_argument("point model width divisor");
    if (training && !dropout)
      throw std::invalid_argument("point model training needs a dropout generator");
  }
  /// @brief Return a validated source width divided by this adapter divisor.
  ///
  /// @param full Positive original channel count.
  /// @return max(1,full/divisor).
  /// @throws std::invalid_argument For nonpositive original width.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p, s;
/// m::PointLayers l(p, s, false, false, 4, nullptr);
/// const auto width = l.Width(64); // 16
/// @endcode
  /// @par References
  /// - Qi,Su,Mo and Guibas(2017),PointNet:Deep Learning on Point Sets for3DClassification and Segmentation, https://arxiv.org/abs/1612.00593;Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching units use Glorot dense/bias/populationBN eps.001/decay.9;compact width/4/256point/threeclass fits and seed0/strict-radius input-order grouping are disclosed adaptations,not full benchmark reproduction.
  /// @see PointNet(), PointNetPlusPlus()
  [[nodiscard]] int Width(int full) const {
    if (full < 1)
      throw std::invalid_argument("positive original point width required");
    return std::max(1, full / divisor);
  }

  /// @brief Apply shared Glorot dense,bias,population BN and ReLU over the last axis.
  ///
  /// Moments reduce every axis except channels. Population variance in both
  /// training normalization and EMA,.9decay/eps.001,gamma1/beta0,mean0/variance1.
  /// Pre-BN biases and exact identity T-Net upstream paths may have zero adjoints;
  /// zero is not a disconnected graph. Updates require explicit training&&update.
  ///
  /// @param name Prefix with/W,/b,/bn/gamma,/bn/beta and moving/bn/mean,/bn/variance.
  /// @param x Floating rank>=2,matching store dtype,positive last-axis width.
  /// @param out Positive output channels.
  /// @return Same leading axes with last widthout;input/alllearned graphs retained.
  /// @throws std::invalid_argument For invalid shape,dtype,width or stored leaves.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p, s;
/// m::PointLayers l(p, s, false, false, 4, nullptr);
/// const auto y = l.Unit("u", n::ones({2, 5, 3}), 8);
/// @endcode
  /// @par References
  /// - Qi,Su,Mo and Guibas(2017),PointNet:Deep Learning on Point Sets for3DClassification and Segmentation, https://arxiv.org/abs/1612.00593;Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching units use Glorot dense/bias/populationBN eps.001/decay.9;compact width/4/256point/threeclass fits and seed0/strict-radius input-order grouping are disclosed adaptations,not full benchmark reproduction.
  /// @see PointNet(), PointNetPlusPlus()
  Tensor Unit(const std::string &name, const Tensor &x, int out) {
    if (x.dim() < 2 || x.size(-1) < 1 || out < 1 || x.dtype() != p.dtype())
      throw std::invalid_argument("valid shared point unit shape/dtype/width required");
    const Tensor y = nn::dense(p, name, x, out);
    const Tensor gamma = p.get(name + "/bn/gamma", {out}, nn::init::ones());
    const Tensor beta = p.get(name + "/bn/beta", {out}, nn::init::zeros());
    const Tensor mean = state.get(name + "/bn/mean", {out}, nn::init::zeros());
    const Tensor variance = state.get(name + "/bn/variance", {out}, nn::init::ones());
    Tensor mu = mean.detach(), var = variance.detach();
    if (training) {
      nn::Axes axes;
      for (std::int64_t i = 0; i + 1 < y.dim(); ++i)
        axes.push_back(i);
      mu = nn::mean(y, axes);
      const Tensor centered = y - mu;
      var = nn::mean(centered * centered, axes);
      if (update_stats) {
        nn::NoGradGuard guard;
        state.set(name + "/bn/mean", mean.detach() * .9 + mu.detach() * .1);
        state.set(name + "/bn/variance", variance.detach() * .9 + var.detach() * .1);
      }
    }
    return nn::relu((y - mu) * (gamma * nn::rsqrt(var + .001)) + beta);
  }

  /// @brief Predict an exactly identity-initialized learned T-Net matrix.
  ///
  /// Shared widths64/128/1024 then max points,FC512/256 scaled bydivisor.
  /// Zero finalW/b plus identity means upstream T-Net leaves initially have
  /// zero gradients until the final transform head moves. All leaves remain
  /// connected. Does not itself apply the predicted matrix or fit parameters.
  ///
  /// @param name Prefix ofshared MLP/FC/head leaves.
  /// @param features Floating[B,N,C] matching store dtype.
  /// @param dimension Positive target matrix side<=1024.
  /// @return Floating[B,dimension,dimension],initially exactlyidentity.
  /// @throws std::invalid_argument For invalid feature geometry,matrix dimension or stored leaves.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p, s;
/// m::PointLayers l(p, s, false, false, 4, nullptr);
/// const auto transform = l.Transform("t", n::ones({1, 5, 3}), 3);
/// @endcode
  /// @par References
  /// - Qi,Su,Mo and Guibas(2017),PointNet:Deep Learning on Point Sets for3DClassification and Segmentation, https://arxiv.org/abs/1612.00593;Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching units use Glorot dense/bias/populationBN eps.001/decay.9;compact width/4/256point/threeclass fits and seed0/strict-radius input-order grouping are disclosed adaptations,not full benchmark reproduction.
  /// @see PointNet(), PointNetPlusPlus()
  Tensor Transform(const std::string &name, const Tensor &features, int dimension) {
    if (features.dim() != 3 || features.size(0) < 1 || features.size(1) < 1 || dimension < 1 ||
        dimension > 1024)
      throw std::invalid_argument(
          "positive T-Net feature shape and matrixside<=1024 required");
    Tensor x = Unit(name + "/mlp1", features, Width(64));
    x = Unit(name + "/mlp2", x, Width(128));
    x = Unit(name + "/mlp3", x, Width(1024));
    x = nn::max(x, {1});
    x = Unit(name + "/fc1", x, Width(512));
    x = Unit(name + "/fc2", x, Width(256));
    const Tensor w =
        p.get(name + "/head/W", {Width(256), dimension * dimension}, nn::init::zeros());
    const Tensor b = p.get(name + "/head/b", {dimension * dimension}, nn::init::zeros());
    const Tensor identity =
        nn::reshape(nn::eye(dimension, p.dtype()), {dimension * dimension});
    return nn::reshape(nn::bias_add(nn::matmul(x, w), b) + identity,
                       {-1, dimension, dimension});
  }

  /// @brief Apply two shared-BN dense hidden layers,explicit dropout and a linear classifier.
  ///
  /// Scaled hidden widths512/256,dropout after BOTH ReLUs during training.
  /// Caller generator ownership and state-mutation policy come from constructor.
  /// No hidden optimizer;head is ordinary Glorotdense/bias.
  ///
  /// @param global Floating[B,C] global features.
  /// @param classes Head classes>=2.
  /// @param keep Finite keep probability(0,1],sourcePointNet.7,PointNet++.5.
  /// @return Floating[B,classes]raw scores.
  /// @throws std::invalid_argument For invalid global geometry,classes,keep or stored leaves.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p, s;
/// m::PointLayers l(p, s, false, false, 4, nullptr);
/// const auto z = l.Classifier(n::ones({2, 8}), 3, .7);
/// @endcode
  /// @par References
  /// - Qi,Su,Mo and Guibas(2017),PointNet:Deep Learning on Point Sets for3DClassification and Segmentation, https://arxiv.org/abs/1612.00593;Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching units use Glorot dense/bias/populationBN eps.001/decay.9;compact width/4/256point/threeclass fits and seed0/strict-radius input-order grouping are disclosed adaptations,not full benchmark reproduction.
  /// @see PointNet(), PointNetPlusPlus()
  Tensor Classifier(const Tensor &global, int classes, double keep) {
    if (global.dim() != 2 || classes < 2 || !std::isfinite(keep) || keep <= 0 || keep > 1)
      throw std::invalid_argument("valid point classifier shape/classes/keep required");
    Tensor x = Unit("classifier/fc1", global, Width(512));
    if (training)
      x = nn::dropout(x, 1 - keep, true, *dropout);
    x = Unit("classifier/fc2", x, Width(256));
    if (training)
      x = nn::dropout(x, 1 - keep, true, *dropout);
    return nn::dense(p, "classifier/head", x, classes);
  }

private:
  nn::ParameterStore &p;     ///< Non-owning learnedstore reference.
  nn::ParameterStore &state; ///< Non-owning moving-population reference.
  bool training;             ///< Batch-moment/dropout mode.
  bool update_stats;         ///< Trueonlyduring requested trainingupdates.
  int divisor;               ///< Validated1..16widthscale.
  nn::Generator *dropout;    ///< Non-owning mutabletrainingRNG,nullableatinference.
};

/// @brief Classify an unordered cloud with both learned T-Nets and every source shared MLP.
///
/// Input3x3T-Net,shared64/64,feature64x64T-Net,shared64/128/1024,
/// max overpoints,512/256classifier withkeep.7dropout. Allwidthsdivided
/// bycfg.divisor;both transform heads start EXACTidentity. Inference logits
/// are permutation invariant up to numerical tolerance. Add the explicit
/// feature-transform penalty to the caller CE objective during fitting.
///
/// @param p Mutable learned store;warm every leaf before optimizer construction.
/// @param state Mutable matching moving-population store;checkpoint with p.
/// @param xyz Floating[B,points,3],positive batch and configured pointcount.
/// @param cfg Full/compact widths/geometry/classes.
/// @param training False inference;true batch moments and two classifier dropouts.
/// @param update_stats False bydefault;only training&&true updates moving moments.
/// @param dropout_gen Required mutable generator for training,unused for inference.
/// @return Raw logits,global max features and both learned transform matrices.
/// @throws std::invalid_argument For invalid configured geometry/classes,dtype,trainingRNG or stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(971), s(977);
/// const auto out = m::PointNet(p, s, n::ones({1, 256, 3}), {4, 3, 256, 64, 16, 16, 32});
/// @endcode
/// @par References
/// - Qi,Su,Mo and Guibas(2017),PointNet:Deep Learning on Point Sets for3DClassification and Segmentation, https://arxiv.org/abs/1612.00593;Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching units use Glorot dense/bias/populationBN eps.001/decay.9;compact width/4/256point/threeclass fits and seed0/strict-radius input-order grouping are disclosed adaptations,not full benchmark reproduction.
/// @see PointNet(), PointNetPlusPlus()
inline PointOutput PointNet(nn::ParameterStore &p, nn::ParameterStore &state,
                            const Tensor &xyz, const PointConfig &cfg = {},
                            bool training = false, bool update_stats = false,
                            nn::Generator *dropout_gen = nullptr) {
  if (xyz.dim() != 3 || xyz.size(0) < 1 ||
      xyz.shape() != nn::Shape{xyz.size(0), cfg.points, 3} || xyz.dtype() != p.dtype())
    throw std::invalid_argument("matching positive[B,points,3] point input required");
  if (cfg.classes < 2)
    throw std::invalid_argument("PointNet classes");
  PointOutput out;
  PointLayers layer(p, state, training, training && update_stats, cfg.divisor, dropout_gen);
  out.input_transform = layer.Transform("input_tnet", xyz, 3);
  Tensor x = nn::matmul(xyz, out.input_transform);
  x = layer.Unit("point_mlp1", x, layer.Width(64));
  x = layer.Unit("point_mlp2", x, layer.Width(64));
  out.feature_transform = layer.Transform("feature_tnet", x, layer.Width(64));
  x = nn::matmul(x, out.feature_transform);
  x = layer.Unit("point_mlp3", x, layer.Width(64));
  x = layer.Unit("point_mlp4", x, layer.Width(128));
  x = layer.Unit("point_mlp5", x, layer.Width(1024));
  out.global = nn::max(x, {1});
  out.logits = layer.Classifier(out.global, cfg.classes, .7);
  return out;
}

/// @brief Penalize source feature-transform nonorthogonality with differentiable tensor algebra.
///
/// Matches source.001*L2Loss;caller adds to mean CE and optimizes together.
/// Orthogonal transforms have zero penalty. No optimizer or projection hidden.
///
/// @param transform Floating[B,d,d] feature transforms,Bpositive.
/// @param dimension Positive matrix sided.
/// @return Scalar.0005*sum((A*A_transpose-I)^2),SUM across batch,not mean.
/// @throws std::invalid_argument For invalid matrix/batch/dimension geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto penalty = m::PointTransformPenalty(n::expand_dims(n::eye(3), 0), 3); // 0
/// @endcode
/// @par References
/// - Qi,Su,Mo and Guibas(2017),PointNet:Deep Learning on Point Sets for3DClassification and Segmentation, https://arxiv.org/abs/1612.00593;Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching units use Glorot dense/bias/populationBN eps.001/decay.9;compact width/4/256point/threeclass fits and seed0/strict-radius input-order grouping are disclosed adaptations,not full benchmark reproduction.
/// @see PointNet(), PointNetPlusPlus()
inline Tensor PointTransformPenalty(const Tensor &transform, int dimension) {
  if (dimension < 1 || transform.dim() != 3 || transform.size(0) < 1 ||
      transform.shape() != nn::Shape{transform.size(0), dimension, dimension})
    throw std::invalid_argument("matching positive transform matrix geometry required");
  const Tensor error =
      nn::matmul(transform, transform, false, true) - nn::eye(dimension, transform.dtype());
  return nn::sum(error * error) * .0005;
}

/// @brief Classify point clouds with all three source single-scale set abstractions.
///
/// FirstFPS/radius.2 shared64/64/128/localmax;secondFPS/radius.4
/// concats relativeXYZ+gatheredfeatures then128/128/256/localmax. Global
/// concatcenterXYZ+features,256/512/1024/max,512/256classifier keep.5.
/// Widths scaled but alllayers retained. First-seedFPS/input-order ball queries
/// are ORDERDEPENDENT;do not promise PointNet permutation invariance.
/// Discretechoices frozen,coordinate/featuregathers remain differentiable.
///
/// @param p Mutable learned store;warm every leaf before optimizer construction.
/// @param state Mutable matching moving-population store;checkpoint with p.
/// @param xyz Floating[B,points,3],positive batch and configured pointcount.
/// @param cfg Full/compact widths/geometry/classes.
/// @param training False inference;true batch moments and two classifier dropouts.
/// @param update_stats False bydefault;only training&&true updates moving moments.
/// @param dropout_gen Required mutable generator for training,unused for inference.
/// @return Raw logits,globalfeatures,bothcenter levels and detached neighbor/count metadata.
/// @throws std::invalid_argument For invalid input/architecture/grouping geometry,dtype,RNG or stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(971), s(977);
/// const auto out = m::PointNetPlusPlus(p, s, n::ones({1, 256, 3}), {4, 3, 256, 64, 16, 16, 32});
/// @endcode
/// @par References
/// - Qi,Su,Mo and Guibas(2017),PointNet:Deep Learning on Point Sets for3DClassification and Segmentation, https://arxiv.org/abs/1612.00593;Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching units use Glorot dense/bias/populationBN eps.001/decay.9;compact width/4/256point/threeclass fits and seed0/strict-radius input-order grouping are disclosed adaptations,not full benchmark reproduction.
/// @see PointNet(), PointNetPlusPlus()
inline PointOutput PointNetPlusPlus(nn::ParameterStore &p, nn::ParameterStore &state,
                                    const Tensor &xyz, const PointConfig &cfg = {},
                                    bool training = false, bool update_stats = false,
                                    nn::Generator *dropout_gen = nullptr) {
  if (xyz.dim() != 3 || xyz.size(0) < 1 ||
      xyz.shape() != nn::Shape{xyz.size(0), cfg.points, 3} || xyz.dtype() != p.dtype() ||
      cfg.first_centers < 1 || cfg.second_centers < 1 || cfg.first_neighbors < 1 ||
      cfg.second_neighbors < 1)
    throw std::invalid_argument(
        "positive PointNet++ configured input/group geometry required");
  if (cfg.classes < 2 || cfg.points < cfg.first_centers ||
      cfg.first_centers < cfg.second_centers)
    throw std::invalid_argument("PointNet++ dimensions");
  PointOutput out;
  PointLayers layer(p, state, training, training && update_stats, cfg.divisor, dropout_gen);
  const auto first = PointGroup(xyz, cfg.first_centers, .2f, cfg.first_neighbors);
  out.centroids1 = first.centers;
  out.neighbors1 = first.indices;
  out.counts1 = first.counts;
  Tensor x = layer.Unit("sa1/mlp1", first.relative, layer.Width(64));
  x = layer.Unit("sa1/mlp2", x, layer.Width(64));
  x = layer.Unit("sa1/mlp3", x, layer.Width(128));
  x = nn::max(x, {2});
  const auto second = PointGroup(first.centers, cfg.second_centers, .4f, cfg.second_neighbors);
  out.centroids2 = second.centers;
  out.neighbors2 = second.indices;
  out.counts2 = second.counts;
  x = nn::concat({second.relative, PointGather(x, second.indices)}, 3);
  x = layer.Unit("sa2/mlp1", x, layer.Width(128));
  x = layer.Unit("sa2/mlp2", x, layer.Width(128));
  x = layer.Unit("sa2/mlp3", x, layer.Width(256));
  x = nn::max(x, {2});
  x = nn::expand_dims(nn::concat({second.centers, x}, 2), 1);
  x = layer.Unit("sa3/mlp1", x, layer.Width(256));
  x = layer.Unit("sa3/mlp2", x, layer.Width(512));
  x = layer.Unit("sa3/mlp3", x, layer.Width(1024));
  out.global = nn::reshape(nn::max(x, {2}), {-1, layer.Width(1024)});
  out.logits = layer.Classifier(out.global, cfg.classes, .5);
  return out;
}

} // namespace datamunge::nn::models
