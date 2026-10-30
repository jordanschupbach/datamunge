#pragma once
/// @file sensor_models.hpp
/// @brief Strided residual multichannel classifiers and explicit fitting-weight CE.
#include "datamunge/nn/time_series_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Configure raw channel count,class inventory and residual temporal width.
///
/// Metadata only. ECG200 uses1channel/2classes,BasicMotions6channels/4classes.
/// Full width64 and compact8 retain the same strided residual structure.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SensorConfig c{6, 4, 8};
/// @endcode
/// @par References
/// - He et al.(2016),Deep Residual Learning for Image Recognition, https://arxiv.org/abs/1512.03385. Source teaching1D variant uses three strided5/3 convolution stages with learned1x1 skip,He-uniform initialization,per-channel fitting normalization and global temporal pooling,no BN or claim of exact paper architecture.
/// @see SensorCnn()
struct SensorConfig {
  int channels = 1; ///< Positive number of observed raw signal channels.
  int classes = 2;  ///< Number of source classes>=2.
  int base = 64;    ///< Positive first residual width,doubled each stage;compact8.
};
/// @brief Own raw sensor logits,normalized inputs and every strided residual activation.
///
/// Tensor handles retain learned/input graphs;normalization state is detached.
/// No labels or class weights are needed to produce logits.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(587), state(593);
/// const auto out = m::SensorCnn(p, state, n::ones({2, 100, 6}), m::SensorConfig{6, 4, 8});
/// @endcode
/// @par References
/// - He et al.(2016),Deep Residual Learning for Image Recognition, https://arxiv.org/abs/1512.03385. Source teaching1D variant uses three strided5/3 convolution stages with learned1x1 skip,He-uniform initialization,per-channel fitting normalization and global temporal pooling,no BN or claim of exact paper architecture.
/// @see SensorCnn()
struct SensorOutput {
  Tensor logits;     ///< Raw unbounded[N,classes] head scores.
  Tensor normalized; ///< Raw[N,T,channels] centered/scaled by saved fitting channel states.
  Tensor features;   ///< Final time-averaged[N,4base] residual features.
  std::vector<Tensor> stages; ///< Three residual/ReLU outputs with successive ceil(time/2).
};

/// @brief Predict raw multichannel sensor classes with three strided residual temporal stages.
///
/// Creation defaults mean0/std1;caller explicitly replaces them with fitting-only
/// population statistics before optimization,or restores a paired checkpoint.
/// Detach normalization;each block applies5x1 stride2/ReLU then3x1 stride1,
/// adds a learned1x1 stride2 skip and applies ReLU. TensorFlow SAME determines
/// odd/even stride phase;time becomes ceil(T/2) each block. Widths base/2base/4base,
/// He-uniform convolutions,biases0. Global time mean feeds a linear class head.
/// No BN,dropout,target,class weights or optimizer enter forward. Repeated
/// warmed inference never mutates learned values or saved channel statistics.
///
/// @param p Mutable learned store;warm all20 leaves before optimizer construction.
/// @param state Same-dtype separate store;two arrays of length channels hold fitting mean/std.
/// @param signal Floating[N,T,channels],N/T>0,matching both stores dtype.
/// @param config Positive channels/base and classes>=2;default1/2/64.
/// @return Raw logits[N,classes],normalized input,three stages and global average features.
/// @throws std::invalid_argument For invalid dimensions,dtype,normalization state or incompatible leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(587), state(593);
/// const auto out = m::SensorCnn(p, state, n::ones({2, 100, 6}), m::SensorConfig{6, 4, 8});
/// @endcode
/// @par References
/// - He et al.(2016),Deep Residual Learning for Image Recognition, https://arxiv.org/abs/1512.03385. Source teaching1D variant uses three strided5/3 convolution stages with learned1x1 skip,He-uniform initialization,per-channel fitting normalization and global temporal pooling,no BN or claim of exact paper architecture.
/// @see SensorOutput, SensorCE()
inline SensorOutput SensorCnn(nn::ParameterStore &p, nn::ParameterStore &state,
                              const Tensor &signal, const SensorConfig &config = {}) {
  if (config.channels < 1 || config.classes < 2 || config.base < 1)
    throw std::invalid_argument("ECG/sensor model dimensions");
  if (signal.dim() != 3 || signal.size(0) < 1 || signal.size(1) < 1 ||
      signal.size(2) != config.channels || signal.dtype() != p.dtype() ||
      state.dtype() != p.dtype())
    throw std::invalid_argument("aligned floating temporal channel input/stores required");
  SensorOutput result;
  const Tensor mean = state.get("normalization/mean", {config.channels}, nn::init::zeros());
  const Tensor stddev = state.get("normalization/std", {config.channels}, nn::init::ones());
  if (!nn::all(nn::is_finite(mean)).item<bool>() ||
      !nn::all(nn::logical_and(nn::is_finite(stddev), nn::greater(stddev, 0.))).item<bool>())
    throw std::invalid_argument("finite saved channel mean and positive std required");
  result.normalized = nn::div(nn::sub(signal, mean.detach()), stddev.detach());
  Tensor x = result.normalized;
  for (int block = 0; block < 3; ++block) {
    const auto name = "stage" + std::to_string(block + 1);
    const int width = config.base * (1 << block);
    nn::ConvLayerOptions first;
    first.stride = 2;
    first.init = nn::init::he_uniform();
    first.activation = nn::Activation::Relu;
    Tensor hidden = nn::conv1d(p, name + "/first", x, width, 5, first);
    nn::ConvLayerOptions second;
    second.init = nn::init::he_uniform();
    hidden = nn::conv1d(p, name + "/second", hidden, width, 3, second);
    nn::ConvLayerOptions projection;
    projection.stride = 2;
    projection.init = nn::init::he_uniform();
    const Tensor skip = nn::conv1d(p, name + "/projection", x, width, 1, projection);
    x = nn::relu(nn::add(hidden, skip));
    result.stages.push_back(x);
  }
  result.features = nn::mean(x, {1});
  result.logits = nn::dense(p, "head", result.features, config.classes);
  return result;
}

/// @brief Compute class-weighted categorical CE with raw-observation denominator.
///
/// Training weights convention is N/(classes*class_count),estimated only on
/// fitting labels. Stable log-softmax retains large-logit numerical accuracy.
/// Callers show grad/optimizer step explicitly;all labels/weights are absent
/// from the inference graph. Zero class weight contributes zero finite loss.
///
/// @param logits Floating[N,classes] raw scores,N>0.
/// @param labels Bounded Int32/64[N] original zero-based class identities.
/// @param weights Finite nonnegative[classes] fitting-only class weights.
/// @param classes Number of score columns>=2.
/// @return Scalar -mean_n sum_class onehot*log_softmax*weight,not divided by total weight.
/// @throws std::invalid_argument For invalid geometry,label bounds or weight values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto z = n::tensor({2., 0., 0., 2.}, {2, 2}).requires_grad_();
/// const auto labels = n::tensor({0, 1}).to(n::DType::Int32);
/// const auto loss = m::SensorCE(z, labels, n::tensor({2., .5}), 2);
/// const auto g = n::grad(loss, {z})[0];
/// @endcode
/// @see SensorCnn(), ScoreSensor()

inline Tensor SensorCE(const Tensor &logits, const Tensor &labels, const Tensor &weights,
                       int classes) {
  if (logits.dim() != 2 || logits.size(0) < 1 || classes < 2 || logits.size(1) != classes ||
      labels.shape() != nn::Shape{logits.size(0)} ||
      (labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64) ||
      weights.shape() != nn::Shape{classes} ||
      !nn::all(
           nn::logical_and(nn::greater_equal(labels, 0.), nn::less(labels, double(classes))))
           .item<bool>() ||
      !nn::all(nn::logical_and(nn::is_finite(weights), nn::greater_equal(weights, 0.)))
           .item<bool>())
    throw std::invalid_argument(
        "aligned sensor logits,bounded labels and nonnegative fitting weights required");
  return -nn::mean(nn::sum(nn::one_hot(labels, classes, 1., 0., logits.dtype()) *
                               nn::log_softmax(logits, 1) * weights,
                           {1}));
}
} // namespace datamunge::nn::models
