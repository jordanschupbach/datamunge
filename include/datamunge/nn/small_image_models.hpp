#pragma once
/// @file small_image_models.hpp
/// @brief One-convolution tensor image classifiers with explicit local/global max-pooling and no fitting loops.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
namespace datamunge::nn::models {
/// @brief Shape controls for a shared VALID convolution, ReLU, max-pool and dense classifier.
/// Local pooling uses nonoverlapping pool_size square windows and drops incomplete
/// trailing windows. Global pooling uses the full rectangular convolution map.
/// Global max is invariant to spatial permutations of that map, not every finite
/// input-image translation. Both heads retain one ReLU dense hidden layer.
/// @par Example
/// @code
/// dm::nn::models::SmallImageCnnConfig c;c.global_pool=true;c.hidden=16;
/// dm::nn::ParameterStore p(1);
/// const auto out=dm::nn::models::SmallImageCnn(p,dm::nn::zeros({1,12,12,1}),c);
/// std::print("{}\n",p.count()); // 258
/// @endcode
/// @see SmallImageCnn(), SmallImageCnnOutputs
struct SmallImageCnnConfig {
  std::int64_t filters{8}; ///< Positive number of shared convolution output channels.
  std::int64_t kernel{3}; ///< Positive square VALID kernel, no greater than input spatial dimensions.
  std::int64_t pool_size{2}; ///< Positive local square window/stride; validated positive even in global mode.
  std::int64_t hidden{32}; ///< Positive dense ReLU width.
  std::int64_t classes{2}; ///< Output categorical-logit width, at least two.
  bool global_pool{false}; ///< Full-map max if true; local nonoverlapping max otherwise.
};
/// @brief Actual forward endpoints and max-pool winners, retaining their tensor graphs.
/// NHWC convolution/pooling; flat/hidden/logits use batch-major matrices. pool_argmax
/// is an owned integer tensor with within-image flat indices for max_unpool2d.
/// Tensor handles retain storage/graphs; this record alone performs no forward or
/// fit. Endpoints are not detached, so gradients reach input and all six leaves.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(1);
/// const auto out=dm::nn::models::SmallImageCnn(p,dm::nn::zeros({1,12,12,1}));
/// std::print("{} {}\n",out.pooled.size(1),out.logits.size(1)); // 5 2
/// @endcode
/// @see SmallImageCnn(), SmallImageCnnConfig
struct SmallImageCnnOutputs {
  Tensor image; ///< Original [N,H,W,C] input handle.
  Tensor conv_pre; ///< Actual biased VALID convolution [N,H-k+1,W-k+1,F].
  Tensor conv_relu; ///< Post-ReLU convolution map.
  Tensor pooled; ///< Actual local/global maximum values, NHWC.
  Tensor pool_argmax; ///< Int64 actual within-image winners; ties choose first window-scan maximum.
  Tensor flat; ///< Actual pooled map reshaped [N,pooled_values_per_image].
  Tensor hidden_pre; ///< Actual biased hidden affine map [N,hidden].
  Tensor hidden; ///< Post-ReLU hidden activations.
  Tensor logits; ///< Unnormalised [N,classes] class scores.
};
/// @brief Build/reuse a small differentiable VALID-convolution image classifier without fitting.
/// Six leaves: prefix/conv/{W,b}, prefix/hidden/{W,b}, prefix/output/{W,b}.
/// Convolution and hidden weights use He uniform; output weights use Glorot uniform;
/// all biases start zero. ReLU is explicit and pooling retains actual winners.
/// Input/store must share Float32 or Float64 dtype and positive NHWC dimensions.
/// Shape/configuration checks precede lazy creation; existing leaf-shape conflicts
/// may reject after earlier compatible leaves were accessed. Pixel values follow
/// ordinary tensor arithmetic: this forward does not reject nonfinite data.
/// No optimizer step or input mutation; graph handles/parameters are retained.
/// Work is ordinary dense convolution/pooling/head work; storage retains requested
/// autodiff activations. Concurrent mutation/lazy creation in one store is unsafe.
/// @param p Mutable learnable parameter store, dtype matching image.
/// @param image Floating positive NHWC image batch.
/// @param config Shared-kernel, pooling and head shapes, default 8/3/2/32/2 local.
/// @param prefix Nonempty parameter-name prefix, default cnn.
/// @return Actual forward endpoints and integer pooling winners, retaining autodiff.
/// @throws std::invalid_argument If shape, dtype, prefix, config or existing leaf geometry is incompatible.
/// @throws std::overflow_error If complete tensor dimensions cannot fit.
/// @throws std::length_error If storage capacity is exceeded.
/// @throws std::bad_alloc If initialization or tensor allocation fails.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(1);
/// const auto out=dm::nn::models::SmallImageCnn(p,dm::nn::zeros({1,12,12,1}));
/// std::print("{}\n",p.count()); // 6578
/// @endcode
/// @par References
/// - LeCun et al. (1998), Gradient-based learning applied to document recognition,
///   https://doi.org/10.1109/5.726791. This is a custom small ReLU/max-pool classifier,
///   not LeNet-5 or a pretrained network.
/// @see SmallImageCnnConfig, DenseImageBaseline()
SmallImageCnnOutputs SmallImageCnn(ParameterStore& p,const Tensor& image,SmallImageCnnConfig config={},const std::string& prefix="cnn");
/// @brief Build/reuse a flatten-ReLU-dense image baseline with two learned affine maps.
/// No fitting. Four leaves prefix/hidden/{W,b} and prefix/output/{W,b}; hidden
/// He-uniform and output Glorot-uniform weights, zero biases. Positive NHWC input
/// shares Float32/Float64 dtype with p; flattening retains spatial location.
/// No input mutation; lazy parameter creation is not safe against concurrent store
/// mutation. Graphs remain differentiable in input and learned leaves.
/// @param p Mutable matching floating parameter store.
/// @param image Positive floating NHWC image batch.
/// @param hidden Positive hidden ReLU width, default 32.
/// @param classes Categorical-logit width at least two, default two.
/// @param prefix Nonempty parameter prefix, default mlp.
/// @return Actual unnormalised [N,classes] logits.
/// @throws std::invalid_argument If input/config/prefix/dtype or existing leaves are incompatible.
/// @throws std::overflow_error If tensor dimensions cannot fit.
/// @throws std::length_error If storage capacity is exceeded.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(1);
/// const auto out=dm::nn::models::DenseImageBaseline(p,dm::nn::zeros({1,12,12,1}));
/// std::print("{}\n",p.count()); // 4706
/// @endcode
/// @see SmallImageCnn()
Tensor DenseImageBaseline(ParameterStore& p,const Tensor& image,std::int64_t hidden=32,std::int64_t classes=2,const std::string& prefix="mlp");
}
