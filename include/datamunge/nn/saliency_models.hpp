#pragma once
/// @file saliency_models.hpp
/// @brief Shared digit classifier,raw-logit input gradients and original MNIST data contracts.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/training.hpp"
#include <array>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Represent an explicit post-activation visualization gate.
///
/// Callable(Tensor)->Tensor. Default empty gate selects ordinary ReLU.
/// Custom guided rules can preserve forward values while modifying backward
/// semantics;they need not be the mathematical derivative of the forward model.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SaliencyGate gate = [](const n::Tensor &x) {
///   return n::relu(x);
/// };
/// @endcode
/// @par References
/// - Simonyan,Vedaldi and Zisserman(2013/2014),Deep Inside Convolutional Networks:Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034. This repository uses its own small26826parameter MNIST classifier and raw selected-logit input derivatives,not paper pretrained ImageNet weights or a causal attribution claim.
/// @see SaliencyCnn(),VanillaInputGradient()
using SaliencyGate = std::function<Tensor(const Tensor &)>;

/// @brief Own every endpoint of the shared two-convolution digit classifier.
///
/// Owning handles retain storage/graph. Conv-only execution leaves hidden/logits
/// undefined and supports other image sizes for feature optimization octaves.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(1103);
/// const auto out = m::SaliencyCnn(p, n::zeros({1, 28, 28, 1}));
/// @endcode
/// @par References
/// - Simonyan,Vedaldi and Zisserman(2013/2014),Deep Inside Convolutional Networks:Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034. This repository uses its own small26826parameter MNIST classifier and raw selected-logit input derivatives,not paper pretrained ImageNet weights or a causal attribution claim.
/// @see SaliencyCnn(),VanillaInputGradient()
struct SaliencyCnnOutput {
  Tensor conv1;  ///< First post-gate convolution[N,H,W,8].
  Tensor pool1;  ///< VALID2/stride2 average-pooled first map.
  Tensor conv2;  ///< Second post-gate convolution[N,H/2,W/2,16].
  Tensor pool2;  ///< Second VALID2/stride2 average-pooled map.
  Tensor hidden; ///< Post-gate dense features[N,32],undefined for conv-only requests.
  Tensor logits; ///< Raw categorical scores[N,10],undefined for conv-only requests.
};

/// @brief Apply both shared digit convolutions and average pools without the dense head.
///
/// SAME5x5/8HeUniform/gate,VALIDavg2,SAME3x3/16HeUniform/gate,VALIDavg2.
/// No BN or dropout. Lazily creates four learned leaves;no optimizer update.
///
/// @param p Mutable Float32 learned store;conv parameters lazily initialized.
/// @param image Float32[N,H,W,1],Npositive,H/W>=4.
/// @param gate Optional explicit callable,empty uses ordinary ReLU.
/// @return Four complete convolution/pool endpoints;hidden and logits undefined.
/// @throws std::invalid_argument For invalid input rank/dtype/geometry or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p;
/// const auto f = m::SaliencyConvFeatures(p, n::zeros({1, 20, 20, 1}));
/// @endcode
/// @par References
/// - Simonyan,Vedaldi and Zisserman(2013/2014),Deep Inside Convolutional Networks:Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034. This repository uses its own small26826parameter MNIST classifier and raw selected-logit input derivatives,not paper pretrained ImageNet weights or a causal attribution claim.
/// @see SaliencyCnn(),VanillaInputGradient()
inline SaliencyCnnOutput SaliencyConvFeatures(nn::ParameterStore &p, const Tensor &image,
                                              const SaliencyGate &gate = {}) {
  const SaliencyGate relu = gate ? gate : SaliencyGate([](const Tensor &x) {
    return nn::relu(x);
  });
  nn::ConvLayerOptions opt;
  opt.init = nn::init::he_uniform();
  if (image.dim() != 4 || image.size(0) < 1 || image.size(1) < 4 || image.size(2) < 4 ||
      image.size(3) != 1 || image.dtype() != nn::DType::Float32 || p.dtype() != image.dtype())
    throw std::invalid_argument("positive Float32 NHWC grayscale input required");
  SaliencyCnnOutput out;
  out.conv1 = relu(nn::conv2d(p, "conv1", image, 8, 5, opt));
  out.pool1 = nn::avg_pool2d(out.conv1, 2, 2, nn::Padding::Valid);
  out.conv2 = relu(nn::conv2d(p, "conv2", out.pool1, 16, 3, opt));
  out.pool2 = nn::avg_pool2d(out.conv2, 2, 2, nn::Padding::Valid);
  return out;
}

/// @brief Classify actual28px grayscale digits with the complete shared saliency CNN.
///
/// Two SAMEconvs/VALIDavg2pools,flatten784,GlorotFC32/gate,Glorotlinear10head,
/// bias0,noBN/dropout.26826values/eight leaves. All three hidden activations
/// use the same explicit gate. Lazy initialization only;fitting loop belongs to caller.
///
/// @param p Mutable Float32 learned store;warm all8leaves before optimizer construction.
/// @param image Float32[N,28,28,1],normally original MNIST bytes/255.
/// @param gate Optional post-activation callable,empty uses ordinary ReLU.
/// @return All six endpoints and raw[N,10]categorical logits.
/// @throws std::invalid_argument For invalid shape/dtype or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(1103);
/// const auto out = m::SaliencyCnn(p, n::zeros({1, 28, 28, 1}));
/// @endcode
/// @par References
/// - Simonyan,Vedaldi and Zisserman(2013/2014),Deep Inside Convolutional Networks:Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034. This repository uses its own small26826parameter MNIST classifier and raw selected-logit input derivatives,not paper pretrained ImageNet weights or a causal attribution claim.
/// @see SaliencyCnn(),VanillaInputGradient()
inline SaliencyCnnOutput SaliencyCnn(nn::ParameterStore &p, const Tensor &image,
                                     const SaliencyGate &gate = {}) {
  const SaliencyGate relu = gate ? gate : SaliencyGate([](const Tensor &x) {
    return nn::relu(x);
  });
  if (image.dim() != 4 || image.size(1) != 28 || image.size(2) != 28)
    throw std::invalid_argument("full saliency classifier requires28px inputs");
  auto out = SaliencyConvFeatures(p, image, gate);
  out.hidden =
      relu(nn::dense(p, "hidden", nn::reshape(out.pool2, {out.pool2.size(0), 784}), 32));
  out.logits = nn::dense(p, "head", out.hidden, 10);
  return out;
}

/// @brief Select one fixed RAW categorical logit per image.
///
/// The target vector stays fixed during perturbation/integration. Caller sums
/// per-image scores before input differentiation so each row has unit scale.
///
/// @param logits Floating[N,classes]raw scores.
/// @param target Int32[N]explicit fixed categories in0..classes-1.
/// @param classes Head category count,default10.
/// @return Live floating[N]selected raw scores,without softmax or batch averaging.
/// @throws std::invalid_argument For invalid row/class/target shape,dtype or range.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto score = m::SaliencyScore(n::ones({2, 10}), n::zeros({2}, n::DType::Int32));
/// @endcode
/// @par References
/// - Simonyan,Vedaldi and Zisserman(2013/2014),Deep Inside Convolutional Networks:Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034. This repository uses its own small26826parameter MNIST classifier and raw selected-logit input derivatives,not paper pretrained ImageNet weights or a causal attribution claim.
/// @see SaliencyCnn(),VanillaInputGradient()
inline Tensor SaliencyScore(const Tensor &logits, const Tensor &target, int classes = 10) {
  if (logits.dim() != 2 || target.shape() != nn::Shape{logits.size(0)} ||
      target.dtype() != nn::DType::Int32 || classes < 2 || logits.size(1) != classes ||
      nn::min(target).item() < 0 || nn::max(target).item() >= classes)
    throw std::invalid_argument("one valid fixed target per logit row required");
  return nn::sum(nn::mul(logits, nn::one_hot(target, classes, 1.0, 0.0, logits.dtype())), {1});
}

/// @brief Differentiate the SUM of selected raw logits with respect to detached input images.
///
/// Creates no model optimizer/update;input source and learned leaves stay unchanged.
/// For independent classifier rows,this equals per-image differentiation,not a
/// batch-mean-scaled derivative. Signed local sensitivity is not a causal explanation.
///
/// @param forward Callable(input)->raw[N,classes]logits,with independent batch rows.
/// @param image Input image tensor,detached/cloned before enabling gradients.
/// @param target Int32[N]fixed selected classes,not recomputed after perturbation.
/// @param classes Head class count,default10.
/// @return Detached input-shaped ordinary derivative of the sum of selected raw scores.
/// @throws std::invalid_argument For invalid target/logit contract or unavailable derivative path.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(1103);
/// m::SaliencyCnn(p, n::zeros({1, 28, 28, 1}));
/// const auto g = m::VanillaInputGradient(
///     [&](const n::Tensor &x) {
///       return m::SaliencyCnn(p, x).logits;
///     },
///     n::zeros({1, 28, 28, 1}), n::zeros({1}, n::DType::Int32));
/// @endcode
/// @par References
/// - Simonyan,Vedaldi and Zisserman(2013/2014),Deep Inside Convolutional Networks:Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034. This repository uses its own small26826parameter MNIST classifier and raw selected-logit input derivatives,not paper pretrained ImageNet weights or a causal attribution claim.
/// @see SaliencyCnn(),VanillaInputGradient()
inline Tensor VanillaInputGradient(const std::function<Tensor(const Tensor &)> &forward,
                                   const Tensor &image, const Tensor &target,
                                   int classes = 10) {
  Tensor x = image.detach().clone();
  x.requires_grad_(true);
  return nn::grad(nn::sum(SaliencyScore(forward(x), target, classes)), {x})[0].detach();
}

/// @brief Own aligned original MNIST observations,class labels and source IDs.
///
/// ImagesFloat32[N,28,28,1]in[0,1],labelsInt32[N]0..9,IDsInt32[N].
/// No optimizer or downloaded state belongs to this value type.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SaliencyData d{n::zeros({1, 28, 28, 1}), n::zeros({1}, n::DType::Int32),
///                   n::zeros({1}, n::DType::Int32)};
/// @endcode
/// @par References
/// - Simonyan,Vedaldi and Zisserman(2013/2014),Deep Inside Convolutional Networks:Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034. This repository uses its own small26826parameter MNIST classifier and raw selected-logit input derivatives,not paper pretrained ImageNet weights or a causal attribution claim.
/// @see SaliencyCnn(),VanillaInputGradient()
struct SaliencyData {
  Tensor images; ///< Float32[N,28,28,1]original grayscale bytes/255.
  Tensor labels; ///< Int32[N]digit classes0..9.
  Tensor source; ///< Int32[N]unique original MNIST row ID,official test offset60000.
};

/// @brief Gather aligned original digit observations and metadata with tensors.
///
/// @param d Source image/label/ID handles.
/// @param rows Host row-index metadata,duplicates/order preserved.
/// @return New gathered image/label/ID handles;source unchanged.
/// @throws std::invalid_argument For invalid row indices or incompatible axes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SaliencyData d{n::zeros({2, 28, 28, 1}), n::zeros({2}, n::DType::Int32),
///                   n::arange(0, 2, 1, n::DType::Int32)};
/// const auto chosen = m::SaliencyRows(d, {1});
/// @endcode
/// @par References
/// - Simonyan,Vedaldi and Zisserman(2013/2014),Deep Inside Convolutional Networks:Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034. This repository uses its own small26826parameter MNIST classifier and raw selected-logit input derivatives,not paper pretrained ImageNet weights or a causal attribution claim.
/// @see SaliencyCnn(),VanillaInputGradient()
inline SaliencyData SaliencyRows(const SaliencyData &d,
                                 const std::vector<std::int64_t> &rows) {
  return {nn::take_rows(d.images, rows), nn::take_rows(d.labels, rows),
          nn::take_rows(d.source, rows)};
}

/// @brief Load original pinned balanced MNIST saliency pools from a local archive.
///
/// Requiresoriginal28pxInt32ingressbytes0..255,Int32labels/IDs,balanced classes,
/// unique source IDs and original pixelbytes across all three pools. Officialtest
/// IDs60000..69999,fit/validation0..59999. TensorFloat32byte/255normalization.
/// Archive IDs preserve observation provenance;no partition refitting or network access.
///
/// @param root Directory containing saliency_data.npz,no automatic download.
/// @return 1000fitting/200validation/200test aligned digit records.
/// @throws std::invalid_argument For missing archive/arrays,invalid shapes/dtypes/ranges,balance or duplicated source.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadSaliency(std::filesystem::path(std::getenv("HOME")) /
///                                    ".cache/datamunge/datasets/saliency");
/// @endcode
/// @par References
/// - Simonyan,Vedaldi and Zisserman(2013/2014),Deep Inside Convolutional Networks:Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034. This repository uses its own small26826parameter MNIST classifier and raw selected-logit input derivatives,not paper pretrained ImageNet weights or a causal attribution claim.
/// @see SaliencyCnn(),VanillaInputGradient()
inline std::vector<SaliencyData> LoadSaliency(const std::filesystem::path &root) {
  std::map<std::string, Tensor> a;
  for (const auto &[key, value] : nn::load_npz((root / "saliency_data.npz").string()))
    a.emplace(key, value);
  std::vector<SaliencyData> out;
  std::set<int> ids;
  std::set<std::string> bytes;
  for (const std::string split : {"fit", "validation", "test"}) {
    const int count = split == "fit" ? 1000 : 200;
    const auto x = a.at(split + "_images"), labels = a.at(split + "_labels"),
               source = a.at(split + "_source_ids");
    if (x.shape() != nn::Shape{count, 28, 28, 1} || x.dtype() != nn::DType::Int32 ||
        labels.shape() != nn::Shape{count} || labels.dtype() != nn::DType::Int32 ||
        source.shape() != nn::Shape{count} || source.dtype() != nn::DType::Int32 ||
        nn::min(x).item() < 0 || nn::max(x).item() > 255 || nn::min(labels).item() < 0 ||
        nn::max(labels).item() > 9)
      throw std::invalid_argument("original aligned MNIST byte/metadata geometry required");
    const auto lv = labels.to_vector<int>(), sv = source.to_vector<int>(),
               pixels = x.to_vector<int>();
    std::array<int, 10> counts{};
    for (int i = 0; i < count; ++i) {
      ++counts[lv[i]];
      if (!ids.insert(sv[i]).second || (split == "test" ? (sv[i] < 60000 || sv[i] >= 70000)
                                                        : (sv[i] < 0 || sv[i] >= 60000)))
        throw std::invalid_argument("disjoint original MNIST row IDs required");
      const auto key = std::string(reinterpret_cast<const char *>(pixels.data() + i * 784),
                                   784 * sizeof(int));
      if (!bytes.insert(key).second)
        throw std::invalid_argument("unique original MNIST pixel records required");
    }
    for (int c : counts)
      if (c != count / 10)
        throw std::invalid_argument("balanced original digit classes required");
    out.push_back({x.to(nn::DType::Float32) / 255., labels, source});
  }
  return out;
}

/// @brief Compute7x7coarse intensity summaries with tensor Double average pooling.
///
/// Parameter-free descriptor,no fitting or learned coefficients.
///
/// @param images Float32[N,28,28,1]original normalized images.
/// @return Float64[N,49]means of nonoverlapping4x4pixel cells.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto f = m::SaliencyFeatures(n::ones({2, 28, 28, 1}));
/// @endcode
/// @par References
/// - Simonyan,Vedaldi and Zisserman(2013/2014),Deep Inside Convolutional Networks:Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034. This repository uses its own small26826parameter MNIST classifier and raw selected-logit input derivatives,not paper pretrained ImageNet weights or a causal attribution claim.
/// @see SaliencyCnn(),VanillaInputGradient()
inline Tensor SaliencyFeatures(const Tensor &images) {
  return nn::avg_pool2d(images.to(nn::DType::Float64), 4, 4, nn::Padding::Valid)
      .reshape({images.size(0), 49});
}

/// @brief Predict digits by parameter-free retained-memory nearest coarse-intensity vector.
///
/// No fitted learnable parameters and no optimizer. Double49cell descriptors
/// and squared Euclidean distances use tensor operations;no query statistics.
///
/// @param fit Original fitting images/labels;source IDs unused.
/// @param test Query images;query labels and source IDs unused.
/// @return Float32[Nquery,10]one-hot neighbor scores,first fitting row wins distance ties.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SaliencyData d{n::zeros({2, 28, 28, 1}), n::zeros({2}, n::DType::Int32), {}};
/// const auto q = m::SaliencyNearest(d, d);
/// @endcode
/// @par References
/// - Simonyan,Vedaldi and Zisserman(2013/2014),Deep Inside Convolutional Networks:Visualising Image Classification Models and Saliency Maps, https://arxiv.org/abs/1312.6034. This repository uses its own small26826parameter MNIST classifier and raw selected-logit input derivatives,not paper pretrained ImageNet weights or a causal attribution claim.
/// @see SaliencyCnn(),VanillaInputGradient()
inline Tensor SaliencyNearest(const SaliencyData &fit, const SaliencyData &test) {
  const auto a = SaliencyFeatures(fit.images), b = SaliencyFeatures(test.images);
  const auto distance =
      nn::sum(nn::square(nn::expand_dims(b, 1) - nn::expand_dims(a, 0)), {2});
  return nn::one_hot(nn::gather(fit.labels, nn::argmin(distance, 1), 0), 10);
}
} // namespace datamunge::nn::models
