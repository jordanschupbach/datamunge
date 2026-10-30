#pragma once
/// @file kim_cnn_models.hpp
/// @brief Random word embeddings,wide3/4/5 sentence convolutions and explicit head projection.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <cmath>
#include <limits>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Configure random-embedding Kim sentence-CNN token and branch widths.
///
/// Architecture metadata only. Fitting data chooses vocabulary and time;
/// original full E300/F100 and compactE32/F32 retain all three branches.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::KimConfig c{8000, 64, 32, 32};
/// @endcode
/// @par References
/// - Kim(2014),Convolutional Neural Networks for Sentence Classification, https://aclanthology.org/D14-1181/,doi:10.3115/v1/D14-1181. Source CNN-rand uses learned random embeddings,parallel3/4/5 wide convolutions,ReLU/max-over-time/dropout and explicit caller-owned Adadelta/head maxnorm3;compact32/32 widths and tiny MR holdout are teaching settings.
/// @see KimCnn()
struct KimConfig {
  int vocabulary = 8000; ///< Token inventory including structural PAD0 and learned UNK1.
  int time = 64;         ///< Positive fixed input length after truncation/right padding.
  int embedding = 300;   ///< Positive learned embedding dimension;compact32.
  int filters = 100;     ///< Positive channels per3/4/5 branch;compact32.
};
/// @brief Own raw sentence logits,the complete zero-PAD table and all branch features.
///
/// Handles retain graph/storage lifetimes. Returned features precede dropout;
/// no class labels,normalization state or optimizer are owned by this result.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::KimConfig c{8, 5, 3, 2};
/// n::ParameterStore p(317);
/// const auto out =
///     m::KimCnn(p, n::ones({2, 5}, n::DType::Int32), n::full({2}, 3., n::DType::Int32), c);
/// @endcode
/// @par References
/// - Kim(2014),Convolutional Neural Networks for Sentence Classification, https://aclanthology.org/D14-1181/,doi:10.3115/v1/D14-1181. Source CNN-rand uses learned random embeddings,parallel3/4/5 wide convolutions,ReLU/max-over-time/dropout and explicit caller-owned Adadelta/head maxnorm3;compact32/32 widths and tiny MR holdout are teaching settings.
/// @see KimCnn()
struct KimOutput {
  Tensor logits;    ///< Linear binary[N,2] class scores.
  Tensor features;  ///< Concatenated undropped[N,3filters] max-over-time features.
  Tensor embedding; ///< Full[V,E] table with structural zero PAD row0 and learned rows1..V-1.
  std::vector<Tensor> maps; ///< Wide ReLU maps for kernel3,4,5,in that order.
};

/// @brief Build a tensor time<observed-length mask for padded sequence metadata.
///
/// Lengths are metadata,not fitted values. No normalization or numeric host
/// loop;zero length produces an entirely zero row.
///
/// @param lengths Int32/64[N] observed lengths0..time,N>0.
/// @param time Positive number of padded time positions<=Int32 maximum.
/// @param dtype Result dtype,defaultFloat32;Bool and other numeric casts follow tensor rules.
/// @return Tensor[N,time],one for active positions,zero otherwise.
/// @throws std::invalid_argument For invalid shape,dtype,time or out-of-bounds length.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto mask = m::LengthMask(n::tensor({2, 0}).to(n::DType::Int32), 4);
/// @endcode
/// @see KimCnn()
inline Tensor LengthMask(const Tensor &lengths, std::int64_t time,
                         nn::DType dtype = nn::DType::Float32) {
  if (lengths.dim() != 1 || lengths.size(0) < 1 || time < 1 ||
      time > std::numeric_limits<std::int32_t>::max() ||
      (lengths.dtype() != nn::DType::Int32 && lengths.dtype() != nn::DType::Int64) ||
      !nn::all(nn::logical_and(nn::greater_equal(lengths, 0.),
                               nn::less_equal(lengths, double(time))))
           .item<bool>())
    throw std::invalid_argument("bounded integer sequence lengths and positive time required");
  const Tensor positions = nn::arange(0, static_cast<double>(time), 1, nn::DType::Int32);
  return nn::cast(nn::less(nn::expand_dims(positions, 0),
                           nn::expand_dims(nn::cast(lengths, nn::DType::Int32), 1)),
                  dtype);
}

/// @brief Project each classifier-weight column to Euclidean norm<=3 using tensors.
///
/// Projection does not mutate w. Callers explicitly copy the result into the
/// learned leaf under NoGradGuard AFTER their optimizer step. This keeps the
/// optimizer and constraint visible;biases and other kernels are not projected.
/// Zero columns remain zero;no epsilon or hidden fitting modifies the cap.
///
/// @param w Finite floating[features,classes] classifier kernel.
/// @return Independent tensor expression w/max(1,column_norm/3),same shape/dtype.
/// @throws std::invalid_argument For invalid matrix geometry or nonfinite values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto projected = m::KimMaxNorm(n::tensor({3., .1, 4., .2}, {2, 2}));
/// @endcode
/// @par References
/// - Kim(2014),Convolutional Neural Networks for Sentence Classification, https://aclanthology.org/D14-1181/,doi:10.3115/v1/D14-1181. Source CNN-rand uses learned random embeddings,parallel3/4/5 wide convolutions,ReLU/max-over-time/dropout and explicit caller-owned Adadelta/head maxnorm3;compact32/32 widths and tiny MR holdout are teaching settings.
/// @see KimCnn()
inline Tensor KimMaxNorm(const Tensor &w) {
  if (w.dim() != 2 || w.size(0) < 1 || w.size(1) < 1 ||
      !nn::all(nn::is_finite(w)).item<bool>())
    throw std::invalid_argument("finite nonempty classifier matrix required");
  const Tensor norm = nn::sqrt(nn::sum(nn::square(w), {0}, true));
  return nn::div(w, nn::maximum(nn::scalar_like(1.0, w), nn::div(norm, 3.0)));
}

/// @brief Classify observed word IDs with learned embeddings and three wide max-pooled convolutions.
///
/// Only[V-1,E] embedding/nonpad rows learn,U(-.25,.25);PAD row0 is constant0.
/// Mask gathered embeddings by observed length,pad4zero positions each side,
/// then parallel VALID kernels3/4/5 with Glorot-uniform weights/zero biases
/// and ReLU. Wide lengths are T+6,T+5,T+4. Maximum over ALL branch time
/// positions concatenates3filter features,optional dropout,then linear2logits.
/// Fully padded boundary windows remain candidates and can carry learned bias;
/// this matches source wide convolution,not a masked-window pooling variant.
/// No pretrained vectors,labels,optimizer or maxnorm update hide in forward.
/// Caller explicitly chooses Adadelta and projects only classifier/W after step.
///
/// @param p Mutable learned store;warm all9 leaves before constructing a tensor optimizer.
/// @param tokens Int32/64[N,time] IDs0..vocabulary-1,including bounded padding substitutions.
/// @param lengths Int32/64[N] observed lengths0..time;positions outside length mask to zero.
/// @param config Positive vocabulary>=2,time,embedding and branch-filter widths.
/// @param training False by default;true activates caller-generator dropout when keep<1.
/// @param keep Finite dropout keep probability(0,1],default1;source fitting uses.5.
/// @param gen Required mutable generator only when training&&keep<1.
/// @return Binary logits,undropped pooled features,full zero-PAD table and three ReLU maps.
/// @throws std::invalid_argument For invalid architecture,token/length geometry or bounds,dropout policy or stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::KimConfig c{8, 5, 3, 2};
/// n::ParameterStore p(317);
/// const auto out =
///     m::KimCnn(p, n::ones({2, 5}, n::DType::Int32), n::full({2}, 3., n::DType::Int32), c);
/// @endcode
/// @par References
/// - Kim(2014),Convolutional Neural Networks for Sentence Classification, https://aclanthology.org/D14-1181/,doi:10.3115/v1/D14-1181. Source CNN-rand uses learned random embeddings,parallel3/4/5 wide convolutions,ReLU/max-over-time/dropout and explicit caller-owned Adadelta/head maxnorm3;compact32/32 widths and tiny MR holdout are teaching settings.
/// @see LengthMask(), KimMaxNorm()
inline KimOutput KimCnn(nn::ParameterStore &p, const Tensor &tokens, const Tensor &lengths,
                        const KimConfig &config = {}, bool training = false, double keep = 1.0,
                        nn::Generator *gen = nullptr) {
  if (config.vocabulary < 2 || config.time < 1 || config.embedding < 1 || config.filters < 1)
    throw std::invalid_argument("Kim CNN dimensions");
  if (tokens.dim() != 2 || tokens.size(0) < 1 || tokens.size(1) != config.time ||
      lengths.shape() != nn::Shape{tokens.size(0)} ||
      (tokens.dtype() != nn::DType::Int32 && tokens.dtype() != nn::DType::Int64) ||
      !nn::all(nn::logical_and(nn::greater_equal(tokens, 0.),
                               nn::less(tokens, double(config.vocabulary))))
           .item<bool>() ||
      !std::isfinite(keep) || keep <= 0 || keep > 1 || (training && keep < 1 && !gen))
    throw std::invalid_argument(
        "bounded observed token IDs,lengths and explicit dropout policy required");
  KimOutput out;
  const Tensor learned = p.get("embedding/nonpad", {config.vocabulary - 1, config.embedding},
                               nn::init::uniform(.25));
  out.embedding = nn::concat({nn::zeros({1, config.embedding}, learned.dtype()), learned}, 0);
  const Tensor mask = LengthMask(lengths, config.time, learned.dtype());
  const Tensor x = nn::mul(nn::gather(out.embedding, tokens, 0), nn::expand_dims(mask, 2));
  const Tensor padded = nn::pad(x, {{0, 0}, {4, 4}, {0, 0}});
  std::vector<Tensor> pooled;
  for (int kernel : {3, 4, 5}) {
    nn::ConvLayerOptions opt;
    opt.padding = nn::Padding::Valid;
    opt.activation = nn::Activation::Relu;
    const Tensor map =
        nn::conv1d(p, "window" + std::to_string(kernel), padded, config.filters, kernel, opt);
    out.maps.push_back(map);
    pooled.push_back(nn::max(map, {1}));
  }
  out.features = nn::concat(pooled, 1);
  Tensor feature = out.features;
  if (training && keep < 1)
    feature = nn::dropout(feature, 1 - keep, true, *gen);
  out.logits = nn::dense(p, "classifier", feature, 2);
  return out;
}

} // namespace datamunge::nn::models
