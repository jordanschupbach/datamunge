#pragma once

/// @file episodic_classifier.hpp
/// @brief Shared convolutional encoder and few-shot scoring rules.

#include <stdexcept>
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Choose a support-conditioned scoring rule.
/// @par Example
/// @code
/// auto rule=dm::nn::EpisodeMethod::Prototype;
/// @endcode
enum class EpisodeMethod {
  Matching, ///< Cosine attention over individual support examples.
  Prototype, ///< Negative squared distance to each class mean.
  Relation, ///< Learned comparison with each class sum.
};

/// @brief Features and class predictions for one episode.
/// @par Example
/// @code
/// dm::nn::EpisodeScores result;
/// @endcode
struct EpisodeScores {
  Tensor embedding; ///< Support and query embeddings, in input row order.
  Tensor prototype; ///< Class means or sums, undefined for matching.
  Tensor scores; ///< Support attention for matching, class logits otherwise.
  Tensor probability; ///< Per-query class probabilities or independent relation scores.
};

/// @brief Encode 28×28 mono support and query images and score the queries.
/// @param model Trainable store initialized on first use. Use one store per method.
/// @param images Floating [support+query,28,28,1], support rows first.
/// @param labels Integer support labels [support].
/// @param classes Positive class count, with at least one support row per class.
/// @param method Matching, prototype or relation scoring.
/// @return Embeddings, optional class summaries, scores and probabilities.
/// @throws std::invalid_argument For invalid shapes, labels, dtype or method.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// auto out=dm::nn::episodic_classify(model,dm::nn::zeros({3,28,28,1}),
///     dm::nn::arange(0,2,1,dm::nn::DType::Int32),2,dm::nn::EpisodeMethod::Prototype);
/// @endcode
/// @par References
/// - Vinyals et al. (2016), Matching Networks. https://arxiv.org/abs/1606.04080
/// - Snell et al. (2017), Prototypical Networks. https://arxiv.org/abs/1703.05175
/// - Sung et al. (2018), Learning to Compare. https://arxiv.org/abs/1711.06025
inline EpisodeScores episodic_classify(ParameterStore& model,const Tensor& images,const Tensor& labels,
                                       int classes,EpisodeMethod method) {
  if (!images.defined() || !labels.defined() || images.dim()!=4 || labels.dim()!=1 ||
      labels.size(0)<1 || images.size(0)<=labels.size(0) ||
      images.shape()!=Shape{images.size(0),28,28,1} || images.dtype()!=model.dtype() ||
      (labels.dtype()!=DType::Int32 && labels.dtype()!=DType::Int64) ||
      classes<1 || min(labels).item()<0 || max(labels).item()>=classes ||
      (method!=EpisodeMethod::Matching && method!=EpisodeMethod::Prototype && method!=EpisodeMethod::Relation))
    throw std::invalid_argument("Invalid few-shot episode");
  const auto membership=one_hot(labels,classes,1,0,model.dtype());
  const auto counts=sum(membership,{0});
  if (min(counts).item()<1) throw std::invalid_argument("Missing support class");
  const ConvLayerOptions options{.activation=Activation::Relu};
  const auto first=max_pool2d(conv2d(model,"conv1",images,12,3,options),2,2);
  const auto second=max_pool2d(conv2d(model,"conv2",first,16,3,options),2,2);
  auto embedding=dense(model,"embedding",reshape(second,{images.size(0),784}),32);
  if (method==EpisodeMethod::Matching)
    embedding=embedding/sqrt(maximum(sum(square(embedding),{1},true),full({},1e-10,model.dtype())));
  const auto support=slice_axis(embedding,0,0,labels.size(0));
  const auto query=slice_axis(embedding,0,labels.size(0),images.size(0));
  if (method==EpisodeMethod::Matching) {
    const auto score=softmax(12*matmul(query,transpose(support)));
    return {embedding,Tensor(),score,matmul(score,membership)};
  }
  const auto total=matmul(transpose(membership),support);
  if (method==EpisodeMethod::Prototype) {
    const auto prototype=total/expand_dims(counts,1);
    const auto score=-sum(square(expand_dims(query,1)-expand_dims(prototype,0)),{2});
    return {embedding,prototype,score,softmax(score)};
  }
  const Shape shape{query.size(0),classes,32};
  const auto pair=concat({broadcast_to(expand_dims(total,0),shape),
                          broadcast_to(expand_dims(query,1),shape)},2);
  const auto hidden=dense(model,"relation_hidden",pair,32,Activation::Relu);
  const auto score=squeeze(dense(model,"relation_head",hidden,1),{2});
  return {embedding,total,score,sigmoid(score)};
}

}  // namespace datamunge::nn
