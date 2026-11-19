#pragma once

/// @file xvector.hpp
/// @brief Temporal speaker embedding and enrollment verification.

#include <cmath>
#include <stdexcept>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Embed log-mel clips with a temporal CNN and statistics pooling.
/// @param model Store whose parameters initialize on first use.
/// @param audio Floating [batch, frames, bands] in the store dtype, with positive dimensions.
/// @return Embeddings [batch,16] before the speaker classifier activation.
/// @throws std::invalid_argument For incompatible input or stored parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(4);
/// auto vectors = dm::nn::xvector_embedding(model,dm::nn::zeros({2,38,24}));
/// @endcode
/// @par References
/// - Snyder et al. (2018), X-Vectors. doi:10.1109/ICASSP.2018.8461375
/// @see speaker_centroid_scores()
inline Tensor xvector_embedding(ParameterStore& model,const Tensor& audio) {
  if (!audio.defined() || audio.dim()!=3 || audio.size(0)<1 || audio.size(1)<1 ||
      audio.size(2)<1 || audio.dtype()!=model.dtype())
    throw std::invalid_argument("Invalid x-vector audio features");
  const ConvLayerOptions options{.activation=Activation::Relu};
  auto hidden=conv1d(model,"tdnn1",(audio+6)*.2,24,5,options);
  hidden=conv1d(model,"tdnn2",hidden,24,3,options);
  hidden=conv1d(model,"tdnn3",hidden,24,3,options);
  const auto center=mean(hidden,{1},true);
  const auto scale=sqrt(mean(square(hidden-center),{1})+1e-5);
  return dense(model,"segment",concat({squeeze(center,{1}),scale},1),16);
}

/// @brief Score probes against unit centroids of normalized enrollment embeddings.
/// @param gallery Enrollment embeddings [clips,width].
/// @param labels Integer enrollment class IDs [clips].
/// @param probes Probe embeddings [probes,width].
/// @param speakers Positive number of enrolled classes, all represented in labels.
/// @return Double-precision cosine scores [probes,speakers].
/// @throws std::invalid_argument For incompatible shapes, invalid labels or nonfinite or zero norms.
/// @par Example
/// @code
/// auto gallery = dm::nn::eye(2);
/// auto scores = dm::nn::speaker_centroid_scores(gallery,
///     dm::nn::arange(0,2,1,dm::nn::DType::Int64),gallery,2);
/// @endcode
/// @see xvector_embedding()
inline Tensor speaker_centroid_scores(const Tensor& gallery,const Tensor& labels,
                                      const Tensor& probes,int speakers) {
  if (!gallery.defined() || !probes.defined() || !labels.defined() ||
      gallery.dim()!=2 || probes.dim()!=2 || gallery.size(0)<1 || probes.size(0)<1 ||
      gallery.size(1)<1 || gallery.size(1)!=probes.size(1) ||
      labels.shape()!=Shape{gallery.size(0)} || speakers<1 ||
      (labels.dtype()!=DType::Int32 && labels.dtype()!=DType::Int64) ||
      min(labels).item()<0 || max(labels).item()>=speakers)
    throw std::invalid_argument("Invalid enrollment or probe dimensions");
  /// Normalize rows and reject singular or nonfinite vectors.
  const auto unit=[](const Tensor& x) {
    const auto norm=sqrt(sum(square(x),{1},true));
    for (double value : norm.to_vector<double>())
      if (!(value>0) || !std::isfinite(value)) throw std::invalid_argument("Invalid speaker vector norm");
    return x/norm;
  };
  const auto membership=one_hot(labels,speakers,1,0,DType::Float64);
  const auto centroid=unit(matmul(transpose(membership),unit(gallery.to(DType::Float64))));
  return matmul(unit(probes.to(DType::Float64)),transpose(centroid));
}

}  // namespace datamunge::nn
