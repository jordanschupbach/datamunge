#pragma once

/// @file stacked_vqa.hpp
/// @brief Two-hop spatial attention for visual question answering.

#include <stdexcept>
#include <string>
#include <vector>

#include "datamunge/nn/vqa_models.hpp"

namespace datamunge::nn {

/// @brief Answer logits and interpretable intermediate attention tensors.
/// @par Example
/// @code
/// dm::nn::StackedVqaResult result;
/// @endcode
struct StackedVqaResult {
  Tensor logits; ///< Answer scores [batch,3].
  Tensor attention; ///< Two spatial attention maps [batch,2,16].
  Tensor regions; ///< Spatial features with coordinates [batch,16,16].
  Tensor states; ///< Question and its two refinements [batch,3,16].
};

/// @brief Answer color questions using two successive spatial reads.
/// The first read refines a length-masked LSTM representation and the
/// second read uses that refined state. Coordinates are appended to CNN regions.
/// @param model Trainable store initialized on first use.
/// @param image RGB input [batch,8,8,3] in model dtype.
/// @param tokens Integer question IDs [batch,3] in [0,3].
/// @param lengths Integer unpadded lengths [batch] in [0,3].
/// @return Answer logits, attention maps, region features and query states.
/// @throws std::invalid_argument For incompatible shapes, tokens, dtype or parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// auto out=dm::nn::stacked_vqa(model,dm::nn::zeros({1,8,8,3}),
///    dm::nn::zeros({1,3},dm::nn::DType::Int32),dm::nn::zeros({1},dm::nn::DType::Int32));
/// @endcode
/// @par References
/// - Yang et al. (2016), Stacked Attention Networks for Image Question Answering.
///   https://arxiv.org/abs/1511.02274
/// @see models::VqaLstm()
inline StackedVqaResult stacked_vqa(ParameterStore& model,const Tensor& image,
                                   const Tensor& tokens,const Tensor& lengths) {
  if (!image.defined() || image.dim()!=4 || image.size(0)<1 ||
      image.shape()!=Shape{image.size(0),8,8,3} || image.dtype()!=model.dtype() ||
      !tokens.defined() || !lengths.defined() || tokens.shape()!=Shape{image.size(0),3}) {
    throw std::invalid_argument("Invalid stacked VQA inputs");
  }
  const auto count=image.size(0);
  const auto features=conv2d(model,"image",image,14,3,{.activation=Activation::Relu});
  const auto values=reshape(avg_pool2d(features,2,2),{count,16,14});
  std::vector<double> xy;
  for (int j=0;j<16;++j) {
    xy.push_back((j%4-1.5)/1.5);
    xy.push_back((j/4-1.5)/1.5);
  }
  const auto grid=from_vector(xy,{1,16,2}).to(model.dtype());
  const auto regions=concat({values,broadcast_to(grid,{count,16,2})},2);
  auto state=models::VqaLstm(model,tokens,lengths,{.vocabulary=4,.time=3,.embedding=8,.hidden=16}).h;
  std::vector<Tensor> maps,states{state};
  for (int hop=0;hop<2;++hop) {
    const auto name="hop_"+std::to_string(hop);
    const auto keys=dense(model,name+"/image",regions,16);
    const auto query=expand_dims(dense(model,name+"/query",state,16),1);
    const auto scores=squeeze(dense(model,name+"/score",tanh(keys+query),1),{2});
    const auto mass=softmax(scores);
    state=state+sum(regions*expand_dims(mass,2),{1});
    maps.push_back(mass);
    states.push_back(state);
  }
  return {dense(model,"answers",state,3),stack(maps,1),regions,stack(states,1)};
}

}  // namespace datamunge::nn
