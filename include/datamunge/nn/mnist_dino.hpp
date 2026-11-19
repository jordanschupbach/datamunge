#pragma once

/// @file mnist_dino.hpp
/// @brief DINO multi-crop training components for MNIST.

#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>

namespace datamunge::nn {

/// @brief Pooled features and normalized prototype scores for a DINO crop.
/// @par Example
/// @code
/// const dm::nn::MnistDinoOutput empty{};
/// @endcode
struct MnistDinoOutput {
  Tensor features; ///< Pooled encoder features [batch,24].
  Tensor logits; ///< Cosine scores against 32 learned prototypes [batch,32].
};

/// @brief Encode unit-scale MNIST pixels with a patch encoder and DINO projection head.
/// @param parameters Trainable CPU parameter store.
/// @param pixels Floating CPU [batch,28,28,1], matching the store dtype.
/// @return Pooled features and cosine scores.
/// @throws std::invalid_argument For incompatible image shape, dtype or backend.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto encoded = dm::nn::mnist_dino_encode(model, dm::nn::zeros({2,28,28,1}));
/// @endcode
/// @see dino_cross_view_loss()
inline MnistDinoOutput mnist_dino_encode(ParameterStore& parameters, const Tensor& pixels) {
  if (!pixels.defined() || pixels.dim()!=4 || pixels.size(0)<1 ||
      pixels.size(1)!=28 || pixels.size(2)!=28 || pixels.size(3)!=1 ||
      !is_floating(pixels.dtype()) || pixels.dtype()!=parameters.dtype() ||
      pixels.device().backend!=compute::Backend::CPU)
    throw std::invalid_argument("mnist_dino_encode: expected CPU [batch,28,28,1] pixels");
  const auto batch = pixels.size(0);
  const auto patches = pixels.reshape({batch,4,7,4,7}).transpose({0,1,3,2,4}).reshape({batch,16,49});
  const auto state = dense(parameters,"patch",patches,24) +
                     parameters.get("position",{1,16,24},init::normal(.02));
  const auto normalized = layer_norm(parameters,"encoder/ln1",state);
  const auto residual = state + multi_head_attention(parameters,"encoder/attention",normalized,normalized,2).output;
  const auto expanded = dense(parameters,"encoder/ff1",layer_norm(parameters,"encoder/ln2",residual),
                              48,Activation::GeluTanh);
  const auto encoded = residual + dense(parameters,"encoder/ff2",expanded,24);
  const auto features = mean(layer_norm(parameters,"norm",encoded),{1});
  const auto hidden = dense(parameters,"head/up",features,48,Activation::GeluTanh);
  const auto bottleneck = dense(parameters,"head/down",hidden,16);
  const auto prototypes = parameters.get("prototypes",{16,32},init::normal(.04));
  const auto unit = [](const Tensor& x, std::int64_t axis) {
    return x*rsqrt(maximum(sum(square(x),{axis},true),scalar(1e-12,x.dtype())));
  };
  return {features,matmul(unit(bottleneck,1),unit(prototypes,0))};
}

/// @brief Generate two global crops and one local crop resized to MNIST geometry.
/// @param pixels CPU Float32 [batch,28,28,1] with intensities in [0,1].
/// @param random Engine advanced in view order and then image order.
/// @return Three nearest-neighbor-resized crops with independent contrast factors.
/// @throws std::invalid_argument For incompatible shape, dtype or backend.
/// @par Example
/// @code
/// std::mt19937 random(7);
/// const auto views = dm::nn::mnist_dino_crops(dm::nn::zeros({2,28,28,1}),random);
/// @endcode
inline std::array<Tensor,3> mnist_dino_crops(const Tensor& pixels, std::mt19937& random) {
  if (!pixels.defined() || pixels.dim()!=4 || pixels.size(0)<1 || pixels.size(1)!=28 ||
      pixels.size(2)!=28 || pixels.size(3)!=1 || pixels.dtype()!=DType::Float32 ||
      pixels.device().backend!=compute::Backend::CPU)
    throw std::invalid_argument("mnist_dino_crops: expected Float32 CPU [batch,28,28,1]");
  const auto values = pixels.to_vector<float>();
  std::array<Tensor,3> result;
  std::vector<float> crop(values.size());
  for (int view=0;view<3;++view) {
    for (std::size_t offset=0;offset<values.size();offset+=784) {
      const int side = view<2 ? 24+random()%5 : 14+random()%7;
      const int y0 = random()%(29-side), x0 = random()%(29-side);
      const float contrast = .85f+.01f*(random()%31);
      for (int y=0;y<28;++y)
        for (int x=0;x<28;++x) {
          const auto source = offset+28*(y0+y*side/28)+x0+x*side/28;
          const auto intensity = std::round(255.f*values[source]);
          crop[offset+y*28+x] = std::min(1.f,contrast*intensity/255.f);
        }
    }
    result[view]=from_vector(crop,pixels.shape());
  }
  return result;
}

/// @brief Average DINO cross-entropy over the four mismatched teacher and student views.
/// @param student Three [batch,prototypes] arrays in global, global, local order.
/// @param teacher Two identically shaped global score arrays. Targets are detached.
/// @param center Prototype score center [prototypes], detached during evaluation.
/// @param student_temperature Positive softmax temperature, default 0.1.
/// @param teacher_temperature Positive softmax temperature, default 0.04.
/// @return Scalar student objective.
/// @throws std::invalid_argument For incompatible score shapes or temperatures.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto x=n::zeros({2,4});
/// const auto loss=n::dino_cross_view_loss({x,x,x},{x,x},n::zeros({4}));
/// @endcode
/// @par References
/// - Caron et al. (2021), Emerging Properties in Self-Supervised Vision Transformers,
///   https://arxiv.org/abs/2104.14294.
inline Tensor dino_cross_view_loss(const std::array<Tensor,3>& student,
                                  const std::array<Tensor,2>& teacher, const Tensor& center,
                                  double student_temperature=.1, double teacher_temperature=.04) {
  if (!student[0].defined() || student[0].dim()!=2 || student[0].numel()==0 ||
      !center.defined() || center.shape()!=Shape{student[0].size(1)} ||
      !(student_temperature>0 && std::isfinite(student_temperature) &&
        teacher_temperature>0 && std::isfinite(teacher_temperature)))
    throw std::invalid_argument("dino_cross_view_loss: invalid scores, center or temperatures");
  for (const auto& logits : student)
    if (!logits.defined() || logits.shape()!=student[0].shape())
      throw std::invalid_argument("dino_cross_view_loss: incompatible student views");
  for (const auto& logits : teacher)
    if (!logits.defined() || logits.shape()!=student[0].shape())
      throw std::invalid_argument("dino_cross_view_loss: incompatible teacher views");
  std::array<Tensor,3> logprob;
  for (int view=0;view<3;++view) logprob[view]=log_softmax(student[view]/student_temperature,-1);
  auto loss=scalar(0.,student[0].dtype());
  for (int global=0;global<2;++global) {
    const auto target=softmax((teacher[global].detach()-center.detach())/teacher_temperature,-1);
    for (int view=0;view<3;++view)
      if (view!=global) loss=loss-mean(sum(target*logprob[view],{-1}));
  }
  return loss*.25;
}

} // namespace datamunge::nn
