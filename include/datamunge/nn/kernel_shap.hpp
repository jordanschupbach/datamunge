#pragma once

/// @file kernel_shap.hpp
/// @brief Background replacement and anchored Shapley-kernel regression.

#include <bit>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Evaluate every feature coalition against a fixed background.
/// @tparam Predict Callable from a batch of rows to one floating score per row.
/// @param point Point to explain, shape [features].
/// @param background Replacement rows [rows,features] in point dtype.
/// @param predict Scorer returning [rows] or [rows,1]. Called once per coalition.
/// @return Coalition means in increasing bit-mask order, including both endpoints.
/// @throws std::invalid_argument For invalid shapes or more than twenty features.
/// @par Example
/// @code
/// auto v=dm::nn::shap_coalition_values(dm::nn::ones({2}),dm::nn::zeros({3,2}),
///     [](const auto& x){return dm::nn::sum(x,{1});});
/// @endcode
/// @see kernel_shap()
template<class Predict>
Tensor shap_coalition_values(const Tensor& point,const Tensor& background,Predict&& predict) {
  if (!point.defined() || !background.defined() || point.dim()!=1 || point.size(0)<1 ||
      point.size(0)>20 || background.dim()!=2 || background.size(0)<1 ||
      point.size(0)!=background.size(1) || point.dtype()!=background.dtype())
    throw std::invalid_argument("Invalid SHAP point or background");
  const auto features=point.size(0);
  std::vector<Tensor> values;
  std::vector<std::int32_t> mask(features);
  for (int id=0;id<(1<<features);++id) {
    for (int j=0;j<features;++j) mask[j]=(id>>j)&1;
    const auto scores=predict(where(from_vector(mask).to(DType::Bool),point,background));
    if (!scores.defined() || (scores.shape()!=Shape{background.size(0)} &&
                             scores.shape()!=Shape{background.size(0),1}))
      throw std::invalid_argument("Invalid SHAP predictor scores");
    values.push_back(mean(scores.to(DType::Float64)));
  }
  return stack(values);
}

/// @brief Fit Shapley-kernel coefficients with exact empty and full anchors.
/// The sampled interior masks must span the reduced design.
/// @param values Coalition means indexed by bit mask, length 2^features.
/// @param interior Nonempty interior mask IDs to fit, in any order.
/// @return Double-precision feature attributions whose sum is full minus empty.
/// @throws std::invalid_argument For invalid dimensions or mask IDs.
/// @par Example
/// @code
/// auto phi=dm::nn::kernel_shap(dm::nn::from_vector(std::vector<double>{0,1,2,3}),{1,2});
/// @endcode
/// @par References
/// - Lundberg and Lee (2017), A Unified Approach to Interpreting Model Predictions.
///   https://arxiv.org/abs/1705.07874
/// @see shap_coalition_values()
inline Tensor kernel_shap(const Tensor& values,const std::vector<int>& interior) {
  if (!values.defined() || values.dim()!=1 || values.size(0)<4 || values.size(0)>(1<<20) ||
      !std::has_single_bit(static_cast<unsigned>(values.size(0))) || interior.empty())
    throw std::invalid_argument("Invalid Kernel SHAP coalition table");
  const auto count=values.size(0);
  const int features=std::countr_zero(static_cast<unsigned>(count));
  const auto v=values.to_vector<double>();
  const auto total=v.back()-v.front();
  std::vector<double> a,y,w;
  for (int id:interior) {
    if (id<=0 || id>=count-1) throw std::invalid_argument("Expected an interior SHAP mask");
    const int last=(id>>(features-1))&1;
    for (int j=0;j<features-1;++j) a.push_back(((id>>j)&1)-last);
    y.push_back(v[id]-v.front()-last*total);
    const int size=std::popcount(static_cast<unsigned>(id));
    double combinations=1;
    for (int j=1;j<=size;++j) combinations=combinations*(features-j+1)/j;
    w.push_back((features-1)/(combinations*size*(features-size)));
  }
  const auto rows=static_cast<std::int64_t>(interior.size());
  const auto design=from_vector(a,{rows,features-1});
  const auto mass=from_vector(w,{rows,1});
  const auto target=from_vector(y,{rows,1});
  const auto transposed=transpose(design);
  const auto beta=solve(matmul(transposed,mass*design),matmul(transposed,mass*target)).reshape({features-1});
  return concat({beta,reshape(total-sum(beta),{1})},0);
}

}  // namespace datamunge::nn
