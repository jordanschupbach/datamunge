#pragma once

/// @file fourier_operator.hpp
/// @brief Resolution-independent one-dimensional Fourier neural operator.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <cmath>
#include <numbers>

namespace datamunge::nn {

/// @brief Apply a two-block Fourier operator to periodic scalar fields.
/// Each block mixes low-frequency complex channels, adds a pointwise
/// affine path and applies tanh. The same parameters work on different
/// grids as long as the grid supports all requested modes.
/// @param model Parameter store for spectral and pointwise weights.
/// @param input Uniformly sampled periodic CPU fields [batch,grid].
/// @param modes Positive retained frequency count including DC. Nyquist is excluded.
/// @param width Positive channel count.
/// @return Predicted scalar fields [batch,grid] with live parameter gradients.
/// @throws std::invalid_argument For incompatible geometry, dtype or mode count.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto out=dm::nn::fourier_operator_1d(model,dm::nn::ones({2,32}));
/// @endcode
/// @par References
/// - Li et al., Fourier Neural Operator for Parametric Partial Differential Equations,
///   https://arxiv.org/abs/2010.08895.
inline Tensor fourier_operator_1d(ParameterStore& model,const Tensor& input,
                                 std::int64_t modes=6,std::int64_t width=8) {
  if (!input.defined() || input.dim()!=2 || !input.numel() || input.dtype()!=model.dtype() ||
      input.device().backend!=compute::Backend::CPU || width<1 || modes<1 ||
      modes>1+(input.size(1)-1)/2)
    throw std::invalid_argument("fourier_operator_1d: incompatible fields or retained modes");
  const auto length=input.size(1);
  std::vector<double> cosine(length*modes),sine(cosine.size());
  std::vector<double> inverse_cos(cosine.size()),inverse_sin(cosine.size());
  for(std::int64_t t=0;t<length;++t)
    for(std::int64_t f=0;f<modes;++f) {
      const double angle=2*std::numbers::pi*t*f/length,scale=f?2.:1.;
      cosine[t*modes+f]=std::cos(angle)/length;
      sine[t*modes+f]=-std::sin(angle)/length;
      inverse_cos[f*length+t]=scale*std::cos(angle);
      inverse_sin[f*length+t]=-scale*std::sin(angle);
    }
  const auto basis=[&](const std::vector<double>& data,const Shape& shape) {
    return cast(from_vector(data,shape),input.dtype());
  };
  const auto c=basis(cosine,{length,modes}), s=basis(sine,{length,modes});
  const auto rc=basis(inverse_cos,{modes,length}), rs=basis(inverse_sin,{modes,length});
  auto state=dense(model,"lift",expand_dims(input,-1),width);
  for(int layer=0;layer<2;++layer) {
    const std::string name="spectral"+std::to_string(layer);
    const auto transposed=state.transpose({0,2,1});
    const auto re=matmul(transposed,c).transpose({2,0,1});
    const auto im=matmul(transposed,s).transpose({2,0,1});
    const auto a=model.get(name+"/real",{modes,width,width},init::normal(.11));
    const auto b=model.get(name+"/imag",{modes,width,width},init::normal(.11));
    const auto mix_re=(matmul(re,a)-matmul(im,b)).transpose({1,2,0});
    const auto mix_im=(matmul(re,b)+matmul(im,a)).transpose({1,2,0});
    const auto spectral=(matmul(mix_re,rc)+matmul(mix_im,rs)).transpose({0,2,1});
    state=tanh(spectral+dense(model,name+"/point",state,width));
  }
  return squeeze(dense(model,"head",state,1),{2});
}

} // namespace datamunge::nn
