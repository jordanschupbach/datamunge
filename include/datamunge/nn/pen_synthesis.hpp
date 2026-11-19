/// @file
/// @brief Text-conditioned pen synthesis with a correlated Gaussian mixture.
#pragma once

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <array>

namespace datamunge::nn {

/// @brief Compute the two-component bivariate pen-density negative log-likelihood.
///
/// The first twelve outputs encode mixture logits, x and y means,
/// x and y softplus scales with floor 0.03, and correlations bounded
/// by 0.95. The final logit predicts a Bernoulli pen lift.
///
/// @param raw Floating [N,T,13] density outputs, N and T positive.
/// @param stroke Floating [N,T,3] x/y increments and binary pen-lift indicators.
/// @return Differentiable scalar mean negative log-likelihood.
/// @throws std::invalid_argument For incompatible shape or dtype.
/// @par Example
/// @code
/// const auto loss = dm::nn::pen_density_nll(dm::nn::zeros({2,12,13}), dm::nn::zeros({2,12,3}));
/// @endcode
/// @par References
/// - Graves (2013), https://arxiv.org/abs/1308.0850
/// @see text_conditioned_pen()
inline Tensor pen_density_nll(const Tensor &raw, const Tensor &stroke) {
  if (raw.dim()!=3 || raw.shape()[0]<1 || raw.shape()[1]<1 || raw.shape()[2]!=13 ||
      stroke.shape()!=Shape{raw.shape()[0],raw.shape()[1],3} ||
      !is_floating(raw.dtype()) || !is_floating(stroke.dtype()))
    throw std::invalid_argument("pen_density_nll: incompatible density or target shape");
  const auto c=[&](int lo,int hi){return slice_axis(raw,2,lo,hi);};
  const auto sx=.03+softplus(c(6,8)), sy=.03+softplus(c(8,10));
  const auto rho=.95*tanh(c(10,12));
  const auto x=(slice_axis(stroke,2,0,1)-c(2,4))/sx;
  const auto y=(slice_axis(stroke,2,1,2)-c(4,6))/sy;
  const auto variance=1.-square(rho);
  const auto quadratic=square(x-rho*y)/variance+square(y);
  const auto norm=log(sx)+log(sy)+1.8378770664093453+.5*log(variance);
  const auto pen=c(12,13);
  return mean(softplus(pen)-slice_axis(stroke,2,2,3)*pen -
              logsumexp(log_softmax(c(0,2),2)-norm-.5*quadratic,{2},true));
}

/// @brief Unroll a text-conditioned two-layer pen synthesizer.
///
/// With teacher strokes, feed the previous observed stroke back. With
/// explicit random draws, feed the previous sampled stroke back. In both
/// paths the single Gaussian attention center advances at each step.
///
/// @param model Owns the learned recurrent, window and density parameters.
/// @param text Integral [N,3] glyph IDs in [0,4).
/// @param input Floating [N,12,3] teacher strokes, or [N,12,4] draw inputs:
/// mixture uniform, two independent standard normals, and pen uniform.
/// @param sample Interpret input as draws and generate autoregressively when true.
/// @return Raw density outputs, window centers, text window vectors, and
/// sampled strokes in order. The last tensor is undefined in teacher mode.
/// @throws std::invalid_argument If geometry, token IDs or dtypes are incompatible.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto state = dm::nn::text_conditioned_pen(model,
///     dm::nn::zeros({2,3},dm::nn::DType::Int64), dm::nn::zeros({2,12,3}));
/// @endcode
/// @par References
/// - Graves (2013), https://arxiv.org/abs/1308.0850
/// @see pen_density_nll()
inline std::array<Tensor,4> text_conditioned_pen(ParameterStore &model,
    const Tensor &text, const Tensor &input, bool sample = false) {
  if(text.dim()!=2 || text.shape()[0]<1 || text.shape()[1]!=3 || !is_integral(text.dtype()) ||
     min(text).item()<0 || max(text).item()>=4 ||
     input.shape()!=Shape{text.shape()[0],12,sample?4:3} || input.dtype()!=model.dtype())
    throw std::invalid_argument("text_conditioned_pen: invalid prompt or recurrent inputs");
  const auto n=text.shape()[0];
  const auto dtype=input.dtype();
  const auto tokens=one_hot(text,4,1,0,dtype);
  auto first=lstm_zero_state(n,16,dtype), second=first;
  auto location=zeros({n,1},dtype), window=zeros({n,4},dtype), stroke=zeros({n,3},dtype);
  const auto positions=tensor({.75,1.75,2.75}).to(dtype);
  std::vector<Tensor> raws, locations, windows, strokes;
  for(int t=0;t<12;++t) {
    first=lstm_step(model,"first",concat({stroke,window},1),first);
    const auto control=dense(model,"window",first.h,3);
    const auto g=[&](int i){return slice_axis(control,1,i,i+1);};
    location=location+.25*exp(g(2));
    const auto w=exp(g(0)-exp(g(1))*square(location-positions));
    window=sum(tokens*expand_dims(w,2),{1});
    second=lstm_step(model,"second",concat({stroke,window,first.h},1),second);
    const auto raw=dense(model,"density",second.h,13);
    const auto row=select(input,1,t);
    if(sample) {
      const auto c=[&](int lo,int hi){return slice_axis(raw,1,lo,hi);};
      const auto noise=[&](int i){return slice_axis(row,1,i,i+1);};
      const auto left=less(noise(0),slice_axis(softmax(c(0,2),1),1,0,1));
      const auto choose=[&](const Tensor &x){return where(left,slice_axis(x,1,0,1),slice_axis(x,1,1,2));};
      const auto rho=choose(.95*tanh(c(10,12)));
      stroke=concat({choose(c(2,4))+choose(.03+softplus(c(6,8)))*noise(1),
          choose(c(4,6))+choose(.03+softplus(c(8,10)))*(rho*noise(1)+sqrt(1.-square(rho))*noise(2)),
          less(noise(3),sigmoid(c(12,13))).to(dtype)},1);
      strokes.push_back(stroke);
    } else {
      stroke=row;
    }
    raws.push_back(raw);
    locations.push_back(location);
    windows.push_back(window);
  }
  return {stack(raws,1),stack(locations,1),stack(windows,1),sample?stack(strokes,1):Tensor{}};
}

}  // namespace datamunge::nn
