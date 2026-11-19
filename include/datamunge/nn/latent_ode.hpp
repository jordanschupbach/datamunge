#pragma once

/// @file latent_ode.hpp
/// @brief Variational ODE-GRU recognition and irregular-time latent decoding.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Internal integration shared by recognition and decoding.
namespace latent_ode_detail {
/// @brief Integrate a named autonomous tanh field by one batched RK4 interval.
inline Tensor advance(ParameterStore& model,const std::string& name,const Tensor& x,const Tensor& dt) {
  const auto f=[&](const Tensor& z) {
    return dense(model,name+"/velocity",dense(model,name+"/hidden",z,12,Activation::Tanh),x.size(1));
  };
  const auto a=f(x), b=f(x+.5*dt*a), c=f(x+.5*dt*b), d=f(x+dt*c);
  return x+(dt*(1.f/6.f))*(a+2.*(b+c)+d);
}
} // namespace latent_ode_detail

/// @brief Latent states at the query times and the scalar readouts.
/// @par Example
/// @code
/// dm::nn::LatentOdeDecoded empty{};
/// @endcode
struct LatentOdeDecoded {
  Tensor prediction; ///< Scalar observations [batch,time].
  Tensor states; ///< Latent trajectories [batch,time,latent].
};

/// @brief Decode a specified initial state at irregular query times.
/// @param model Store for the dynamics and scalar observation head.
/// @param gaps Nonempty CPU matrix [batch,time] with nonnegative gaps. The first is ignored.
/// @param initial CPU matrix [batch,latent] at the first timestamp.
/// @return Latent states and observation predictions with live gradients.
/// @throws std::invalid_argument For incompatible input shape, dtype or device.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto path=dm::nn::latent_ode_decode(model,dm::nn::zeros({2,10}),dm::nn::zeros({2,4}));
/// @endcode
inline LatentOdeDecoded latent_ode_decode(ParameterStore& model,const Tensor& gaps,const Tensor& initial) {
  if (!gaps.defined() || !initial.defined() || gaps.dim()!=2 || initial.dim()!=2 ||
      !gaps.numel() || !initial.numel() || gaps.size(0)!=initial.size(0) ||
      gaps.dtype()!=model.dtype() || initial.dtype()!=model.dtype() ||
      gaps.device().backend!=compute::Backend::CPU || initial.device().backend!=compute::Backend::CPU)
    throw std::invalid_argument("latent_ode_decode: expected aligned gap and state matrices");
  auto state=initial;
  std::vector<Tensor> states,prediction;
  for(std::int64_t i=0;i<gaps.size(1);++i) {
    if(i) state=latent_ode_detail::advance(model,"dynamics",state,slice_axis(gaps,1,i,i+1));
    states.push_back(state);
    prediction.push_back(dense(model,"observation",state,1));
  }
  return {concat(prediction,1),stack(states,1)};
}

/// @brief Diagonal Gaussian posterior and its decoded trajectory.
/// @par Example
/// @code
/// dm::nn::LatentOdeOutput empty{};
/// @endcode
struct LatentOdeOutput {
  LatentOdeDecoded decoded; ///< Readouts and latent states.
  Tensor mean; ///< Initial posterior mean [batch,latent].
  Tensor log_variance; ///< Initial posterior log variance [batch,latent].
};

/// @brief Infer a latent initial distribution from a prefix and decode its sample.
/// @param model Store for recognition, posterior and decoder parameters.
/// @param gaps CPU matrix of nonnegative intervals [batch,time].
/// @param observed CPU prefix observations [batch,context], with context shorter than time.
/// @param noise CPU standardized latent samples [batch,latent]. Use zeros for the mean path.
/// @return Posterior parameters and decoded path with live parameter gradients.
/// @throws std::invalid_argument For invalid geometry, dtype, device or times.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto path=dm::nn::latent_ode(model,dm::nn::zeros({2,10}),
///                                  dm::nn::zeros({2,6}),dm::nn::zeros({2,4}));
/// @endcode
/// @par References
/// - Rubanova et al., Latent ODEs for Irregularly-Sampled Time Series, https://arxiv.org/abs/1907.03907.
inline LatentOdeOutput latent_ode(ParameterStore& model,const Tensor& gaps,const Tensor& observed,
                                 const Tensor& noise) {
  if (!gaps.defined() || !observed.defined() || !noise.defined() || gaps.dim()!=2 ||
      observed.dim()!=2 || noise.dim()!=2 || !gaps.numel() || !observed.numel() || !noise.numel() ||
      observed.size(0)!=gaps.size(0) || noise.size(0)!=gaps.size(0) || observed.size(1)>=gaps.size(1) ||
      observed.dtype()!=model.dtype() || gaps.dtype()!=model.dtype() || noise.dtype()!=model.dtype() ||
      gaps.device().backend!=compute::Backend::CPU || observed.device().backend!=compute::Backend::CPU ||
      noise.device().backend!=compute::Backend::CPU ||
      !all(logical_and(is_finite(gaps),greater_equal(gaps,0.))).item<bool>())
    throw std::invalid_argument("latent_ode: invalid prefix, gaps or latent noise");
  auto state=zeros({gaps.size(0),8},gaps.dtype());
  for(auto t=observed.size(1);t-->0;) {
    state=gru_step(model,"recognition",slice_axis(observed,1,t,t+1),state);
    if(t) state=latent_ode_detail::advance(model,"recognition_ode",state,-slice_axis(gaps,1,t,t+1));
  }
  const auto mean=dense(model,"mean",state,noise.size(1));
  const auto logvar=clip(dense(model,"logvar",state,noise.size(1)),-6.,3.);
  return {latent_ode_decode(model,gaps,mean+exp(.5*logvar)*noise),mean,logvar};
}

} // namespace datamunge::nn
