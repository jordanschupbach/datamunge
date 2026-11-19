#pragma once

/// @file sindy_autoencoder.hpp
/// @brief Sparse polynomial latent dynamics with directional derivatives.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <cmath>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Endpoints and derivative diagnostics of a scalar SINDy autoencoder.
/// @par Example
/// @code
/// const dm::nn::SindyAutoencoderOutput empty{};
/// @endcode
struct SindyAutoencoderOutput {
  Tensor latent; ///< Encoded coordinate [batch,1].
  Tensor latent_dx; ///< Chain-rule derivative inferred from the input tangent.
  Tensor sindy_dx; ///< Cubic-library prediction of the latent derivative.
  Tensor prediction; ///< Decoded observation.
  Tensor prediction_dx; ///< Decoded derivative using the polynomial latent derivative.
  Tensor coefficients; ///< Effective weights in the order 1, z, z², z³.
  Tensor loss; ///< Reconstruction, two derivative errors and L1 penalty.
};

/// @brief Evaluate a scalar SINDy autoencoder and its joint training objective.
/// @param model Store owning the encoder, decoder and polynomial coefficients.
/// @param x Floating CPU observations [batch,signals].
/// @param dx Observed tangents with the same shape and dtype as x.
/// @param mask Optional CPU coefficient weights [4,1]. Undefined enables all terms.
/// @param l1 Finite nonnegative sparsity multiplier, default 0.002.
/// @return Live outputs with gradients through the encoder and decoder tangents.
/// @throws std::invalid_argument For incompatible geometry, dtype, device or penalty.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto x=dm::nn::zeros({2,4});
/// const auto step=dm::nn::sindy_autoencoder(model,x,x);
/// @endcode
/// @par References
/// - Champion et al., Data-driven discovery of coordinates and governing equations,
///   https://arxiv.org/abs/1904.02107.
inline SindyAutoencoderOutput sindy_autoencoder(ParameterStore& model,const Tensor& x,const Tensor& dx,
                                                const Tensor& mask={},double l1=.002) {
  if (!x.defined() || !dx.defined() || x.dim()!=2 || !x.numel() || x.shape()!=dx.shape() ||
      x.dtype()!=model.dtype() || dx.dtype()!=x.dtype() || x.device().backend!=compute::Backend::CPU ||
      dx.device().backend!=compute::Backend::CPU || !(l1>=0 && std::isfinite(l1)) ||
      (mask.defined() && (mask.shape()!=Shape{4,1} || mask.dtype()!=x.dtype() ||
                          mask.device().backend!=compute::Backend::CPU)))
    throw std::invalid_argument("sindy_autoencoder: invalid observations, tangent or mask");
  const auto mlp=[&](const std::string& name,const Tensor& value,const Tensor& tangent,std::int64_t width) {
    constexpr int hidden=16;
    const auto in=value.size(1);
    const auto w=model.get(name+"/w",{in,hidden},init::uniform(std::sqrt(6./(in+hidden))));
    const auto b=model.get(name+"/b",{hidden},init::zeros());
    const auto v=model.get(name+"/v",{hidden,width},init::uniform(std::sqrt(6./(hidden+width))));
    const auto c=model.get(name+"/c",{width},init::zeros());
    const auto h=tanh(matmul(value,w)+b);
    return std::pair{matmul(h,v)+c,matmul((1.-square(h))*matmul(tangent,w),v)};
  };
  const auto [z,dz]=mlp("encoder",x,dx,1);
  const auto raw=model.get("coefficient",{4,1},init::zeros());
  const auto coefficient=mask.defined()?raw*mask:raw;
  const auto z2=square(z);
  const auto dynamics=matmul(concat({ones_like(z),z,z2,z*z2},1),coefficient);
  const auto [prediction,tangent]=mlp("decoder",z,dynamics,x.size(1));
  const auto loss=mean(square(prediction-x))+mean(square(tangent-dx))+
                  mean(square(dynamics-dz))+l1*sum(abs(coefficient));
  return {z,dz,dynamics,prediction,tangent,coefficient,loss};
}

} // namespace datamunge::nn
