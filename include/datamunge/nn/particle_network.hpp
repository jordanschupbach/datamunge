#pragma once
/// @file particle_network.hpp
/// @brief Shared distance-conditioned pair forces on a complete particle graph.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Predict particle accelerations by summing antisymmetric pair messages.
/// Each relative displacement is scaled by the same tanh network of squared
/// distance. Forces are translation invariant and rotation and permutation equivariant.
/// Their sum is zero up to floating-point error.
/// @param model Store for the shared hidden and edge readout weights.
/// @param positions CPU floating tensor [batch,particles,coordinates].
/// @param width Positive hidden width, default 16.
/// @return Accelerations with the same shape and live parameter gradients.
/// @throws std::invalid_argument For incompatible positions or width.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto force=dm::nn::particle_pair_acceleration(model,dm::nn::zeros({2,5,2}));
/// @endcode
inline Tensor particle_pair_acceleration(ParameterStore& model,const Tensor& positions,
                                         std::int64_t width=16) {
  if (!positions.defined() || positions.dim()!=3 || !positions.numel() ||
      positions.dtype()!=model.dtype() || positions.device().backend!=compute::Backend::CPU ||
      width<1)
    throw std::invalid_argument("particle_pair_acceleration: invalid position tensor or width");
  const auto displacement=expand_dims(positions,1)-expand_dims(positions,2);
  const auto distance=sum(square(displacement),{3},true);
  const auto hidden=dense(model,"edge_hidden",distance,width,Activation::Tanh);
  const auto strength=dense(model,"edge_force",hidden,1);
  return sum(displacement*strength,{2});
}

} // namespace datamunge::nn
