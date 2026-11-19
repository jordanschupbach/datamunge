#pragma once

/// @file federated_average.hpp
/// @brief Sample-count-weighted aggregation of client model tensors.

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Average compatible client tensors weighted by their sample counts.
/// @param clients Nonempty sequence of model tensors with identical shape and floating dtype.
/// @param counts Nonnegative sample counts in client order, with positive total.
/// @return Weighted mean in the client tensor dtype.
/// @throws std::invalid_argument For invalid counts or incompatible client tensors.
/// @par Example
/// @code
/// auto merged=dm::nn::federated_average({dm::nn::zeros({2}),dm::nn::ones({2})},{1,3});
/// @endcode
/// @par References
/// - McMahan et al. (2017), Communication-Efficient Learning of Deep Networks from
///   Decentralized Data. https://proceedings.mlr.press/v54/mcmahan17a.html
inline Tensor federated_average(const std::vector<Tensor>& clients,
                                 const std::vector<std::int64_t>& counts) {
  if (clients.empty() || clients.size()!=counts.size() || !clients.front().defined())
    throw std::invalid_argument("Invalid federated averaging inputs");
  const auto& first=clients.front();
  if (first.dtype()!=DType::Float32 && first.dtype()!=DType::Float64)
    throw std::invalid_argument("Client parameters must be floating-point");
  long double total=0;
  for (std::size_t i=0;i<clients.size();++i) {
    if (counts[i]<0 || !clients[i].defined() || clients[i].shape()!=first.shape() ||
        clients[i].dtype()!=first.dtype())
      throw std::invalid_argument("Invalid client shape, dtype or sample count");
    total+=counts[i];
  }
  if (total==0) throw std::invalid_argument("Federated sample count must be positive");
  auto result=zeros_like(first);
  for (std::size_t i=0;i<clients.size();++i)
    if (counts[i]>0) result=result+clients[i]*static_cast<double>(counts[i]/total);
  return result;
}

}  // namespace datamunge::nn
