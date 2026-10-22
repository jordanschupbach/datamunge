#pragma once

/// @file convert.hpp
/// @brief Conversions between `nn::Tensor` and `linalg::Tensor`.

#include "datamunge/linalg/tensor.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// Copies a Float64 or Bool `linalg::Tensor` (String tensors are rejected).
Tensor from_linalg(const linalg::Tensor& tensor);

/// Copies into a `linalg::Tensor`: bool stays Bool, everything else becomes Float64.
linalg::Tensor to_linalg(const Tensor& tensor);

}  // namespace datamunge::nn
